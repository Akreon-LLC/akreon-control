/*
 * Copyright © 2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * @brief Shared ABI, matrix, objective, and solver-config helpers.
 *
 * Supported dimensions are declared in supported_dimensions.hpp. Runtime
 * dispatch selects those static template instances without package generation.
 */

#include "akreon_native/controls.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>

#include <akreon/control/control.hpp>
#include <akreon/linalg/linalg.hpp>
#include <akreon/convex/convex.hpp>

namespace akreon_native::controls_detail {

/**
 * @brief Compile-time list of supported runtime dimensions.
 *
 * @tparam Values dimension values compiled into this binary
 */
template <size_t... Values>
struct SupportedDims {};

/** @brief An explicitly compiled dimension tuple. */
template <size_t... Dims>
struct SupportedShape {
  static constexpr std::array<size_t, sizeof...(Dims)> values{Dims...};
  template <typename Fn>
  static ak_native_status invoke(Fn& fn) {
    return fn.template operator()<Dims...>();
  }
};
template <typename... Shapes>
struct ShapeList {};

#include "supported_dimensions.hpp"

inline constexpr size_t kCompiledPdIpmSteps = PD_IPM_STEPS;

/**
 * @brief Dispatch a runtime dimension to a callable static template branch.
 *
 * @tparam Values compiled dimension values
 * @tparam Fn callable with `fn.template operator()<Dim>()`
 *
 * @param dims compile-time supported dimension list
 * @param value runtime dimension value
 * @param fn static-dimension branch callable
 *
 * @return branch status, or unsupported-dimension status
 */
template <size_t... Values, typename Fn>
constexpr ak_native_status dispatch_dimension(SupportedDims<Values...> dims,
                                              size_t value, Fn&& fn) {
  static_cast<void>(dims);
  ak_native_status status = AK_NATIVE_UNSUPPORTED_DIMENSION;
  bool matched = false;
  ((value == Values
        ? (status = fn.template operator()<Values>(), matched = true, 0)
        : 0),
   ...);
  return matched ? status : AK_NATIVE_UNSUPPORTED_DIMENSION;
}

template <typename DimList>
constexpr bool supports_dimension(DimList dims, size_t value) {
  return dispatch_dimension(dims, value, []<size_t Dim>() {
           static_cast<void>(Dim);
           return AK_NATIVE_OK;
         }) == AK_NATIVE_OK;
}

/**
 * @brief Reset solver diagnostics to neutral values.
 *
 * @param result solver diagnostics output
 */
inline void reset_result(ak_native_solver_result_f64* result) {
  *result = ak_native_solver_result_f64{};
  result->normalized_duality_gap = std::numeric_limits<double>::quiet_NaN();
  result->primal_objective_value = std::numeric_limits<double>::quiet_NaN();
  result->dual_objective_value = std::numeric_limits<double>::quiet_NaN();
  result->dual_residual_norm = std::numeric_limits<double>::quiet_NaN();
  result->primal_residual_norm = std::numeric_limits<double>::quiet_NaN();
  result->equality_residual_norm = std::numeric_limits<double>::quiet_NaN();
  result->objective_value = std::numeric_limits<double>::quiet_NaN();
  result->gamma = std::numeric_limits<double>::quiet_NaN();
}

template <typename Cone>
struct NativeCanonicalConeKind {
  static constexpr uint32_t value = AK_NATIVE_CANONICAL_CONE_UNKNOWN;
};

template <linalg::VectorContainer Vec>
struct NativeCanonicalConeKind<conic::NonnegOrthantCone<Vec>> {
  static constexpr uint32_t value = AK_NATIVE_CANONICAL_CONE_NONNEG_ORTHANT;
};

template <linalg::VectorContainer Vec>
struct NativeCanonicalConeKind<conic::SecondOrderCone<Vec>> {
  static constexpr uint32_t value = AK_NATIVE_CANONICAL_CONE_SECOND_ORDER;
};

template <linalg::VectorContainer Vec, size_t N>
struct NativeCanonicalConeKind<conic::PositiveSemidefiniteCone<Vec, N>> {
  static constexpr uint32_t value =
      AK_NATIVE_CANONICAL_CONE_POSITIVE_SEMIDEFINITE;
};

template <linalg::VectorContainer Vec>
struct NativeCanonicalConeKind<conic::ExpCone<Vec>> {
  static constexpr uint32_t value = AK_NATIVE_CANONICAL_CONE_EXP;
};

template <linalg::VectorContainer Vec>
struct NativeCanonicalConeKind<conic::PowerCone<Vec>> {
  static constexpr uint32_t value = AK_NATIVE_CANONICAL_CONE_POWER;
};

template <typename Cone>
inline constexpr uint32_t native_canonical_cone_kind_v =
    NativeCanonicalConeKind<linalg::remove_cvref_t<Cone>>::value;

inline void append_canonical_record(ak_native_solver_result_f64* result,
                                    const ak_native_canonical_record& record) {
  if (result->canonical_trace_record_count >=
      AK_NATIVE_CANONICAL_TRACE_MAX_RECORDS) {
    result->canonical_trace_truncated = 1;
    return;
  }
  result->canonical_trace_records[result->canonical_trace_record_count++] =
      record;
}

inline void write_canonical_root_record(ak_native_solver_result_f64* result) {
  append_canonical_record(result,
                          ak_native_canonical_record{
                              .record_kind = AK_NATIVE_CANONICAL_RECORD_NODE,
                              .atom_type = 0,
                              .cone_kind = AK_NATIVE_CANONICAL_CONE_UNKNOWN,
                              .node_id = 0,
                              .parent_id = 0,
                              .local_index = 0,
                              .global_index = 0,
                              .offset = 0,
                              .dim = 0,
                          });
}

template <typename Cone>
inline void write_canonical_cone_record(const Cone&, size_t local_index,
                                        size_t& offset,
                                        ak_native_solver_result_f64* result) {
  constexpr size_t dim = conic::cone_dim_t<Cone>;
  append_canonical_record(result,
                          ak_native_canonical_record{
                              .record_kind = AK_NATIVE_CANONICAL_RECORD_CONE,
                              .atom_type = 0,
                              .cone_kind = native_canonical_cone_kind_v<Cone>,
                              .node_id = 0,
                              .parent_id = 0,
                              .local_index = local_index,
                              .global_index = local_index,
                              .offset = offset,
                              .dim = dim,
                          });
  offset += dim;
}

inline void write_canonical_equality_record(
    size_t row_count, ak_native_solver_result_f64* result) {
  if (row_count == 0) return;
  append_canonical_record(
      result, ak_native_canonical_record{
                  .record_kind = AK_NATIVE_CANONICAL_RECORD_EQUALITY,
                  .atom_type = 0,
                  .cone_kind = AK_NATIVE_CANONICAL_CONE_UNKNOWN,
                  .node_id = 0,
                  .parent_id = 0,
                  .local_index = 0,
                  .global_index = 0,
                  .offset = 0,
                  .dim = row_count,
              });
}

template <typename Problem>
inline void write_canonical_trace(const Problem& problem,
                                  ak_native_solver_result_f64* result) {
  result->canonical_trace_record_count = 0;
  result->canonical_trace_truncated = 0;

  write_canonical_root_record(result);

  size_t offset = 0;
  size_t index = 0;
  std::apply(
      [&](const auto&... cones) {
        (write_canonical_cone_record(cones, index++, offset, result), ...);
      },
      problem.constraint_cones());

  using EqMap = decltype(problem.equality_map());
  using EqMat = affine::affine_mat_t<EqMap>;
  write_canonical_equality_record(linalg::mat_rows_v<EqMat>, result);
}

template <typename Vec>
inline size_t copy_vector_values(const Vec& vec, double* out, size_t capacity,
                                 int32_t* truncated) {
  constexpr size_t N = linalg::vector_dim_v<Vec>;
  const size_t count = (N < capacity) ? N : capacity;
  for (size_t i = 0; i < count; ++i) out[i] = static_cast<double>(vec(i));
  *truncated = N > capacity ? 1 : 0;
  return count;
}

template <typename Result>
inline void write_canonical_vectors(const Result& res,
                                    ak_native_solver_result_f64* result) {
  result->cone_value_count = copy_vector_values(
      res.iterate.x, result->cone_primal_values,
      AK_NATIVE_CANONICAL_VECTOR_MAX_VALUES, &result->cone_values_truncated);
  copy_vector_values(res.iterate.y, result->cone_dual_values,
                     result->cone_value_count, &result->cone_values_truncated);
  copy_vector_values(res.rp, result->cone_residual_values,
                     result->cone_value_count, &result->cone_values_truncated);

  result->equality_value_count =
      copy_vector_values(res.iterate.nu, result->equality_dual_values,
                         AK_NATIVE_CANONICAL_VECTOR_MAX_VALUES,
                         &result->equality_values_truncated);
  copy_vector_values(res.req, result->equality_residual_values,
                     result->equality_value_count,
                     &result->equality_values_truncated);
}

/**
 * @brief Copy Akreon Convex primal-dual result diagnostics into the native ABI
 * result.
 *
 * Workflow-level fields such as `objective_value` and `gamma` remain owned by
 * the caller-specific control synthesis wrapper. This helper only mirrors the
 * solver's final primal-dual state.
 *
 * @tparam Result concrete Akreon Convex PrimalDualResult type
 *
 * @param res Akreon Convex primal-dual result
 * @param result native mutable solver result
 */
template <typename Result>
inline void write_primal_dual_result(const Result& res,
                                     ak_native_solver_result_f64* result) {
  result->solver_status_code = static_cast<int32_t>(res.tag);
  result->numerical_event_code = static_cast<int32_t>(res.event);
  result->stalled = res.stalled ? 1 : 0;
  result->num_iter = res.num_iter;
  result->normalized_duality_gap = static_cast<double>(res.mu);
  result->primal_objective_value = static_cast<double>(res.convex_obj);
  result->dual_objective_value = static_cast<double>(res.dual_obj);
  result->dual_residual_norm =
      static_cast<double>(linalg::L2<decltype(res.rd)>()(res.rd));
  result->primal_residual_norm =
      static_cast<double>(linalg::L2<decltype(res.rp)>()(res.rp));
  result->equality_residual_norm =
      static_cast<double>(linalg::L2<decltype(res.req)>()(res.req));
  write_canonical_vectors(res, result);
}

/**
 * @brief Copy the composed native solver artifact set to C ABI outputs.
 *
 * The structured canonical trace and final primal/dual/equality vectors are
 * always part of the solver result.
 *
 * @tparam Result concrete Akreon Convex PrimalDualResult type
 *
 * @param res Akreon Convex primal-dual result
 * @param result native mutable solver result
 */
template <typename Result, typename Problem>
inline void write_solver_result(const Result& res, const Problem& problem,
                                ak_native_solver_result_f64* result) {
  write_primal_dual_result(res, result);
  write_canonical_trace(problem, result);
}

/**
 * @brief Forward declaration for native solver config validation.
 */
inline bool invalid_config(const ak_native_solver_config_f64* config);

/**
 * @brief Solve the scalar continuous-time state-feedback special case.
 *
 * The current generic PSD cone path requires matrix dimensions greater than
 * one, so the one-state case is handled analytically.
 *
 * @param A_view scalar dynamics matrix view
 * @param B_view scalar-state control matrix view
 * @param K gain output view
 * @param result solver diagnostics output
 * @return native status code
 */
inline ak_native_status solve_scalar_state_feedback(
    const ak_native_matrix_view_f64& A_view,
    const ak_native_matrix_view_f64& B_view, const ak_native_matrix_mut_f64& K,
    ak_native_solver_result_f64* result) {
  const double a = A_view.data[0];
  double norm_sq = 0.0;
  for (size_t j = 0; j < B_view.cols; ++j) {
    const double b = B_view.data[j];
    norm_sq += b * b;
  }

  if (norm_sq == 0.0) {
    if (a < 0.0) {
      for (size_t j = 0; j < B_view.cols; ++j) K.data[j * K.stride] = 0.0;
      result->solver_status_code =
          static_cast<int32_t>(::config::SolverResult::OPTIMAL);
      result->numerical_event_code =
          static_cast<int32_t>(linalg::NumericalEvent::NONE);
      result->stalled = 0;
      result->num_iter = 1;
      return AK_NATIVE_OK;
    }
    result->solver_status_code =
        static_cast<int32_t>(::config::SolverResult::INFEASIBLE_OR_UNBOUNDED);
    result->numerical_event_code =
        static_cast<int32_t>(linalg::NumericalEvent::NONE);
    result->stalled = 1;
    result->num_iter = 1;
    return AK_NATIVE_SOLVER_FAILED;
  }

  const double scale = (-1.0 - a) / norm_sq;
  for (size_t j = 0; j < B_view.cols; ++j) {
    K.data[j * K.stride] = scale * B_view.data[j];
  }
  result->solver_status_code =
      static_cast<int32_t>(::config::SolverResult::OPTIMAL);
  result->numerical_event_code =
      static_cast<int32_t>(linalg::NumericalEvent::NONE);
  result->stalled = 0;
  result->num_iter = 1;
  result->objective_value = std::numeric_limits<double>::quiet_NaN();
  result->gamma = std::numeric_limits<double>::quiet_NaN();
  return AK_NATIVE_OK;
}

/**
 * @brief Check that all entries in a borrowed matrix view are finite.
 *
 * @param M borrowed matrix view
 *
 * @return true when every logical entry is finite
 */
inline bool finite_matrix(const ak_native_matrix_view_f64& M) {
  if (M.data == nullptr) return false;
  for (size_t i = 0; i < M.rows; ++i) {
    for (size_t j = 0; j < M.cols; ++j) {
      if (!std::isfinite(M.data[i * M.stride + j])) return false;
    }
  }
  return true;
}

/**
 * @brief Copy a borrowed row-major matrix view into an Akreon Linalg static
 * matrix.
 *
 * @tparam Rows static row count
 * @tparam Cols static column count
 *
 * @param M borrowed matrix view
 *
 * @return Akreon Linalg matrix value
 */
template <size_t Rows, size_t Cols>
linalg::Matrix<double, Rows, Cols> read_matrix(
    const ak_native_matrix_view_f64& M) {
  linalg::Matrix<double, Rows, Cols> out{};
  for (size_t i = 0; i < Rows; ++i) {
    for (size_t j = 0; j < Cols; ++j) out(i, j) = M.data[i * M.stride + j];
  }
  return out;
}

/**
 * @brief Copy a borrowed row or column vector view into an Akreon Linalg static
 * vector.
 *
 * @tparam Dim static vector length
 *
 * @param v borrowed row or column vector view
 *
 * @return Akreon Linalg vector value
 */
template <size_t Dim>
linalg::Vector<double, Dim> read_vector(const ak_native_matrix_view_f64& v) {
  linalg::Vector<double, Dim> out{};
  for (size_t i = 0; i < Dim; ++i) {
    const size_t row = v.cols == 1 ? i : 0;
    const size_t col = v.cols == 1 ? 0 : i;
    out(i) = v.data[row * v.stride + col];
  }
  return out;
}

/**
 * @brief Return the leading decision-vector block with a fixed static length.
 *
 * @tparam Dim requested leading block length
 * @tparam Vec Akreon Linalg vector type
 *
 * @param v Akreon Convex decision vector
 *
 * @return either the original vector or its leading block
 */
template <size_t Dim, typename Vec>
auto leading_decision_vector(const Vec& v) {
  if constexpr (linalg::vector_dim_v<Vec> == Dim) {
    return v;
  } else {
    return std::get<0>(linalg::splice<Dim>(v));
  }
}

/**
 * @brief Return true when a borrowed matrix view has storage.
 *
 * @param view borrowed matrix view
 *
 * @return true when `view.data` is non-null
 */
constexpr bool has_matrix_view(const ak_native_matrix_view_f64& view) {
  return view.data != nullptr;
}

/**
 * @brief Return true when a borrowed matrix view is the canonical empty view.
 *
 * @param view borrowed matrix view
 *
 * @return true when data, shape, and stride are all empty
 */
constexpr bool is_empty_matrix_view(const ak_native_matrix_view_f64& view) {
  return view.data == nullptr && view.rows == 0 && view.cols == 0 &&
         view.stride == 0;
}

/**
 * @brief Compute a linear objective value for a static decision block.
 *
 * @tparam Dim static vector length
 *
 * @param weights objective weights
 * @param z decision vector
 *
 * @return dot product `weights' * z`
 */
template <size_t Dim>
double dot_objective(const linalg::Vector<double, Dim>& weights,
                     const linalg::Vector<double, Dim>& z) {
  double value = 0.0;
  for (size_t i = 0; i < Dim; ++i) value += weights(i) * z(i);
  return value;
}

/**
 * @brief Compute a quadratic objective value for a static decision block.
 *
 * @tparam Dim static vector length
 *
 * @param P quadratic objective matrix
 * @param q linear objective vector
 * @param z decision vector
 *
 * @return `0.5 * z' * P * z + q' * z`
 */
template <size_t Dim>
double quadratic_objective(const linalg::Matrix<double, Dim, Dim>& P,
                           const linalg::Vector<double, Dim>& q,
                           const linalg::Vector<double, Dim>& z) {
  double value = 0.0;
  for (size_t i = 0; i < Dim; ++i) {
    value += q(i) * z(i);
    for (size_t j = 0; j < Dim; ++j) value += 0.5 * z(i) * P(i, j) * z(j);
  }
  return value;
}

/**
 * @brief Copy an Akreon Linalg static matrix into a borrowed mutable row-major
 * view.
 *
 * @tparam Rows static row count
 * @tparam Cols static column count
 *
 * @param M Akreon Linalg matrix value
 * @param out mutable output view
 */
template <size_t Rows, size_t Cols>
void write_matrix(const linalg::Matrix<double, Rows, Cols>& M,
                  ak_native_matrix_mut_f64 out) {
  for (size_t i = 0; i < Rows; ++i) {
    for (size_t j = 0; j < Cols; ++j) out.data[i * out.stride + j] = M(i, j);
  }
}

/**
 * @brief Convert optional C ABI solver config to Akreon Convex solver config.
 *
 * @param config optional borrowed native config
 *
 * @return Akreon Convex primal-dual IPM config
 */
inline config::PrimalDualIPMConfig<double> to_convex_config(
    const ak_native_solver_config_f64* config) {
  auto cfg = config::PrimalDualIPMConfig<double>{};
  cfg.steps = kCompiledPdIpmSteps;

  if (config != nullptr) {
    cfg.backtrack_steps = config->backtrack_steps;
    cfg.primal_feas_tol = config->primal_feas_tol;
    cfg.dual_feas_tol = config->dual_feas_tol;
    cfg.duality_gap_tol = config->duality_gap_tol;
    cfg.reg.global_eps = config->global_reg_eps;
    cfg.reg.eq_eps = config->eq_reg_eps;
  }
  return cfg;
}

/**
 * @brief Interpret a flattened decision block as a static row-major matrix.
 *
 * @tparam Rows matrix row count
 * @tparam Cols matrix column count
 * @tparam Vec Akreon Linalg vector type
 *
 * @param v flattened decision block
 *
 * @return static Akreon Linalg matrix value
 */
template <size_t Rows, size_t Cols, typename Vec>
linalg::Matrix<double, Rows, Cols> decision_block_to_matrix(const Vec& v) {
  linalg::Matrix<double, Rows, Cols> out{};
  for (size_t j = 0; j < Rows * Cols; ++j) { out(j / Cols, j % Cols) = v(j); }
  return out;
}

} // namespace akreon_native::controls_detail
