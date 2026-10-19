/*
 * Copyright © 2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * @brief Runtime ABI validation shared by the C entrypoints.
 *
 * Validation enforces the stable C ABI boundary: borrowed row-major f64 views,
 * finite data, output capacity, solver-compatible dimensions, and explicitly
 * compiled dimension support. It deliberately validates facade shapes rather
 * than exposing Akreon Convex solver internals.
 */

#include "util.hpp"

namespace akreon_native::controls_detail {

/**
 * @brief Generalized feedback ABI contract:
 *   A: N x N, B: N x (W + M), C: Z x N, D: Z x (W + M), K: M x N.
 * The caller passes W; M is derived to mirror the Akreon Control [exogenous,
 * control] input PortLayout.
 *
 * @param A borrowed dynamics matrix view
 * @param B borrowed combined exogenous/control input matrix view
 * @param C borrowed performance output matrix view
 * @param D borrowed performance feedthrough matrix view
 * @param w exogenous input dimension
 * @param K mutable state-feedback gain output view
 *
 * @return native status code
 */
inline ak_native_status validate_generalized_feedback(
    ak_native_matrix_view_f64 A, ak_native_matrix_view_f64 B,
    ak_native_matrix_view_f64 C, ak_native_matrix_view_f64 D, size_t w,
    ak_native_matrix_mut_f64 K) {
  if (A.data == nullptr || B.data == nullptr || C.data == nullptr ||
      D.data == nullptr || K.data == nullptr) {
    return AK_NATIVE_INVALID_ARGUMENT;
  }
  if (A.rows == 0 || A.cols == 0 || B.rows == 0 || B.cols == 0 || C.rows == 0 ||
      C.cols == 0 || D.rows == 0 || D.cols == 0) {
    return AK_NATIVE_INVALID_ARGUMENT;
  }
  if (A.rows != A.cols || B.rows != A.rows || C.cols != A.rows ||
      D.rows != C.rows || D.cols != B.cols) {
    return AK_NATIVE_INVALID_ARGUMENT;
  }
  if (w == 0 || w >= B.cols) return AK_NATIVE_INVALID_ARGUMENT;
  const size_t m = B.cols - w;
  if (A.stride < A.cols || B.stride < B.cols || C.stride < C.cols ||
      D.stride < D.cols) {
    return AK_NATIVE_INVALID_ARGUMENT;
  }
  if (K.rows < m || K.cols < A.rows || K.stride < K.cols) {
    return AK_NATIVE_OUTPUT_BUFFER_TOO_SMALL;
  }
  if (!supports_dimension(SupportedStateDims{}, A.rows) ||
      !supports_dimension(SupportedGeneralizedPortDims{}, w) ||
      !supports_dimension(SupportedGeneralizedPortDims{}, m) ||
      !supports_dimension(SupportedGeneralizedPortDims{}, C.rows)) {
    return AK_NATIVE_UNSUPPORTED_DIMENSION;
  }
  if (!finite_matrix(A) || !finite_matrix(B) || !finite_matrix(C) ||
      !finite_matrix(D)) {
    return AK_NATIVE_INVALID_ARGUMENT;
  }
  return AK_NATIVE_OK;
}

/**
 * @brief Generalized observer ABI contract:
 *   A: N x N, B: N x W, C: (Z + Y) x N,
 *   D: (Z + Y) x W, L: N x Y.
 * All inputs are exogenous. The caller passes Z to split outputs as [z, y].
 *
 * @param A borrowed dynamics matrix view
 * @param B borrowed exogenous input matrix view
 * @param C borrowed combined performance/measurement output matrix view
 * @param D borrowed feedthrough matrix view
 * @param z performance output dimension
 * @param L mutable observer gain output view
 *
 * @return native status code
 */
inline ak_native_status validate_generalized_observer(
    ak_native_matrix_view_f64 A, ak_native_matrix_view_f64 B,
    ak_native_matrix_view_f64 C, ak_native_matrix_view_f64 D, size_t z,
    ak_native_matrix_mut_f64 L) {
  if (A.data == nullptr || B.data == nullptr || C.data == nullptr ||
      D.data == nullptr || L.data == nullptr) {
    return AK_NATIVE_INVALID_ARGUMENT;
  }
  if (A.rows == 0 || A.cols == 0 || B.rows == 0 || B.cols == 0 || C.rows == 0 ||
      C.cols == 0 || D.rows == 0 || D.cols == 0) {
    return AK_NATIVE_INVALID_ARGUMENT;
  }
  if (A.rows != A.cols || B.rows != A.rows || C.cols != A.rows ||
      D.rows != C.rows || D.cols != B.cols) {
    return AK_NATIVE_INVALID_ARGUMENT;
  }
  if (z == 0 || z >= C.rows) return AK_NATIVE_INVALID_ARGUMENT;
  const size_t w = B.cols;
  const size_t y = C.rows - z;
  if (A.stride < A.cols || B.stride < B.cols || C.stride < C.cols ||
      D.stride < D.cols) {
    return AK_NATIVE_INVALID_ARGUMENT;
  }
  if (L.rows < A.rows || L.cols < y || L.stride < L.cols) {
    return AK_NATIVE_OUTPUT_BUFFER_TOO_SMALL;
  }
  if (!supports_dimension(SupportedStateDims{}, A.rows) ||
      !supports_dimension(SupportedGeneralizedPortDims{}, w) ||
      !supports_dimension(SupportedGeneralizedPortDims{}, z) ||
      !supports_dimension(SupportedGeneralizedPortDims{}, y)) {
    return AK_NATIVE_UNSUPPORTED_DIMENSION;
  }
  if (!finite_matrix(A) || !finite_matrix(B) || !finite_matrix(C) ||
      !finite_matrix(D)) {
    return AK_NATIVE_INVALID_ARGUMENT;
  }
  return AK_NATIVE_OK;
}

/**
 * @brief Direct feedback ABI contract:
 *   A: N x N, B: N x M, K: M x N.
 *
 * @param A borrowed dynamics matrix view
 * @param B borrowed control-input matrix view
 * @param K mutable state-feedback gain output view
 *
 * @return native status code
 */
inline ak_native_status validate_state_feedback(ak_native_matrix_view_f64 A,
                                                ak_native_matrix_view_f64 B,
                                                ak_native_matrix_mut_f64 K) {
  if (A.data == nullptr || B.data == nullptr || K.data == nullptr) {
    return AK_NATIVE_INVALID_ARGUMENT;
  }
  if (A.rows == 0 || A.cols == 0 || B.rows == 0 || B.cols == 0) {
    return AK_NATIVE_INVALID_ARGUMENT;
  }
  if (A.rows != A.cols || B.rows != A.rows) {
    return AK_NATIVE_INVALID_ARGUMENT;
  }
  if (A.stride < A.cols || B.stride < B.cols) {
    return AK_NATIVE_INVALID_ARGUMENT;
  }
  if (K.rows < B.cols || K.cols < A.rows || K.stride < K.cols) {
    return AK_NATIVE_OUTPUT_BUFFER_TOO_SMALL;
  }
  if (!supports_dimension(StateFeedbackStateDims{}, A.rows) ||
      !supports_dimension(StateFeedbackInputDims{}, B.cols)) {
    return AK_NATIVE_UNSUPPORTED_DIMENSION;
  }
  if (!finite_matrix(A) || !finite_matrix(B)) {
    return AK_NATIVE_INVALID_ARGUMENT;
  }
  return AK_NATIVE_OK;
}

/**
 * @brief Direct observer ABI contract:
 *   A: N x N, C: Y x N, L: N x Y.
 *
 * @param A borrowed dynamics matrix view
 * @param C borrowed measurement-output matrix view
 * @param L mutable observer gain output view
 *
 * @return native status code
 */
inline ak_native_status validate_state_observer(ak_native_matrix_view_f64 A,
                                                ak_native_matrix_view_f64 C,
                                                ak_native_matrix_mut_f64 L) {
  if (A.data == nullptr || C.data == nullptr || L.data == nullptr) {
    return AK_NATIVE_INVALID_ARGUMENT;
  }
  if (A.rows == 0 || A.cols == 0 || C.rows == 0 || C.cols == 0) {
    return AK_NATIVE_INVALID_ARGUMENT;
  }
  if (A.rows != A.cols || C.cols != A.rows) {
    return AK_NATIVE_INVALID_ARGUMENT;
  }
  if (A.stride < A.cols || C.stride < C.cols) {
    return AK_NATIVE_INVALID_ARGUMENT;
  }
  if (L.rows < A.rows || L.cols < C.rows || L.stride < L.cols) {
    return AK_NATIVE_OUTPUT_BUFFER_TOO_SMALL;
  }
  if (!supports_dimension(StateObserverStateDims{}, A.rows) ||
      !supports_dimension(StateObserverMeasurementDims{}, C.rows)) {
    return AK_NATIVE_UNSUPPORTED_DIMENSION;
  }
  if (!finite_matrix(A) || !finite_matrix(C)) {
    return AK_NATIVE_INVALID_ARGUMENT;
  }
  return AK_NATIVE_OK;
}

/**
 * @brief Generalized polytopic feedback reuses the nominal generalized-feedback
 * contract and requires every vertex to have exactly the same shape.
 *
 * @param A borrowed array of dynamics matrix views
 * @param B borrowed array of combined input matrix views
 * @param C borrowed array of performance output matrix views
 * @param D borrowed array of feedthrough matrix views
 * @param vertex_count number of vertices in each borrowed array
 * @param w exogenous input dimension
 * @param K mutable state-feedback gain output view
 *
 * @return native status code
 */
inline ak_native_status validate_polytopic_feedback(
    const ak_native_matrix_view_f64* A, const ak_native_matrix_view_f64* B,
    const ak_native_matrix_view_f64* C, const ak_native_matrix_view_f64* D,
    size_t vertex_count, size_t w, ak_native_matrix_mut_f64 K) {
  if (A == nullptr || B == nullptr || C == nullptr || D == nullptr) {
    return AK_NATIVE_INVALID_ARGUMENT;
  }
  if (vertex_count == 0) return AK_NATIVE_INVALID_ARGUMENT;
  if (!supports_dimension(SupportedPolytopeDims{}, vertex_count)) {
    return AK_NATIVE_UNSUPPORTED_DIMENSION;
  }
  const auto valid =
      validate_generalized_feedback(A[0], B[0], C[0], D[0], w, K);
  if (valid != AK_NATIVE_OK) return valid;
  for (size_t v = 1; v < vertex_count; ++v) {
    if (A[v].rows != A[0].rows || A[v].cols != A[0].cols ||
        B[v].rows != B[0].rows || B[v].cols != B[0].cols ||
        C[v].rows != C[0].rows || C[v].cols != C[0].cols ||
        D[v].rows != D[0].rows || D[v].cols != D[0].cols) {
      return AK_NATIVE_INVALID_ARGUMENT;
    }
    const auto vertex_valid =
        validate_generalized_feedback(A[v], B[v], C[v], D[v], w, K);
    if (vertex_valid != AK_NATIVE_OK) return vertex_valid;
  }
  return AK_NATIVE_OK;
}

/**
 * @brief Generalized polytopic observer reuses the nominal generalized-observer
 * contract and requires every vertex to have exactly the same shape.
 *
 * @param A borrowed array of dynamics matrix views
 * @param B borrowed array of exogenous input matrix views
 * @param C borrowed array of combined performance/measurement output views
 * @param D borrowed array of feedthrough matrix views
 * @param vertex_count number of vertices in each borrowed array
 * @param z performance output dimension
 * @param L mutable observer gain output view
 *
 * @return native status code
 */
inline ak_native_status validate_polytopic_observer(
    const ak_native_matrix_view_f64* A, const ak_native_matrix_view_f64* B,
    const ak_native_matrix_view_f64* C, const ak_native_matrix_view_f64* D,
    size_t vertex_count, size_t z, ak_native_matrix_mut_f64 L) {
  if (A == nullptr || B == nullptr || C == nullptr || D == nullptr) {
    return AK_NATIVE_INVALID_ARGUMENT;
  }
  if (vertex_count == 0) return AK_NATIVE_INVALID_ARGUMENT;
  if (!supports_dimension(SupportedPolytopeDims{}, vertex_count)) {
    return AK_NATIVE_UNSUPPORTED_DIMENSION;
  }
  const auto valid =
      validate_generalized_observer(A[0], B[0], C[0], D[0], z, L);
  if (valid != AK_NATIVE_OK) return valid;
  for (size_t v = 1; v < vertex_count; ++v) {
    if (A[v].rows != A[0].rows || A[v].cols != A[0].cols ||
        B[v].rows != B[0].rows || B[v].cols != B[0].cols ||
        C[v].rows != C[0].rows || C[v].cols != C[0].cols ||
        D[v].rows != D[0].rows || D[v].cols != D[0].cols) {
      return AK_NATIVE_INVALID_ARGUMENT;
    }
    const auto vertex_valid =
        validate_generalized_observer(A[v], B[v], C[v], D[v], z, L);
    if (vertex_valid != AK_NATIVE_OK) return vertex_valid;
  }
  return AK_NATIVE_OK;
}

/**
 * @brief Return true when a vector-like matrix view has a specific length.
 *
 * @param view borrowed row or column vector view
 * @param size expected vector length
 *
 * @return true when the view is finite and has the requested length
 */
inline bool vector_view_has_size(const ak_native_matrix_view_f64& view,
                                 size_t size) {
  return has_matrix_view(view) &&
         ((view.rows == size && view.cols == 1) ||
          (view.rows == 1 && view.cols == size)) &&
         view.stride >= view.cols && finite_matrix(view);
}

/**
 * @brief Return true when a matrix view has a specific finite shape.
 *
 * @param view borrowed matrix view
 * @param rows expected row count
 * @param cols expected column count
 *
 * @return true when the view is finite and has the requested shape
 */
inline bool matrix_view_has_shape(const ak_native_matrix_view_f64& view,
                                  size_t rows, size_t cols) {
  return has_matrix_view(view) && view.rows == rows && view.cols == cols &&
         view.stride >= view.cols && finite_matrix(view);
}

/**
 * @brief Return the packed symmetric-matrix dimension for an n by n matrix.
 *
 * @param n square matrix dimension
 *
 * @return packed symmetric storage length
 */
constexpr size_t smat_dim(size_t n) { return n * (n + 1) / 2; }

/**
 * @brief Polytopic feedback objective validation is separate from vertex
 * validation because the decision size depends on N/M after the vertices are
 * known.
 *
 * @param objective optional native objective
 * @param n state dimension
 * @param m control input dimension
 *
 * @return native status code
 */
inline ak_native_status validate_polytopic_feedback_objective(
    const ak_native_polytopic_state_feedback_objective_f64* objective, size_t n,
    size_t m) {
  if (objective == nullptr) return AK_NATIVE_OK;
  const bool has_p = has_matrix_view(objective->p_weights);
  const bool has_y = has_matrix_view(objective->y_weights);
  const bool has_P = has_matrix_view(objective->quadratic_P);
  const bool has_q = has_matrix_view(objective->quadratic_q);
  if (has_p != has_y || has_P != has_q) return AK_NATIVE_INVALID_ARGUMENT;
  if ((has_p || has_y) && (has_P || has_q)) return AK_NATIVE_INVALID_ARGUMENT;
  if ((!has_p && !is_empty_matrix_view(objective->p_weights)) ||
      (!has_y && !is_empty_matrix_view(objective->y_weights)) ||
      (!has_P && !is_empty_matrix_view(objective->quadratic_P)) ||
      (!has_q && !is_empty_matrix_view(objective->quadratic_q))) {
    return AK_NATIVE_INVALID_ARGUMENT;
  }
  const size_t p_dim = smat_dim(n);
  const size_t y_dim = m * n;
  const size_t z_dim = p_dim + y_dim;
  if (has_p && (!vector_view_has_size(objective->p_weights, p_dim) ||
                !vector_view_has_size(objective->y_weights, y_dim))) {
    return AK_NATIVE_INVALID_ARGUMENT;
  }
  if (has_P && (!matrix_view_has_shape(objective->quadratic_P, z_dim, z_dim) ||
                !vector_view_has_size(objective->quadratic_q, z_dim))) {
    return AK_NATIVE_INVALID_ARGUMENT;
  }
  return AK_NATIVE_OK;
}

/**
 * @brief Polytopic observer objective validation mirrors feedback but
 * uses the observer [Q, W] decision size.
 *
 * @param objective optional native objective
 * @param n state dimension
 * @param y measurement output dimension
 *
 * @return native status code
 */
inline ak_native_status validate_polytopic_observer_objective(
    const ak_native_polytopic_state_observer_objective_f64* objective, size_t n,
    size_t y) {
  if (objective == nullptr) return AK_NATIVE_OK;
  const bool has_P = has_matrix_view(objective->quadratic_P);
  const bool has_q = has_matrix_view(objective->quadratic_q);
  if (has_P != has_q) return AK_NATIVE_INVALID_ARGUMENT;
  if ((!has_P && !is_empty_matrix_view(objective->quadratic_P)) ||
      (!has_q && !is_empty_matrix_view(objective->quadratic_q))) {
    return AK_NATIVE_INVALID_ARGUMENT;
  }
  const size_t z_dim = smat_dim(n) + n * y;
  if (has_P && (!matrix_view_has_shape(objective->quadratic_P, z_dim, z_dim) ||
                !vector_view_has_size(objective->quadratic_q, z_dim))) {
    return AK_NATIVE_INVALID_ARGUMENT;
  }
  return AK_NATIVE_OK;
}

/**
 * @brief Polytopic feedback validates only A/B vertices plus the
 * optional objective; there is no disturbance/performance channel in this API.
 *
 * @param A borrowed array of dynamics matrix views
 * @param B borrowed array of control matrix views
 * @param vertex_count number of vertices in each borrowed array
 * @param objective optional native objective
 * @param K mutable state-feedback gain output view
 *
 * @return native status code
 */
inline ak_native_status validate_polytopic_feedback(
    const ak_native_matrix_view_f64* A, const ak_native_matrix_view_f64* B,
    size_t vertex_count,
    const ak_native_polytopic_state_feedback_objective_f64* objective,
    ak_native_matrix_mut_f64 K) {
  if (A == nullptr || B == nullptr) return AK_NATIVE_INVALID_ARGUMENT;
  if (vertex_count == 0) return AK_NATIVE_INVALID_ARGUMENT;
  if (!supports_dimension(SupportedPolytopeDims{}, vertex_count)) {
    return AK_NATIVE_UNSUPPORTED_DIMENSION;
  }
  if (A[0].data == nullptr || B[0].data == nullptr || K.data == nullptr) {
    return AK_NATIVE_INVALID_ARGUMENT;
  }
  if (A[0].rows == 0 || A[0].cols == 0 || B[0].rows == 0 || B[0].cols == 0) {
    return AK_NATIVE_INVALID_ARGUMENT;
  }
  if (A[0].rows != A[0].cols || B[0].rows != A[0].rows) {
    return AK_NATIVE_INVALID_ARGUMENT;
  }
  if (A[0].stride < A[0].cols || B[0].stride < B[0].cols) {
    return AK_NATIVE_INVALID_ARGUMENT;
  }
  if (K.rows < B[0].cols || K.cols < A[0].rows || K.stride < K.cols) {
    return AK_NATIVE_OUTPUT_BUFFER_TOO_SMALL;
  }
  if (!supports_dimension(PolytopicStateFeedbackStateDims{}, A[0].rows) ||
      !supports_dimension(PolytopicStateFeedbackInputDims{}, B[0].cols)) {
    return AK_NATIVE_UNSUPPORTED_DIMENSION;
  }
  if (!finite_matrix(A[0]) || !finite_matrix(B[0])) {
    return AK_NATIVE_INVALID_ARGUMENT;
  }
  for (size_t v = 1; v < vertex_count; ++v) {
    if (A[v].rows != A[0].rows || A[v].cols != A[0].cols ||
        B[v].rows != B[0].rows || B[v].cols != B[0].cols ||
        A[v].stride < A[v].cols || B[v].stride < B[v].cols ||
        !finite_matrix(A[v]) || !finite_matrix(B[v])) {
      return AK_NATIVE_INVALID_ARGUMENT;
    }
  }
  return validate_polytopic_feedback_objective(objective, A[0].rows, B[0].cols);
}

/**
 * @brief Polytopic observer validates only A/C vertices plus the
 * optional objective; the Akreon Control input matrix is fixed internally by
 * the policy.
 *
 * @param A borrowed array of dynamics matrix views
 * @param C borrowed array of measurement matrix views
 * @param vertex_count number of vertices in each borrowed array
 * @param objective optional native objective
 * @param L mutable observer gain output view
 *
 * @return native status code
 */
inline ak_native_status validate_polytopic_observer(
    const ak_native_matrix_view_f64* A, const ak_native_matrix_view_f64* C,
    size_t vertex_count,
    const ak_native_polytopic_state_observer_objective_f64* objective,
    ak_native_matrix_mut_f64 L) {
  if (A == nullptr || C == nullptr) return AK_NATIVE_INVALID_ARGUMENT;
  if (vertex_count == 0) return AK_NATIVE_INVALID_ARGUMENT;
  if (!supports_dimension(SupportedPolytopeDims{}, vertex_count)) {
    return AK_NATIVE_UNSUPPORTED_DIMENSION;
  }
  if (A[0].data == nullptr || C[0].data == nullptr || L.data == nullptr) {
    return AK_NATIVE_INVALID_ARGUMENT;
  }
  if (A[0].rows == 0 || A[0].cols == 0 || C[0].rows == 0 || C[0].cols == 0) {
    return AK_NATIVE_INVALID_ARGUMENT;
  }
  if (A[0].rows != A[0].cols || C[0].cols != A[0].rows) {
    return AK_NATIVE_INVALID_ARGUMENT;
  }
  if (A[0].stride < A[0].cols || C[0].stride < C[0].cols) {
    return AK_NATIVE_INVALID_ARGUMENT;
  }
  if (L.rows < A[0].rows || L.cols < C[0].rows || L.stride < L.cols) {
    return AK_NATIVE_OUTPUT_BUFFER_TOO_SMALL;
  }
  if (!supports_dimension(PolytopicStateObserverStateDims{}, A[0].rows) ||
      !supports_dimension(PolytopicStateObserverMeasurementDims{}, C[0].rows)) {
    return AK_NATIVE_UNSUPPORTED_DIMENSION;
  }
  if (!finite_matrix(A[0]) || !finite_matrix(C[0])) {
    return AK_NATIVE_INVALID_ARGUMENT;
  }
  for (size_t v = 1; v < vertex_count; ++v) {
    if (A[v].rows != A[0].rows || A[v].cols != A[0].cols ||
        C[v].rows != C[0].rows || C[v].cols != C[0].cols ||
        A[v].stride < A[v].cols || C[v].stride < C[v].cols ||
        !finite_matrix(A[v]) || !finite_matrix(C[v])) {
      return AK_NATIVE_INVALID_ARGUMENT;
    }
  }
  return validate_polytopic_observer_objective(objective, A[0].rows, C[0].rows);
}

/**
 * @brief Native config requires steps to match compiled PD_IPM_STEPS. That
 * keeps the runtime request consistent with the compiled solver budget.
 *
 * @param config optional native solver config
 *
 * @return true when the config is non-null and invalid
 */
inline bool invalid_config(const ak_native_solver_config_f64* config) {
  return config != nullptr &&
         (config->steps != kCompiledPdIpmSteps ||
          config->backtrack_steps == 0 ||
          !std::isfinite(config->primal_feas_tol) ||
          !std::isfinite(config->dual_feas_tol) ||
          !std::isfinite(config->duality_gap_tol) ||
          !std::isfinite(config->global_reg_eps) ||
          !std::isfinite(config->eq_reg_eps) ||
          config->primal_feas_tol <= 0.0 || config->dual_feas_tol <= 0.0 ||
          config->duality_gap_tol <= 0.0 || config->global_reg_eps < 0.0 ||
          config->eq_reg_eps < 0.0);
}

} // namespace akreon_native::controls_detail
