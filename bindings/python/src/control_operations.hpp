/*
 * Copyright © 2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>

#include "akreon_native/akreon_native.h"
#include "control_operation_specs.hpp"
#include "control_shape_validation.hpp"
#include "ndarray_bridge.hpp"
#include "solver_config_bridge.hpp"
#include "solver_result_bridge.hpp"

namespace akreon_native::python {

inline void append_finite_metric(std::ostringstream& message, const char* name,
                                 double value) {
  if (std::isfinite(value)) message << ", " << name << "=" << value;
}

inline void throw_on_status(ak_native_status status, const char* workflow_name,
                            const ak_native_solver_result_f64& native_result,
                            const std::string& problem_context = {}) {
  if (status == AK_NATIVE_OK) return;

  std::ostringstream message;
  message << workflow_name << " solve failed: ";
  if (status == AK_NATIVE_UNSUPPORTED_DIMENSION) {
    message << "DIMENSION_MISMATCH (native status UNSUPPORTED_DIMENSION)";
    if (!problem_context.empty()) message << "; received " << problem_context;
  } else if (status == AK_NATIVE_NUMERICAL_ERROR ||
             status == AK_NATIVE_SOLVER_FAILED) {
    message << (status == AK_NATIVE_NUMERICAL_ERROR ? "NUMERICAL_ERROR"
                                                    : "SOLVER_FAILED")
            << "; event="
            << ak_native_numerical_event_name(
                   native_result.numerical_event_code)
            << "; solver_status="
            << ak_native_solver_status_name(native_result.solver_status_code)
            << "; iterations=" << native_result.num_iter
            << "; stalled=" << (native_result.stalled != 0 ? "true" : "false");
    append_finite_metric(message, "gap", native_result.normalized_duality_gap);
    append_finite_metric(message, "primal_residual",
                         native_result.primal_residual_norm);
    append_finite_metric(message, "dual_residual",
                         native_result.dual_residual_norm);
    append_finite_metric(message, "equality_residual",
                         native_result.equality_residual_norm);
    if (!problem_context.empty()) message << "; problem=" << problem_context;
  } else {
    message << ak_native_status_name(status);
    if (!problem_context.empty()) message << "; received " << problem_context;
  }
  throw nb::value_error(message.str().c_str());
}

inline std::string feedback_context(const Array2D& A, const Array2D& B) {
  std::ostringstream os;
  os << "A.shape=" << shapeN<2>(A) << ", B.shape=" << shapeN<2>(B)
     << " (u=" << B.shape(1) << ")";
  return os.str();
}

inline std::string observer_context(const Array2D& A, const Array2D& C) {
  std::ostringstream os;
  os << "A.shape=" << shapeN<2>(A) << ", C.shape=" << shapeN<2>(C)
     << " (y=" << C.shape(0) << ")";
  return os.str();
}

inline std::string generalized_feedback_context(const Array2D& A,
                                                const Array2D& B,
                                                const Array2D& C,
                                                const Array2D& D,
                                                size_t exogenous_dim) {
  std::ostringstream os;
  os << "A.shape=" << shapeN<2>(A) << ", B.shape=" << shapeN<2>(B)
     << ", C.shape=" << shapeN<2>(C) << ", D.shape=" << shapeN<2>(D)
     << " with [w,u] -> z dimensions (w=" << exogenous_dim
     << ", u=" << B.shape(1) - exogenous_dim << ", z=" << C.shape(0) << ")";
  return os.str();
}

inline std::string generalized_observer_context(const Array2D& A,
                                                const Array2D& B,
                                                const Array2D& C,
                                                const Array2D& D,
                                                size_t performance_dim) {
  std::ostringstream os;
  os << "A.shape=" << shapeN<2>(A) << ", B.shape=" << shapeN<2>(B)
     << ", C.shape=" << shapeN<2>(C) << ", D.shape=" << shapeN<2>(D)
     << " with w -> [z,y] dimensions (w=" << B.shape(1)
     << ", z=" << performance_dim << ", y=" << C.shape(0) - performance_dim
     << ")";
  return os.str();
}

inline std::string polytopic_context(const Array3D& A, const Array3D& B,
                                     const Array3D& C, const Array3D& D,
                                     const std::string& channels) {
  std::ostringstream os;
  os << "A.shape=" << shapeN<3>(A) << ", B.shape=" << shapeN<3>(B)
     << ", C.shape=" << shapeN<3>(C) << ", D.shape=" << shapeN<3>(D) << ", "
     << channels;
  return os.str();
}

inline StateFeedbackResult finish_feedback_result(
    nb::object K_object, const ak_native_solver_result_f64& native_result) {
  StateFeedbackResult result{.K = K_object};
  fill_common_result(native_result, result);
  return result;
}

inline StateObserverResult finish_observer_result(
    nb::object L_object, const ak_native_solver_result_f64& native_result) {
  StateObserverResult result{.L = L_object};
  fill_common_result(native_result, result);
  return result;
}

inline StateFeedbackResult solve_state_feedback(const Array2D& A,
                                                const Array2D& B,
                                                const SolverConfig& config) {
  constexpr const char* op = "solve_state_feedback";
  validate_state_feedback_shapes(A, B, op);

  auto K_object = make_numpy_empty(B.shape(1), A.shape(0));
  const auto K = nb::cast<MutableArray2D>(K_object);
  ak_native_solver_config_f64 native_config{};
  const auto* config_ptr = native_config_ptr(config, native_config);
  ak_native_solver_result_f64 native_result{};
  ak_native_status status{};
  {
    nb::gil_scoped_release release;
    status = ak_state_feedback_solve_f64(view_from_array(A), view_from_array(B),
                                         config_ptr, mut_view_from_array(K),
                                         &native_result);
  }
  throw_on_status(status, "state feedback", native_result,
                  feedback_context(A, B));
  return finish_feedback_result(K_object, native_result);
}

inline StateObserverResult solve_state_observer(const Array2D& A,
                                                const Array2D& C,
                                                const SolverConfig& config) {
  constexpr const char* op = "solve_state_observer";
  validate_state_observer_shapes(A, C, op);

  auto L_object = make_numpy_empty(A.shape(0), C.shape(0));
  const auto L = nb::cast<MutableArray2D>(L_object);
  ak_native_solver_config_f64 native_config{};
  const auto* config_ptr = native_config_ptr(config, native_config);
  ak_native_solver_result_f64 native_result{};
  ak_native_status status{};
  {
    nb::gil_scoped_release release;
    status = ak_state_observer_solve_f64(view_from_array(A), view_from_array(C),
                                         config_ptr, mut_view_from_array(L),
                                         &native_result);
  }
  throw_on_status(status, "state observer", native_result,
                  observer_context(A, C));
  return finish_observer_result(L_object, native_result);
}

template <typename Op>
StateFeedbackResult solve_generalized_state_feedback(
    const Array2D& A, const Array2D& B, const Array2D& C, const Array2D& D,
    size_t exogenous_dim, const SolverConfig& config) {
  validate_generalized_state_feedback_shapes(A, B, C, D, exogenous_dim,
                                             Op::python_name);

  auto K_object = make_numpy_empty(B.shape(1) - exogenous_dim, A.shape(0));
  const auto K = nb::cast<MutableArray2D>(K_object);
  ak_native_solver_config_f64 native_config{};
  const auto* config_ptr = native_config_ptr(config, native_config);
  ak_native_solver_result_f64 native_result{};
  ak_native_status status{};
  {
    nb::gil_scoped_release release;
    status = Op::call(view_from_array(A), view_from_array(B),
                      view_from_array(C), view_from_array(D), exogenous_dim,
                      config_ptr, mut_view_from_array(K), &native_result);
  }
  throw_on_status(status, Op::workflow_name, native_result,
                  generalized_feedback_context(A, B, C, D, exogenous_dim));
  return finish_feedback_result(K_object, native_result);
}

template <typename Op>
StateObserverResult solve_generalized_state_observer(
    const Array2D& A, const Array2D& B, const Array2D& C, const Array2D& D,
    size_t performance_dim, const SolverConfig& config) {
  validate_generalized_state_observer_shapes(A, B, C, D, performance_dim,
                                             Op::python_name);

  auto L_object = make_numpy_empty(A.shape(0), C.shape(0) - performance_dim);
  const auto L = nb::cast<MutableArray2D>(L_object);
  ak_native_solver_config_f64 native_config{};
  const auto* config_ptr = native_config_ptr(config, native_config);
  ak_native_solver_result_f64 native_result{};
  ak_native_status status{};
  {
    nb::gil_scoped_release release;
    status = Op::call(view_from_array(A), view_from_array(B),
                      view_from_array(C), view_from_array(D), performance_dim,
                      config_ptr, mut_view_from_array(L), &native_result);
  }
  throw_on_status(status, Op::workflow_name, native_result,
                  generalized_observer_context(A, B, C, D, performance_dim));
  return finish_observer_result(L_object, native_result);
}

template <typename Op>
StateFeedbackResult solve_hx_polytopic_state_feedback(
    const Array3D& A, const Array3D& B, const Array3D& C, const Array3D& D,
    size_t exogenous_dim, const SolverConfig& config) {
  validate_hx_polytopic_state_feedback_shapes(A, B, C, D, exogenous_dim,
                                              Op::python_name);

  auto K_object = make_numpy_empty(B.shape(2) - exogenous_dim, A.shape(1));
  const auto K = nb::cast<MutableArray2D>(K_object);
  auto A_views = vertex_views_from_array(A);
  auto B_views = vertex_views_from_array(B);
  auto C_views = vertex_views_from_array(C);
  auto D_views = vertex_views_from_array(D);
  ak_native_solver_config_f64 native_config{};
  const auto* config_ptr = native_config_ptr(config, native_config);
  ak_native_solver_result_f64 native_result{};
  ak_native_status status{};
  {
    nb::gil_scoped_release release;
    status = Op::call(A_views.data(), B_views.data(), C_views.data(),
                      D_views.data(), A.shape(0), exogenous_dim, config_ptr,
                      mut_view_from_array(K), &native_result);
  }
  throw_on_status(
      status, Op::workflow_name, native_result,
      polytopic_context(
          A, B, C, D,
          "[w,u] -> z dimensions (vertices=" + std::to_string(A.shape(0)) +
              ", w=" + std::to_string(exogenous_dim) +
              ", u=" + std::to_string(B.shape(2) - exogenous_dim) +
              ", z=" + std::to_string(C.shape(1)) + ")"));
  return finish_feedback_result(K_object, native_result);
}

template <typename Op>
StateObserverResult solve_hx_polytopic_state_observer(
    const Array3D& A, const Array3D& B, const Array3D& C, const Array3D& D,
    size_t performance_dim, const SolverConfig& config) {
  validate_hx_polytopic_state_observer_shapes(A, B, C, D, performance_dim,
                                              Op::python_name);

  auto L_object = make_numpy_empty(A.shape(1), C.shape(1) - performance_dim);
  const auto L = nb::cast<MutableArray2D>(L_object);
  auto A_views = vertex_views_from_array(A);
  auto B_views = vertex_views_from_array(B);
  auto C_views = vertex_views_from_array(C);
  auto D_views = vertex_views_from_array(D);
  ak_native_solver_config_f64 native_config{};
  const auto* config_ptr = native_config_ptr(config, native_config);
  ak_native_solver_result_f64 native_result{};
  ak_native_status status{};
  {
    nb::gil_scoped_release release;
    status = Op::call(A_views.data(), B_views.data(), C_views.data(),
                      D_views.data(), A.shape(0), performance_dim, config_ptr,
                      mut_view_from_array(L), &native_result);
  }
  throw_on_status(
      status, Op::workflow_name, native_result,
      polytopic_context(
          A, B, C, D,
          "w -> [z,y] dimensions (vertices=" + std::to_string(A.shape(0)) +
              ", w=" + std::to_string(B.shape(2)) +
              ", z=" + std::to_string(performance_dim) +
              ", y=" + std::to_string(C.shape(1) - performance_dim) + ")"));
  return finish_observer_result(L_object, native_result);
}

inline StateFeedbackResult solve_polytopic_state_feedback(
    const Array3D& A, const Array3D& B, const SolverConfig& config,
    const nb::object& p_weights, const nb::object& y_weights,
    const nb::object& quadratic_P, const nb::object& quadratic_q) {
  constexpr const char* op = "solve_polytopic_state_feedback";
  validate_polytopic_state_feedback_shapes(A, B, op);
  if ((!p_weights.is_none() || !y_weights.is_none()) &&
      (!quadratic_P.is_none() || !quadratic_q.is_none())) {
    fail_shape(op, "linear and quadratic objectives cannot be mixed");
  }
  if (p_weights.is_none() != y_weights.is_none()) {
    fail_shape(op, "p_weights and y_weights must be passed together");
  }
  if (quadratic_P.is_none() != quadratic_q.is_none()) {
    fail_shape(op, "quadratic_P and quadratic_q must be passed together");
  }

  auto K_object = make_numpy_empty(B.shape(2), A.shape(1));
  const auto K = nb::cast<MutableArray2D>(K_object);
  auto A_views = vertex_views_from_array(A);
  auto B_views = vertex_views_from_array(B);
  const ak_native_polytopic_state_feedback_objective_f64 objective{
      .p_weights = optional_view(p_weights, "p_weights"),
      .y_weights = optional_view(y_weights, "y_weights"),
      .quadratic_P = optional_view(quadratic_P, "quadratic_P"),
      .quadratic_q = optional_view(quadratic_q, "quadratic_q"),
  };
  ak_native_solver_config_f64 native_config{};
  const auto* config_ptr = native_config_ptr(config, native_config);
  ak_native_solver_result_f64 native_result{};
  ak_native_status status{};
  {
    nb::gil_scoped_release release;
    status = ak_polytopic_state_feedback_solve_f64(
        A_views.data(), B_views.data(), A.shape(0), &objective, config_ptr,
        mut_view_from_array(K), &native_result);
  }
  {
    std::ostringstream context;
    context << "A.shape=" << shapeN<3>(A) << ", B.shape=" << shapeN<3>(B)
            << " (vertices=" << A.shape(0) << ", u=" << B.shape(2) << ")";
    throw_on_status(status, "polytopic state feedback", native_result,
                    context.str());
  }
  return finish_feedback_result(K_object, native_result);
}

inline StateObserverResult solve_polytopic_state_observer(
    const Array3D& A, const Array3D& C, const SolverConfig& config,
    const nb::object& quadratic_P, const nb::object& quadratic_q) {
  constexpr const char* op = "solve_polytopic_state_observer";
  validate_polytopic_state_observer_shapes(A, C, op);
  if (quadratic_P.is_none() != quadratic_q.is_none()) {
    fail_shape(op, "quadratic_P and quadratic_q must be passed together");
  }

  auto L_object = make_numpy_empty(A.shape(1), C.shape(1));
  const auto L = nb::cast<MutableArray2D>(L_object);
  auto A_views = vertex_views_from_array(A);
  auto C_views = vertex_views_from_array(C);
  const ak_native_polytopic_state_observer_objective_f64 objective{
      .quadratic_P = optional_view(quadratic_P, "quadratic_P"),
      .quadratic_q = optional_view(quadratic_q, "quadratic_q"),
  };
  ak_native_solver_config_f64 native_config{};
  const auto* config_ptr = native_config_ptr(config, native_config);
  ak_native_solver_result_f64 native_result{};
  ak_native_status status{};
  {
    nb::gil_scoped_release release;
    status = ak_polytopic_state_observer_solve_f64(
        A_views.data(), C_views.data(), A.shape(0), &objective, config_ptr,
        mut_view_from_array(L), &native_result);
  }
  {
    std::ostringstream context;
    context << "A.shape=" << shapeN<3>(A) << ", C.shape=" << shapeN<3>(C)
            << " (vertices=" << A.shape(0) << ", y=" << C.shape(1) << ")";
    throw_on_status(status, "polytopic state observer", native_result,
                    context.str());
  }
  return finish_observer_result(L_object, native_result);
}

template <typename Op>
void bind_generalized_state_feedback_op(nb::module_& m) {
  m.def(
      Op::python_name,
      [](const Array2D& A, const Array2D& B, const Array2D& C, const Array2D& D,
         size_t exogenous_dim, const SolverConfig& config) {
        return solve_generalized_state_feedback<Op>(A, B, C, D, exogenous_dim,
                                                    config);
      },
      "A"_a, "B"_a, "C"_a, "D"_a, "exogenous_dim"_a,
      "config"_a = SolverConfig{});
}

template <typename Op>
void bind_generalized_state_observer_op(nb::module_& m) {
  m.def(
      Op::python_name,
      [](const Array2D& A, const Array2D& B, const Array2D& C, const Array2D& D,
         size_t performance_dim, const SolverConfig& config) {
        return solve_generalized_state_observer<Op>(A, B, C, D, performance_dim,
                                                    config);
      },
      "A"_a, "B"_a, "C"_a, "D"_a, "performance_dim"_a,
      "config"_a = SolverConfig{});
}

template <typename Op>
void bind_polytopic_state_feedback_op(nb::module_& m) {
  m.def(
      Op::python_name,
      [](const Array3D& A, const Array3D& B, const Array3D& C, const Array3D& D,
         size_t exogenous_dim, const SolverConfig& config) {
        return solve_hx_polytopic_state_feedback<Op>(A, B, C, D, exogenous_dim,
                                                     config);
      },
      "A"_a, "B"_a, "C"_a, "D"_a, "exogenous_dim"_a,
      "config"_a = SolverConfig{});
}

template <typename Op>
void bind_polytopic_state_observer_op(nb::module_& m) {
  m.def(
      Op::python_name,
      [](const Array3D& A, const Array3D& B, const Array3D& C, const Array3D& D,
         size_t performance_dim, const SolverConfig& config) {
        return solve_hx_polytopic_state_observer<Op>(A, B, C, D,
                                                     performance_dim, config);
      },
      "A"_a, "B"_a, "C"_a, "D"_a, "performance_dim"_a,
      "config"_a = SolverConfig{});
}

inline void bind_state_feedback_operation(nb::module_& m) {
  m.def("solve_state_feedback", &solve_state_feedback, "A"_a, "B"_a,
        "config"_a = SolverConfig{});
}

inline void bind_state_observer_operation(nb::module_& m) {
  m.def("solve_state_observer", &solve_state_observer, "A"_a, "C"_a,
        "config"_a = SolverConfig{});
}

inline void bind_polytopic_state_feedback_operation(nb::module_& m) {
  m.def("solve_polytopic_state_feedback", &solve_polytopic_state_feedback,
        "A"_a, "B"_a, "config"_a = SolverConfig{}, "p_weights"_a = nb::none(),
        "y_weights"_a = nb::none(), "quadratic_P"_a = nb::none(),
        "quadratic_q"_a = nb::none());
}

inline void bind_polytopic_state_observer_operation(nb::module_& m) {
  m.def("solve_polytopic_state_observer", &solve_polytopic_state_observer,
        "A"_a, "C"_a, "config"_a = SolverConfig{}, "quadratic_P"_a = nb::none(),
        "quadratic_q"_a = nb::none());
}

inline void bind_direct_operations(nb::module_& m) {
  bind_state_feedback_operation(m);
  bind_state_observer_operation(m);
  bind_polytopic_state_feedback_operation(m);
  bind_polytopic_state_observer_operation(m);
}

inline void bind_generalized_operations(nb::module_& m) {
  bind_generalized_state_feedback_op<H2StateFeedbackOp>(m);
  bind_generalized_state_feedback_op<HinfStateFeedbackOp>(m);
  bind_generalized_state_observer_op<H2StateObserverOp>(m);
  bind_generalized_state_observer_op<HinfStateObserverOp>(m);
  bind_polytopic_state_feedback_op<H2PolytopicStateFeedbackOp>(m);
  bind_polytopic_state_feedback_op<HinfPolytopicStateFeedbackOp>(m);
  bind_polytopic_state_observer_op<H2PolytopicStateObserverOp>(m);
  bind_polytopic_state_observer_op<HinfPolytopicStateObserverOp>(m);
}

} // namespace akreon_native::python
