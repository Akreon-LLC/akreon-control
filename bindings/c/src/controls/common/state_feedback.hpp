/*
 * Copyright © 2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/** @brief Direct continuous-time state-feedback solve and dispatch. */

#include "dispatch.hpp"

namespace akreon_native::controls_detail {

/**
 * @brief Direct state-feedback policy for a control-only plant. The dispatch
 * layer selects N/M; this policy owns Akreon Control type construction and gain
 * recovery.
 */
struct StateFeedbackProblem {
  using StateDims = StateFeedbackStateDims;
  using InputDims = StateFeedbackInputDims;

  template <size_t N, size_t M>
  constexpr static ak_native_status solve(
      const ak_native_matrix_view_f64& A_view,
      const ak_native_matrix_view_f64& B_view,
      const ak_native_solver_config_f64* config,
      const ak_native_matrix_mut_f64& K_out,
      ak_native_solver_result_f64* result) {
    using StateMat = linalg::Matrix<double, N, N>;
    using InputMat = linalg::Matrix<double, N, M>;
    using OutputMat = linalg::Matrix<double, N, N>;
    using SS = control::StateSpace<control::TimeDomainType::CONTINUOUS,
                                   StateMat, InputMat, OutputMat>;
    using Inputs =
        control::PortLayout<control::Port<control::PortRole::CONTROL, 0, M>>;
    using Outputs = control::PortLayout<
        control::Port<control::PortRole::MEASUREMENT, 0, N>>;
    using Plant = control::PlantModel<SS, Inputs, Outputs>;
    using SF = control::StateFeedback<Plant>;

    /**
     * @brief Direct feedback observes the full state, so C is the identity and
     * the native ABI only needs A/B/K.
     */
    constexpr size_t ZDim = SF::ZDim;
    constexpr size_t PDim = SF::PDim;
    constexpr size_t YDim = SF::YDim;

    const auto A = read_matrix<N, N>(A_view);
    const auto B = read_matrix<N, M>(B_view);
    const auto C = linalg::make_identity<double, N>();
    const auto plant = Plant(SS(A, B, C));

    using ZSpace = linalg::RowVector<double, ZDim>;
    const auto Z = convex::Variable<convex::VarMeta<ZSpace>>();

    linalg::Matrix<double, ZDim, ZDim> P_obj{};
    for (size_t i = 0; i < ZDim; ++i) P_obj(i, i) = 1.0;
    linalg::Vector<double, ZDim> q_obj{};
    const auto obj = convex::make_quad_form(P_obj, q_obj, Z);
    const auto prob = control::make_state_feedback_problem(plant, Z, obj);

    using Solver = convex::PrimalDualIPM<decltype(prob)>;
    const Solver solver(prob, to_convex_config(config));
    const auto& res = solver.res;
    write_solver_result(res, prob, result);

    if (res.event != linalg::NumericalEvent::NONE) {
      return linalg::is_fatal_event(res.event) ? AK_NATIVE_NUMERICAL_ERROR
                                               : AK_NATIVE_SOLVER_FAILED;
    }
    if (res.stalled || res.tag != ::config::SolverResult::OPTIMAL) {
      return AK_NATIVE_SOLVER_FAILED;
    }

    /**
     * @brief The Akreon Convex decision vector is [P, Y]. P/Y are converted to
     * K through the controller helper so native does not duplicate gain
     * algebra.
     */
    const auto z_split = linalg::splice<PDim>(res.iterate.z);
    const auto P_res = linalg::make_smat<N>(std::get<0>(z_split));
    linalg::Matrix<double, M, N> Y_res{};
    for (size_t j = 0; j < YDim; ++j) {
      const size_t row = j / N;
      const size_t col = j % N;
      Y_res(row, col) = std::get<1>(z_split)(j);
    }

    const auto sf = SF();
    const auto [K_res, K_event] = sf.K(P_res, Y_res);
    result->numerical_event_code = static_cast<int32_t>(K_event);
    if (K_event != linalg::NumericalEvent::NONE)
      return AK_NATIVE_NUMERICAL_ERROR;

    write_matrix(K_res, K_out);
    return AK_NATIVE_OK;
  }
};

/**
 * @brief Dispatch runtime state dimension to a static template value.
 *
 * @param A borrowed dynamics matrix view
 * @param B borrowed control matrix view
 * @param config optional borrowed native config
 * @param K mutable gain output view
 * @param result solver diagnostics output
 * @return native status code
 */
constexpr ak_native_status dispatch_state_feedback(
    const ak_native_matrix_view_f64& A, const ak_native_matrix_view_f64& B,
    const ak_native_solver_config_f64* config,
    const ak_native_matrix_mut_f64& K, ak_native_solver_result_f64* result) {
  switch (A.rows) {
    case 1:
      static_cast<void>(config);
      return solve_scalar_state_feedback(A, B, K, result);
    default:
      return dispatch_dimensions(
          std::tuple{typename StateFeedbackProblem::StateDims{},
                     typename StateFeedbackProblem::InputDims{}},
          std::array<size_t, 2>{A.rows, B.cols}, [&]<size_t N, size_t M>() {
            return StateFeedbackProblem::template solve<N, M>(A, B, config, K,
                                                              result);
          });
  }
}

} // namespace akreon_native::controls_detail
