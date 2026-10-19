/*
 * Copyright © 2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/** @brief Direct Luenberger observer solve and dispatch. */

#include "dispatch.hpp"

namespace akreon_native::controls_detail {

/**
 * @brief Direct observer policy for a measurement-only plant. The dummy control
 * input exists only because the Akreon Control StateSpace type includes an
 * input matrix.
 */
struct StateObserverProblem {
  using StateDims = StateObserverStateDims;
  using MeasurementDims = StateObserverMeasurementDims;

  template <size_t N, size_t Y>
  static ak_native_status solve(const ak_native_matrix_view_f64& A_view,
                                const ak_native_matrix_view_f64& C_view,
                                const ak_native_solver_config_f64* config,
                                const ak_native_matrix_mut_f64& L_out,
                                ak_native_solver_result_f64* result) {
    using StateMat = linalg::Matrix<double, N, N>;
    using InputMat = linalg::Matrix<double, N, 1>;
    using OutputMat = linalg::Matrix<double, Y, N>;
    using SS = control::StateSpace<control::TimeDomainType::CONTINUOUS,
                                   StateMat, InputMat, OutputMat>;
    using Inputs =
        control::PortLayout<control::Port<control::PortRole::CONTROL, 0, 1>>;
    using Outputs = control::PortLayout<
        control::Port<control::PortRole::MEASUREMENT, 0, Y>>;
    using Plant = control::PlantModel<SS, Inputs, Outputs>;
    using Observer = control::LuenbergerObserver<Plant>;

    /**
     * @brief Direct observer synthesis consumes A/C. The B matrix is
     * intentionally zero-sized-by-contract from the caller perspective and
     * fixed internally.
     */
    const auto A = read_matrix<N, N>(A_view);
    const auto B = InputMat{};
    const auto C = read_matrix<Y, N>(C_view);
    const auto plant = Plant(SS(A, B, C));

    using ZSpace = linalg::RowVector<double, Observer::ZDim>;
    const auto Z = convex::Variable<convex::VarMeta<ZSpace>>();

    linalg::Matrix<double, Observer::ZDim, Observer::ZDim> P_obj{};
    for (size_t i = 0; i < Observer::ZDim; ++i) P_obj(i, i) = 1.0;
    linalg::Vector<double, Observer::ZDim> q_obj{};
    const auto obj = convex::make_quad_form(P_obj, q_obj, Z);
    const auto prob = control::make_observer_problem(plant, Z, obj);

    using Solver = convex::PrimalDualIPM<decltype(prob)>;
    auto cfg = to_convex_config(config);
    const Solver solver(prob, cfg);
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
     * @brief The Akreon Convex decision vector is [Q, W]. Q/W are converted to
     * L through the observer helper so native does not duplicate gain algebra.
     */
    const auto z_dec = leading_decision_vector<Observer::ZDim>(res.iterate.z);
    const auto q_w_split = linalg::splice<Observer::QDim>(z_dec);
    const auto Q_res = linalg::make_smat<N>(std::get<0>(q_w_split));
    const auto W_res = decision_block_to_matrix<N, Y>(std::get<1>(q_w_split));
    const auto [L_res, L_event] = Observer().L(Q_res, W_res);
    result->numerical_event_code = static_cast<int32_t>(L_event);
    if (L_event != linalg::NumericalEvent::NONE)
      return AK_NATIVE_NUMERICAL_ERROR;

    write_matrix(L_res, L_out);
    return AK_NATIVE_OK;
  }
};

/**
 * @brief Dispatch runtime state dimension to a static observer template.
 *
 * @param A borrowed dynamics matrix view
 * @param C borrowed measurement matrix view
 * @param config optional borrowed native config
 * @param L mutable observer gain output view
 * @param result solver diagnostics output
 * @return native status code
 */
inline ak_native_status dispatch_state_observer(
    const ak_native_matrix_view_f64& A, const ak_native_matrix_view_f64& C,
    const ak_native_solver_config_f64* config,
    const ak_native_matrix_mut_f64& L, ak_native_solver_result_f64* result) {
  return dispatch_dimensions(
      std::tuple{typename StateObserverProblem::StateDims{},
                 typename StateObserverProblem::MeasurementDims{}},
      std::array<size_t, 2>{A.rows, C.rows}, [&]<size_t N, size_t Y>() {
        return StateObserverProblem::template solve<N, Y>(A, C, config, L,
                                                          result);
      });
}

} // namespace akreon_native::controls_detail
