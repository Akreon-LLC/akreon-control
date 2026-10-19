/*
 * Copyright © 2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/** @brief Polytopic state-observer solve and dispatch. */

#include "util.hpp"

namespace akreon_native::controls_detail {

/**
 * @brief Polytopic observer policy. It is kept in common because it
 * solves the Akreon Control polytopic observer problem directly rather than
 * H2/Hinf variants.
 */
struct PolytopicStateObserverProblem {
  using VertexDims = PolytopicStateObserverVertexDims;
  using StateDims = PolytopicStateObserverStateDims;
  using MeasurementDims = PolytopicStateObserverMeasurementDims;

  template <size_t V, size_t N, size_t Y>
  static ak_native_status solve(
      const ak_native_matrix_view_f64* A_views,
      const ak_native_matrix_view_f64* C_views,
      const ak_native_polytopic_state_observer_objective_f64* objective,
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
    using PObs = control::PolytopicObserver<Plant, V>;

    /**
     * @brief Each vertex shares the same static plant type. The input matrix is
     * fixed internally because direct observer synthesis consumes A/C from
     * native.
     */
    const auto make_vertex = [&]<size_t Vertex>() {
      return Plant(SS(read_matrix<N, N>(A_views[Vertex]), InputMat{},
                      read_matrix<Y, N>(C_views[Vertex])));
    };
    const auto make_poly =
        [&]<size_t... Vertices>(std::index_sequence<Vertices...>) {
          return typename PObs::Polytope{
              make_vertex.template operator()<Vertices>()...};
        };
    const auto poly = make_poly(std::make_index_sequence<V>{});

    using ZSpace = linalg::RowVector<double, PObs::ZDim>;
    const auto Z = convex::Variable<convex::VarMeta<ZSpace>>();
    const bool has_quadratic =
        objective != nullptr && has_matrix_view(objective->quadratic_P);

    /**
     * @brief The objective branch changes the Akreon Control problem type, so
     * solver execution is isolated in one lambda that handles result/debug
     * extraction uniformly.
     */
    auto run_solver = [&](const auto& prob) {
      using Solver = convex::PrimalDualIPM<std::decay_t<decltype(prob)>>;
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

      const auto z_dec = leading_decision_vector<PObs::ZDim>(res.iterate.z);
      const auto q_w_split = linalg::splice<PObs::QDim>(z_dec);
      const auto Q = linalg::make_smat<N>(std::get<0>(q_w_split));
      const auto W = decision_block_to_matrix<N, Y>(std::get<1>(q_w_split));
      /**
       * @brief Report the caller's quadratic objective when provided; otherwise
       * the default regularizer is internal and objective_value remains NaN.
       */
      if (has_quadratic) {
        const auto P_obj =
            read_matrix<PObs::ZDim, PObs::ZDim>(objective->quadratic_P);
        const auto q_obj = read_vector<PObs::ZDim>(objective->quadratic_q);
        result->objective_value = quadratic_objective(P_obj, q_obj, z_dec);
      }

      const auto [L_res, L_event] = PObs().L(Q, W);
      result->numerical_event_code = static_cast<int32_t>(L_event);
      if (L_event != linalg::NumericalEvent::NONE) {
        return AK_NATIVE_NUMERICAL_ERROR;
      }
      write_matrix(L_res, L_out);
      return AK_NATIVE_OK;
    };

    if (has_quadratic) {
      const auto P =
          read_matrix<PObs::ZDim, PObs::ZDim>(objective->quadratic_P);
      const auto q = read_vector<PObs::ZDim>(objective->quadratic_q);
      const auto obj =
          control::make_polytopic_observer_quadratic_objective<Plant, V>(P, q,
                                                                         Z);
      const auto prob = control::make_polytopic_observer_problem(poly, Z, obj);
      return run_solver(prob);
    }
    /**
     * @brief No objective means use an identity regularizer over [Q, W] to pick
     * a stable feasible observer without exposing a fabricated objective value.
     */
    linalg::Matrix<double, PObs::ZDim, PObs::ZDim> P_obj{};
    for (size_t i = 0; i < PObs::ZDim; ++i) P_obj(i, i) = 1.0;
    linalg::Vector<double, PObs::ZDim> q_obj{};
    const auto obj =
        control::make_polytopic_observer_quadratic_objective<Plant, V>(
            P_obj, q_obj, Z);
    const auto prob = control::make_polytopic_observer_problem(poly, Z, obj);
    return run_solver(prob);
  }
};

/**
 * @brief Dispatch polytopic observer measurement dimension.
 *
 * @tparam V static vertex count
 * @tparam N static state dimension
 *
 * @param y runtime measurement output dimension
 * @param A borrowed array of dynamics matrix views
 * @param C borrowed array of measurement matrix views
 * @param objective optional native objective
 * @param config optional native solver config
 * @param L mutable observer gain output view
 * @param result solver diagnostics output
 * @return native status code
 */
template <size_t V, size_t N>
ak_native_status dispatch_polytopic_observer_y(
    size_t y, const ak_native_matrix_view_f64* A,
    const ak_native_matrix_view_f64* C,
    const ak_native_polytopic_state_observer_objective_f64* objective,
    const ak_native_solver_config_f64* config,
    const ak_native_matrix_mut_f64& L, ak_native_solver_result_f64* result) {
  return dispatch_dimension(
      typename PolytopicStateObserverProblem::MeasurementDims{}, y,
      [&]<size_t Y>() {
        return PolytopicStateObserverProblem::template solve<V, N, Y>(
            A, C, objective, config, L, result);
      });
}

/**
 * @brief Dispatch polytopic observer state dimension.
 *
 * @tparam V static vertex count
 *
 * @param A borrowed array of dynamics matrix views
 * @param C borrowed array of measurement matrix views
 * @param objective optional native objective
 * @param config optional native solver config
 * @param L mutable observer gain output view
 * @param result solver diagnostics output
 * @return native status code
 */
template <size_t V>
ak_native_status dispatch_polytopic_observer_n(
    const ak_native_matrix_view_f64* A, const ak_native_matrix_view_f64* C,
    const ak_native_polytopic_state_observer_objective_f64* objective,
    const ak_native_solver_config_f64* config,
    const ak_native_matrix_mut_f64& L, ak_native_solver_result_f64* result) {
  return dispatch_dimension(typename PolytopicStateObserverProblem::StateDims{},
                            A[0].rows, [&]<size_t N>() {
                              return dispatch_polytopic_observer_y<V, N>(
                                  C[0].rows, A, C, objective, config, L,
                                  result);
                            });
}

/**
 * @brief Dispatch polytopic observer vertex count.
 *
 * @param A borrowed array of dynamics matrix views
 * @param C borrowed array of measurement matrix views
 * @param vertex_count runtime vertex count
 * @param objective optional native objective
 * @param config optional native solver config
 * @param L mutable observer gain output view
 * @param result solver diagnostics output
 * @return native status code
 */
inline ak_native_status dispatch_polytopic_observer(
    const ak_native_matrix_view_f64* A, const ak_native_matrix_view_f64* C,
    size_t vertex_count,
    const ak_native_polytopic_state_observer_objective_f64* objective,
    const ak_native_solver_config_f64* config,
    const ak_native_matrix_mut_f64& L, ak_native_solver_result_f64* result) {
  return dispatch_dimension(
      typename PolytopicStateObserverProblem::VertexDims{}, vertex_count,
      [&]<size_t V>() {
        return dispatch_polytopic_observer_n<V>(A, C, objective, config, L,
                                                result);
      });
}

} // namespace akreon_native::controls_detail
