/*
 * Copyright © 2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/** @brief Polytopic state-feedback solve and dispatch. */

#include "util.hpp"

namespace akreon_native::controls_detail {

/**
 * @brief Polytopic feedback policy. It is kept in common because it is
 * neither H2 optimal nor Hinf robust; it solves the Akreon Control polytopic
 * feedback problem with an optional native objective.
 */
struct PolytopicStateFeedbackProblem {
  using VertexDims = PolytopicStateFeedbackVertexDims;
  using StateDims = PolytopicStateFeedbackStateDims;
  using InputDims = PolytopicStateFeedbackInputDims;

  template <size_t V, size_t N, size_t M>
  static ak_native_status solve(
      const ak_native_matrix_view_f64* A_views,
      const ak_native_matrix_view_f64* B_views,
      const ak_native_polytopic_state_feedback_objective_f64* objective,
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
    using PSF = control::PolytopicStateFeedback<Plant, V>;

    /**
     * @brief Each vertex shares the same static plant type. The vertex pack
     * keeps the polytope construction compile-time while the matrix data
     * remains borrowed.
     */
    const auto make_vertex = [&]<size_t Vertex>() {
      return Plant(SS(read_matrix<N, N>(A_views[Vertex]),
                      read_matrix<N, M>(B_views[Vertex]),
                      linalg::make_identity<double, N>()));
    };
    const auto make_poly =
        [&]<size_t... Vertices>(std::index_sequence<Vertices...>) {
          return typename PSF::Polytope{
              make_vertex.template operator()<Vertices>()...};
        };
    const auto poly = make_poly(std::make_index_sequence<V>{});

    using ZSpace = linalg::RowVector<double, PSF::ZDim>;
    const auto Z = convex::Variable<convex::VarMeta<ZSpace>>();
    const bool has_linear =
        objective != nullptr && has_matrix_view(objective->p_weights);
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

      const auto z_dec = leading_decision_vector<PSF::ZDim>(res.iterate.z);
      const auto p_y_split = linalg::splice<PSF::PDim>(z_dec);
      const auto P = linalg::make_smat<N>(std::get<0>(p_y_split));
      const auto Y = decision_block_to_matrix<M, N>(std::get<1>(p_y_split));
      /**
       * @brief Report the caller's chosen objective when one was provided;
       * otherwise the default regularizer is internal and objective_value
       * remains NaN.
       */
      if (has_linear) {
        const auto p = read_vector<PSF::PDim>(objective->p_weights);
        const auto y = read_vector<PSF::YDim>(objective->y_weights);
        result->objective_value = dot_objective(p, std::get<0>(p_y_split)) +
                                  dot_objective(y, std::get<1>(p_y_split));
      } else if (has_quadratic) {
        const auto P_obj =
            read_matrix<PSF::ZDim, PSF::ZDim>(objective->quadratic_P);
        const auto q_obj = read_vector<PSF::ZDim>(objective->quadratic_q);
        result->objective_value = quadratic_objective(P_obj, q_obj, z_dec);
      }

      const auto [K_res, K_event] = PSF().K(P, Y);
      result->numerical_event_code = static_cast<int32_t>(K_event);
      if (K_event != linalg::NumericalEvent::NONE) {
        return AK_NATIVE_NUMERICAL_ERROR;
      }
      write_matrix(K_res, K_out);
      return AK_NATIVE_OK;
    };

    /**
     * @brief Linear objectives are represented with a tiny quadratic
     * regularizer so Akreon Convex can use a single quadratic objective
     * path.
     */
    if (has_linear) {
      const auto p = read_vector<PSF::PDim>(objective->p_weights);
      const auto y = read_vector<PSF::YDim>(objective->y_weights);
      linalg::Matrix<double, PSF::ZDim, PSF::ZDim> P_reg{};
      linalg::Vector<double, PSF::ZDim> q{};
      for (size_t i = 0; i < PSF::ZDim; ++i) P_reg(i, i) = 1.0;
      for (size_t i = 0; i < PSF::PDim; ++i) q(i) = p(i);
      for (size_t i = 0; i < PSF::YDim; ++i) q(PSF::PDim + i) = y(i);
      const auto obj =
          control::make_polytopic_state_feedback_quadratic_objective<Plant, V>(
              P_reg, q, Z);
      const auto prob =
          control::make_polytopic_state_feedback_problem(poly, Z, obj);
      return run_solver(prob);
    }
    if (has_quadratic) {
      const auto P = read_matrix<PSF::ZDim, PSF::ZDim>(objective->quadratic_P);
      const auto q = read_vector<PSF::ZDim>(objective->quadratic_q);
      const auto obj =
          control::make_polytopic_state_feedback_quadratic_objective<Plant, V>(
              P, q, Z);
      const auto prob =
          control::make_polytopic_state_feedback_problem(poly, Z, obj);
      return run_solver(prob);
    }
    /**
     * @brief No objective means use an identity regularizer over [P, Y] to pick
     * a stable feasible controller without exposing a fabricated objective
     * value.
     */
    linalg::Matrix<double, PSF::ZDim, PSF::ZDim> P_obj{};
    for (size_t i = 0; i < PSF::ZDim; ++i) P_obj(i, i) = 1.0;
    linalg::Vector<double, PSF::ZDim> q_obj{};
    const auto obj =
        control::make_polytopic_state_feedback_quadratic_objective<Plant, V>(
            P_obj, q_obj, Z);
    const auto prob =
        control::make_polytopic_state_feedback_problem(poly, Z, obj);
    return run_solver(prob);
  }
};

/**
 * @brief Dispatch polytopic feedback control dimension.
 *
 * @tparam V static vertex count
 * @tparam N static state dimension
 *
 * @param m runtime control input dimension
 * @param A borrowed array of dynamics matrix views
 * @param B borrowed array of control matrix views
 * @param objective optional native objective
 * @param config optional native solver config
 * @param K mutable state-feedback gain output view
 * @param result solver diagnostics output
 * @return native status code
 */
template <size_t V, size_t N>
ak_native_status dispatch_polytopic_feedback_m(
    size_t m, const ak_native_matrix_view_f64* A,
    const ak_native_matrix_view_f64* B,
    const ak_native_polytopic_state_feedback_objective_f64* objective,
    const ak_native_solver_config_f64* config,
    const ak_native_matrix_mut_f64& K, ak_native_solver_result_f64* result) {
  return dispatch_dimension(
      typename PolytopicStateFeedbackProblem::InputDims{}, m, [&]<size_t M>() {
        return PolytopicStateFeedbackProblem::template solve<V, N, M>(
            A, B, objective, config, K, result);
      });
}

/**
 * @brief Dispatch polytopic feedback state dimension.
 *
 * @tparam V static vertex count
 *
 * @param A borrowed array of dynamics matrix views
 * @param B borrowed array of control matrix views
 * @param objective optional native objective
 * @param config optional native solver config
 * @param K mutable state-feedback gain output view
 * @param result solver diagnostics output
 * @return native status code
 */
template <size_t V>
ak_native_status dispatch_polytopic_feedback_n(
    const ak_native_matrix_view_f64* A, const ak_native_matrix_view_f64* B,
    const ak_native_polytopic_state_feedback_objective_f64* objective,
    const ak_native_solver_config_f64* config,
    const ak_native_matrix_mut_f64& K, ak_native_solver_result_f64* result) {
  return dispatch_dimension(typename PolytopicStateFeedbackProblem::StateDims{},
                            A[0].rows, [&]<size_t N>() {
                              return dispatch_polytopic_feedback_m<V, N>(
                                  B[0].cols, A, B, objective, config, K,
                                  result);
                            });
}

/**
 * @brief Dispatch polytopic feedback vertex count.
 *
 * @param A borrowed array of dynamics matrix views
 * @param B borrowed array of control matrix views
 * @param vertex_count runtime vertex count
 * @param objective optional native objective
 * @param config optional native solver config
 * @param K mutable state-feedback gain output view
 * @param result solver diagnostics output
 * @return native status code
 */
inline ak_native_status dispatch_polytopic_feedback(
    const ak_native_matrix_view_f64* A, const ak_native_matrix_view_f64* B,
    size_t vertex_count,
    const ak_native_polytopic_state_feedback_objective_f64* objective,
    const ak_native_solver_config_f64* config,
    const ak_native_matrix_mut_f64& K, ak_native_solver_result_f64* result) {
  return dispatch_dimension(
      typename PolytopicStateFeedbackProblem::VertexDims{}, vertex_count,
      [&]<size_t V>() {
        return dispatch_polytopic_feedback_n<V>(A, B, objective, config, K,
                                                result);
      });
}

} // namespace akreon_native::controls_detail
