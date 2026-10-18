/*
 * Copyright © 2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/** @brief Hinf robust-control policies with embedded static solve
 * implementations. */

#include "../common/util.hpp"

namespace akreon_native::controls_detail {

/**
 * @brief Policy consumed by dispatch_generalized_feedback. It owns the
 * Hinf-specific Akreon Control plant aliases, optimization problem
 * construction, gamma extraction, and gain recovery.
 */
struct HinfStateFeedbackProblem {
  using Shapes = HinfFeedbackShapes;
  using StateDims = HinfStateFeedbackStateDims;
  using ExogenousDims = HinfStateFeedbackExogenousDims;
  using ControlDims = HinfStateFeedbackControlDims;
  using PerformanceDims = HinfStateFeedbackPerformanceDims;

  template <size_t N, size_t W, size_t M, size_t Z>
  static ak_native_status solve(const ak_native_matrix_view_f64& A_view,
                                const ak_native_matrix_view_f64& B_view,
                                const ak_native_matrix_view_f64& C_view,
                                const ak_native_matrix_view_f64& D_view,
                                const ak_native_solver_config_f64* config,
                                const ak_native_matrix_mut_f64& K_out,
                                ak_native_solver_result_f64* result) {
    using StateMat = linalg::Matrix<double, N, N>;
    using InputMat = linalg::Matrix<double, N, W + M>;
    using OutputMat = linalg::Matrix<double, Z, N>;
    using SS = control::StateSpace<control::TimeDomainType::CONTINUOUS,
                                   StateMat, InputMat, OutputMat>;
    using Inputs =
        control::PortLayout<control::Port<control::PortRole::EXOGENOUS, 0, W>,
                            control::Port<control::PortRole::CONTROL, W, M>>;
    using Outputs = control::PortLayout<
        control::Port<control::PortRole::PERFORMANCE, 0, Z>>;
    using Plant = control::PlantModel<SS, Inputs, Outputs>;
    using Hinf = control::HinfStateFeedback<Plant>;

    /**
     * @brief The ABI supplies raw row-major views; the policy is the first
     * place where those views become statically dimensioned Akreon Linalg
     * matrices and port layouts.
     */
    const auto plant =
        Plant(SS(read_matrix<N, N>(A_view), read_matrix<N, W + M>(B_view),
                 read_matrix<Z, N>(C_view), read_matrix<Z, W + M>(D_view)));
    using DecVarSpace = linalg::RowVector<double, Hinf::DecVarDim>;
    const auto dec = convex::Variable<convex::VarMeta<DecVarSpace>>();
    const auto prob = control::make_hinf_state_feedback_problem(plant, dec);
    using Solver = convex::PrimalDualIPM<decltype(prob)>;
    auto cfg = to_convex_config(config);
    if (config == nullptr) cfg.duality_gap_tol = 1e-3;
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
     * @brief Hinf feedback stores Q/Y/gamma in the solver decision vector. Q
     * and Y recover K; gamma is surfaced through the native result.
     */
    const auto q_y_gamma_split = linalg::splice<Hinf::QDim>(res.iterate.z);
    const auto Q = linalg::make_smat<N>(std::get<0>(q_y_gamma_split));
    const auto y_gamma_split =
        linalg::splice<Hinf::YDim>(std::get<1>(q_y_gamma_split));
    const auto Y = decision_block_to_matrix<M, N>(std::get<0>(y_gamma_split));
    result->gamma = std::get<1>(y_gamma_split)(0);

    const auto [K_res, K_event] = Hinf().K(Q, Y);
    result->numerical_event_code = static_cast<int32_t>(K_event);
    if (K_event != linalg::NumericalEvent::NONE)
      return AK_NATIVE_NUMERICAL_ERROR;
    write_matrix(K_res, K_out);
    return AK_NATIVE_OK;
  }
};

/**
 * @brief Observer variant of the Hinf generalized plant. Measurement and
 * performance ports are both present, so the policy carries Y and Z
 * compile-time aliases.
 */
struct HinfStateObserverProblem {
  using StateDims = HinfStateObserverStateDims;
  using ExogenousDims = HinfStateObserverExogenousDims;
  using PerformanceDims = HinfStateObserverPerformanceDims;
  using MeasurementDims = HinfStateObserverMeasurementDims;

  template <size_t N, size_t W, size_t Z, size_t Y>
  static ak_native_status solve(const ak_native_matrix_view_f64& A_view,
                                const ak_native_matrix_view_f64& B_view,
                                const ak_native_matrix_view_f64& C_view,
                                const ak_native_matrix_view_f64& D_view,
                                const ak_native_solver_config_f64* config,
                                const ak_native_matrix_mut_f64& L_out,
                                ak_native_solver_result_f64* result) {
    using StateMat = linalg::Matrix<double, N, N>;
    using InputMat = linalg::Matrix<double, N, W>;
    using OutputMat = linalg::Matrix<double, Z + Y, N>;
    using SS = control::StateSpace<control::TimeDomainType::CONTINUOUS,
                                   StateMat, InputMat, OutputMat>;
    using Inputs =
        control::PortLayout<control::Port<control::PortRole::EXOGENOUS, 0, W>>;
    using Outputs = control::PortLayout<
        control::Port<control::PortRole::PERFORMANCE, 0, Z>,
        control::Port<control::PortRole::MEASUREMENT, Z, Y>>;
    using Plant = control::PlantModel<SS, Inputs, Outputs>;
    using Hinf = control::HinfStateObserver<Plant>;

    /**
     * @brief Inputs are [w], outputs are [z, y], matching the observer
     * generalized-plant contract used by Akreon Control.
     */
    const auto plant =
        Plant(SS(read_matrix<N, N>(A_view), read_matrix<N, W>(B_view),
                 read_matrix<Z + Y, N>(C_view), read_matrix<Z + Y, W>(D_view)));
    using DecVarSpace = linalg::RowVector<double, Hinf::DecVarDim>;
    const auto dec = convex::Variable<convex::VarMeta<DecVarSpace>>();
    const auto prob = control::make_hinf_state_observer_problem(plant, dec);
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
     * @brief Hinf observer stores Q/W/gamma in the decision vector. Q and W
     * recover L; gamma is surfaced through the native result.
     */
    const auto q_w_gamma_split = linalg::splice<Hinf::QDim>(res.iterate.z);
    const auto Q = linalg::make_smat<N>(std::get<0>(q_w_gamma_split));
    const auto w_gamma_split =
        linalg::splice<Hinf::WDim>(std::get<1>(q_w_gamma_split));
    const auto Wmat =
        decision_block_to_matrix<N, Y>(std::get<0>(w_gamma_split));
    result->gamma = std::get<1>(w_gamma_split)(0);

    const auto [L_res, L_event] = Hinf().L(Q, Wmat);
    result->numerical_event_code = static_cast<int32_t>(L_event);
    if (L_event != linalg::NumericalEvent::NONE)
      return AK_NATIVE_NUMERICAL_ERROR;
    write_matrix(L_res, L_out);
    return AK_NATIVE_OK;
  }
};

/**
 * @brief Polytopic Hinf feedback policy. Dispatch fixes V/N/W/M/Z first; this
 * policy then materializes the Akreon Control polytope as a compile-time pack
 * of vertices.
 */
struct HinfPolytopicStateFeedbackProblem {
  using VertexDims = HinfPolytopicStateFeedbackVertexDims;
  using StateDims = HinfPolytopicStateFeedbackStateDims;
  using ExogenousDims = HinfPolytopicStateFeedbackExogenousDims;
  using ControlDims = HinfPolytopicStateFeedbackControlDims;
  using PerformanceDims = HinfPolytopicStateFeedbackPerformanceDims;

  template <size_t V, size_t N, size_t W, size_t M, size_t Z>
  static ak_native_status solve(const ak_native_matrix_view_f64* A_views,
                                const ak_native_matrix_view_f64* B_views,
                                const ak_native_matrix_view_f64* C_views,
                                const ak_native_matrix_view_f64* D_views,
                                const ak_native_solver_config_f64* config,
                                const ak_native_matrix_mut_f64& K_out,
                                ak_native_solver_result_f64* result) {
    using StateMat = linalg::Matrix<double, N, N>;
    using InputMat = linalg::Matrix<double, N, W + M>;
    using OutputMat = linalg::Matrix<double, Z, N>;
    using SS = control::StateSpace<control::TimeDomainType::CONTINUOUS,
                                   StateMat, InputMat, OutputMat>;
    using Inputs =
        control::PortLayout<control::Port<control::PortRole::EXOGENOUS, 0, W>,
                            control::Port<control::PortRole::CONTROL, W, M>>;
    using Outputs = control::PortLayout<
        control::Port<control::PortRole::PERFORMANCE, 0, Z>>;
    using Plant = control::PlantModel<SS, Inputs, Outputs>;
    using Hinf = control::HinfPolytopicStateFeedback<Plant, V>;

    /**
     * @brief Build the Akreon Control polytope without dynamic allocation; each
     * vertex is a Plant value constructed from the matching ABI matrix views.
     */
    const auto make_vertex = [&]<size_t Vertex>() {
      return Plant(SS(read_matrix<N, N>(A_views[Vertex]),
                      read_matrix<N, W + M>(B_views[Vertex]),
                      read_matrix<Z, N>(C_views[Vertex]),
                      read_matrix<Z, W + M>(D_views[Vertex])));
    };
    const auto make_poly =
        [&]<size_t... Vertices>(std::index_sequence<Vertices...>) {
          return typename Hinf::Polytope{
              make_vertex.template operator()<Vertices>()...};
        };
    const auto poly = make_poly(std::make_index_sequence<V>{});

    using DecVarSpace = linalg::RowVector<double, Hinf::DecVarDim>;
    const auto dec = convex::Variable<convex::VarMeta<DecVarSpace>>();
    const auto prob =
        control::make_hinf_polytopic_state_feedback_problem(poly, dec);
    using Solver = convex::PrimalDualIPM<decltype(prob)>;
    auto cfg = to_convex_config(config);
    if (config == nullptr) cfg.duality_gap_tol = 1e-3;
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
     * @brief The polytopic Hinf feedback decision layout matches the nominal
     * Hinf layout: Q/Y recover K and gamma is copied to the result.
     */
    const auto q_y_gamma_split = linalg::splice<Hinf::QDim>(res.iterate.z);
    const auto Q = linalg::make_smat<N>(std::get<0>(q_y_gamma_split));
    const auto y_gamma_split =
        linalg::splice<Hinf::YDim>(std::get<1>(q_y_gamma_split));
    const auto Y = decision_block_to_matrix<M, N>(std::get<0>(y_gamma_split));
    result->gamma = std::get<1>(y_gamma_split)(0);

    const auto [K_res, K_event] = Hinf().K(Q, Y);
    result->numerical_event_code = static_cast<int32_t>(K_event);
    if (K_event != linalg::NumericalEvent::NONE)
      return AK_NATIVE_NUMERICAL_ERROR;
    write_matrix(K_res, K_out);
    return AK_NATIVE_OK;
  }
};

/**
 * @brief Polytopic Hinf observer policy. The vertex pack shares one observer
 * gain across all plants in the polytope.
 */
struct HinfPolytopicStateObserverProblem {
  using VertexDims = HinfPolytopicStateObserverVertexDims;
  using StateDims = HinfPolytopicStateObserverStateDims;
  using ExogenousDims = HinfPolytopicStateObserverExogenousDims;
  using PerformanceDims = HinfPolytopicStateObserverPerformanceDims;
  using MeasurementDims = HinfPolytopicStateObserverMeasurementDims;

  template <size_t V, size_t N, size_t W, size_t Z, size_t Y>
  static ak_native_status solve(const ak_native_matrix_view_f64* A_views,
                                const ak_native_matrix_view_f64* B_views,
                                const ak_native_matrix_view_f64* C_views,
                                const ak_native_matrix_view_f64* D_views,
                                const ak_native_solver_config_f64* config,
                                const ak_native_matrix_mut_f64& L_out,
                                ak_native_solver_result_f64* result) {
    using StateMat = linalg::Matrix<double, N, N>;
    using InputMat = linalg::Matrix<double, N, W>;
    using OutputMat = linalg::Matrix<double, Z + Y, N>;
    using SS = control::StateSpace<control::TimeDomainType::CONTINUOUS,
                                   StateMat, InputMat, OutputMat>;
    using Inputs =
        control::PortLayout<control::Port<control::PortRole::EXOGENOUS, 0, W>>;
    using Outputs = control::PortLayout<
        control::Port<control::PortRole::PERFORMANCE, 0, Z>,
        control::Port<control::PortRole::MEASUREMENT, Z, Y>>;
    using Plant = control::PlantModel<SS, Inputs, Outputs>;
    using Hinf = control::HinfPolytopicStateObserver<Plant, V>;

    /**
     * @brief The compiled dimension configuration decides V; this block only
     * maps each ABI vertex into the fixed Akreon Control plant type required by
     * HinfPolytopicStateObserver.
     */
    const auto make_vertex = [&]<size_t Vertex>() {
      return Plant(SS(read_matrix<N, N>(A_views[Vertex]),
                      read_matrix<N, W>(B_views[Vertex]),
                      read_matrix<Z + Y, N>(C_views[Vertex]),
                      read_matrix<Z + Y, W>(D_views[Vertex])));
    };
    const auto make_poly =
        [&]<size_t... Vertices>(std::index_sequence<Vertices...>) {
          return typename Hinf::Polytope{
              make_vertex.template operator()<Vertices>()...};
        };
    const auto poly = make_poly(std::make_index_sequence<V>{});

    using DecVarSpace = linalg::RowVector<double, Hinf::DecVarDim>;
    const auto dec = convex::Variable<convex::VarMeta<DecVarSpace>>();
    const auto prob =
        control::make_hinf_polytopic_state_observer_problem(poly, dec);
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

    const auto q_w_gamma_split = linalg::splice<Hinf::QDim>(res.iterate.z);
    const auto Q = linalg::make_smat<N>(std::get<0>(q_w_gamma_split));
    const auto w_gamma_split =
        linalg::splice<Hinf::WDim>(std::get<1>(q_w_gamma_split));
    const auto Wmat =
        decision_block_to_matrix<N, Y>(std::get<0>(w_gamma_split));
    result->gamma = std::get<1>(w_gamma_split)(0);

    const auto [L_res, L_event] = Hinf().L(Q, Wmat);
    result->numerical_event_code = static_cast<int32_t>(L_event);
    if (L_event != linalg::NumericalEvent::NONE)
      return AK_NATIVE_NUMERICAL_ERROR;
    write_matrix(L_res, L_out);
    return AK_NATIVE_OK;
  }
};

} // namespace akreon_native::controls_detail
