/*
 * Copyright © 2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/** @brief H2 optimal-control policies with embedded static solve
 * implementations. */

#include "../common/util.hpp"

namespace akreon_native::controls_detail {

/**
 * @brief Policy consumed by dispatch_generalized_feedback. It owns the
 * H2-specific Akreon Control plant aliases, optimization problem construction,
 * and gain recovery.
 */
struct H2StateFeedbackProblem {
  using StateDims = H2StateFeedbackStateDims;
  using ExogenousDims = H2StateFeedbackExogenousDims;
  using ControlDims = H2StateFeedbackControlDims;
  using PerformanceDims = H2StateFeedbackPerformanceDims;

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
    using H2 = control::H2StateFeedback<Plant>;

    /**
     * @brief The ABI supplies raw row-major views; the policy is the first
     * place where those views become statically dimensioned Akreon Linalg
     * matrices and port layouts.
     */
    const auto plant =
        Plant(SS(read_matrix<N, N>(A_view), read_matrix<N, W + M>(B_view),
                 read_matrix<Z, N>(C_view), read_matrix<Z, W + M>(D_view)));
    using DecVarSpace = linalg::RowVector<double, H2::DecVarDim>;
    const auto dec = convex::Variable<convex::VarMeta<DecVarSpace>>();
    const auto prob = control::make_h2_state_feedback_problem(plant, dec);
    using Solver = convex::PrimalDualIPM<decltype(prob)>;
    auto cfg = to_convex_config(config);
    if (config == nullptr) { cfg.duality_gap_tol = 1e-3; }
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
     * @brief H2 feedback stores Q/Y/Z in the solver decision vector. Q and Y
     * recover K; trace(Z) is the reported H2 objective.
     */
    const auto q_yz_split = linalg::splice<H2::QDim>(res.iterate.z);
    const auto Q = linalg::make_smat<N>(std::get<0>(q_yz_split));
    const auto y_z_split = linalg::splice<H2::YDim>(std::get<1>(q_yz_split));
    const auto Y = decision_block_to_matrix<M, N>(std::get<0>(y_z_split));
    const auto Zmat = linalg::make_smat<Z>(std::get<1>(y_z_split));
    double trace{};
    for (size_t i = 0; i < Z; ++i) trace += Zmat(i, i);
    result->objective_value = trace;

    const auto [K_res, K_event] = H2().K(Q, Y);
    result->numerical_event_code = static_cast<int32_t>(K_event);
    if (K_event != linalg::NumericalEvent::NONE)
      return AK_NATIVE_NUMERICAL_ERROR;
    write_matrix(K_res, K_out);
    return AK_NATIVE_OK;
  }
};

/**
 * @brief Observer variant of the H2 generalized plant. Measurement and
 * performance ports are both present, so the policy carries Y and Z
 * compile-time aliases.
 */
struct H2StateObserverProblem {
  using StateDims = H2StateObserverStateDims;
  using ExogenousDims = H2StateObserverExogenousDims;
  using PerformanceDims = H2StateObserverPerformanceDims;
  using MeasurementDims = H2StateObserverMeasurementDims;

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
    using H2 = control::H2StateObserver<Plant>;

    /**
     * @brief Inputs are [w], outputs are [z, y], matching the observer
     * generalized-plant contract used by Akreon Control.
     */
    const auto plant =
        Plant(SS(read_matrix<N, N>(A_view), read_matrix<N, W>(B_view),
                 read_matrix<Z + Y, N>(C_view), read_matrix<Z + Y, W>(D_view)));
    using DecVarSpace = linalg::RowVector<double, H2::DecVarDim>;
    const auto dec = convex::Variable<convex::VarMeta<DecVarSpace>>();
    const auto prob = control::make_h2_state_observer_problem(plant, dec);
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
     * @brief H2 observer stores Q/W/Z in the decision vector. Q and W recover
     * L; trace(Z) is the reported objective.
     */
    const auto q_wz_split = linalg::splice<H2::QDim>(res.iterate.z);
    const auto Q = linalg::make_smat<N>(std::get<0>(q_wz_split));
    const auto w_z_split = linalg::splice<H2::WDim>(std::get<1>(q_wz_split));
    const auto Wmat = decision_block_to_matrix<N, Y>(std::get<0>(w_z_split));
    const auto Zmat = linalg::make_smat<W>(std::get<1>(w_z_split));
    double trace{};
    for (size_t i = 0; i < W; ++i) trace += Zmat(i, i);
    result->objective_value = trace;

    const auto [L_res, L_event] = H2().L(Q, Wmat);
    result->numerical_event_code = static_cast<int32_t>(L_event);
    if (L_event != linalg::NumericalEvent::NONE)
      return AK_NATIVE_NUMERICAL_ERROR;
    write_matrix(L_res, L_out);
    return AK_NATIVE_OK;
  }
};

/**
 * @brief Polytopic H2 feedback policy. Dispatch fixes V/N/W/M/Z first; this
 * policy then materializes the Akreon Control polytope as a compile-time pack
 * of vertices.
 */
struct H2PolytopicStateFeedbackProblem {
  using VertexDims = H2PolytopicStateFeedbackVertexDims;
  using StateDims = H2PolytopicStateFeedbackStateDims;
  using ExogenousDims = H2PolytopicStateFeedbackExogenousDims;
  using ControlDims = H2PolytopicStateFeedbackControlDims;
  using PerformanceDims = H2PolytopicStateFeedbackPerformanceDims;

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
    using H2 = control::H2PolytopicStateFeedback<Plant, V>;

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
    const auto make_poly = [&]<size_t... Vertices>(
                               std::index_sequence<Vertices...>) {
      return
          typename H2::Polytope{make_vertex.template operator()<Vertices>()...};
    };
    const auto poly = make_poly(std::make_index_sequence<V>{});

    using DecVarSpace = linalg::RowVector<double, H2::DecVarDim>;
    const auto dec = convex::Variable<convex::VarMeta<DecVarSpace>>();
    const auto prob =
        control::make_h2_polytopic_state_feedback_problem(poly, dec);
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
     * @brief The polytopic H2 feedback decision layout matches the nominal H2
     * layout: Q/Y recover K and Z contributes trace(Z).
     */
    const auto q_yz_split = linalg::splice<H2::QDim>(res.iterate.z);
    const auto Q = linalg::make_smat<N>(std::get<0>(q_yz_split));
    const auto y_z_split = linalg::splice<H2::YDim>(std::get<1>(q_yz_split));
    const auto Y = decision_block_to_matrix<M, N>(std::get<0>(y_z_split));
    const auto Zmat = linalg::make_smat<Z>(std::get<1>(y_z_split));
    double trace{};
    for (size_t i = 0; i < Z; ++i) trace += Zmat(i, i);
    result->objective_value = trace;

    const auto [K_res, K_event] = H2().K(Q, Y);
    result->numerical_event_code = static_cast<int32_t>(K_event);
    if (K_event != linalg::NumericalEvent::NONE)
      return AK_NATIVE_NUMERICAL_ERROR;
    write_matrix(K_res, K_out);
    return AK_NATIVE_OK;
  }
};

/**
 * @brief Polytopic H2 observer policy. The vertex pack shares one observer gain
 * across all plants in the polytope.
 */
struct H2PolytopicStateObserverProblem {
  using VertexDims = H2PolytopicStateObserverVertexDims;
  using StateDims = H2PolytopicStateObserverStateDims;
  using ExogenousDims = H2PolytopicStateObserverExogenousDims;
  using PerformanceDims = H2PolytopicStateObserverPerformanceDims;
  using MeasurementDims = H2PolytopicStateObserverMeasurementDims;

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
    using H2 = control::H2PolytopicStateObserver<Plant, V>;

    /**
     * @brief The compiled dimension configuration decides V; this block only
     * maps each ABI vertex into the fixed Akreon Control plant type required by
     * H2PolytopicStateObserver.
     */
    const auto make_vertex = [&]<size_t Vertex>() {
      return Plant(SS(read_matrix<N, N>(A_views[Vertex]),
                      read_matrix<N, W>(B_views[Vertex]),
                      read_matrix<Z + Y, N>(C_views[Vertex]),
                      read_matrix<Z + Y, W>(D_views[Vertex])));
    };
    const auto make_poly = [&]<size_t... Vertices>(
                               std::index_sequence<Vertices...>) {
      return
          typename H2::Polytope{make_vertex.template operator()<Vertices>()...};
    };
    const auto poly = make_poly(std::make_index_sequence<V>{});

    using DecVarSpace = linalg::RowVector<double, H2::DecVarDim>;
    const auto dec = convex::Variable<convex::VarMeta<DecVarSpace>>();
    const auto prob =
        control::make_h2_polytopic_state_observer_problem(poly, dec);
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

    const auto q_wz_split = linalg::splice<H2::QDim>(res.iterate.z);
    const auto Q = linalg::make_smat<N>(std::get<0>(q_wz_split));
    const auto w_z_split = linalg::splice<H2::WDim>(std::get<1>(q_wz_split));
    const auto Wmat = decision_block_to_matrix<N, Y>(std::get<0>(w_z_split));
    const auto Zmat = linalg::make_smat<W>(std::get<1>(w_z_split));
    double trace{};
    for (size_t i = 0; i < W; ++i) trace += Zmat(i, i);
    result->objective_value = trace;

    const auto [L_res, L_event] = H2().L(Q, Wmat);
    result->numerical_event_code = static_cast<int32_t>(L_event);
    if (L_event != linalg::NumericalEvent::NONE)
      return AK_NATIVE_NUMERICAL_ERROR;
    write_matrix(L_res, L_out);
    return AK_NATIVE_OK;
  }
};

} // namespace akreon_native::controls_detail
