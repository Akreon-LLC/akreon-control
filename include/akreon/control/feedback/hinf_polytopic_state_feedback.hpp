/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "akreon/control/feedback/hinf_state_feedback.hpp"
#include "akreon/convex/convex_problem.hpp"
#include "akreon/convex/intersection.hpp"
#include "akreon/convex/variable.hpp"

namespace control {

/**
 * @brief Hinf Polytopic State Feedback
 *
 * Uses a common Hinf state-feedback decision variable
 *   z = [svec(Q); vec(Y); gamma]
 *
 * across every plant vertex and intersects each vertex Hinf certificate:
 *   Q     : inverse BRL Lyapunov variable
 *   Y     : lifted feedback product, Y = K Q
 *   gamma : shared induced L2-gain bound
 */
template <ContinuousPlantModelLike Plant, std::size_t N>
struct HinfPolytopicStateFeedback {
  /// @brief extracting alias and types
  using Model = linalg::remove_cvref_t<Plant>;
  using T = plant_type_t<Model>;
  using VertexFeedback = HinfStateFeedback<Model>;

  static constexpr std::size_t StateDim = VertexFeedback::StateDim;
  static constexpr std::size_t InputDim = VertexFeedback::InputDim;
  static constexpr std::size_t ExogenousDim = VertexFeedback::ExogenousDim;
  static constexpr std::size_t OutputDim = VertexFeedback::OutputDim;
  static constexpr std::size_t QDim = VertexFeedback::QDim;
  static constexpr std::size_t YDim = VertexFeedback::YDim;
  static constexpr std::size_t GammaDim = VertexFeedback::GammaDim;
  static constexpr std::size_t DecVarDim = VertexFeedback::DecVarDim;

  using QMat = typename VertexFeedback::QMat;
  using YMat = typename VertexFeedback::YMat;
  using KMat = typename VertexFeedback::KMat;
  using DecVarSpace = typename VertexFeedback::DecVarSpace;
  using Polytope = linalg::Vector<Plant, N>;

  /**
   * @brief operator overload to form Hinf polytopic state feedback
   *
   * @param poly (Polytope of plant models)
   * @param Z_hat (dec var)
   * @param margin (PDness)
   *
   * @return Hinf LMIs
   */
  template <convex::AtomLike DecVar>
  constexpr auto operator()(const Polytope& poly, const DecVar& Z_hat,
                            T margin = 1e-6) const {
    return apply(poly, Z_hat, margin);
  }

  /**
   * @brief apply
   *
   * @param poly (Polytope of plant models)
   * @param Z_hat (dec var)
   * @param margin (PDness)
   *
   * @return Hinf LMIs
   */
  template <convex::AtomLike DecVar>
  constexpr auto apply(const Polytope& poly, const DecVar& Z_hat,
                       T margin = 1e-6) const {
    static_assert(convex::is_var_type_v<DecVar>, "Input atom is not a var");
    using DecVarDom = convex::meta_dom_t<convex::atom_meta_t<DecVar>>;
    static_assert(convex::dom_vartype_v<DecVarDom> == convex::VarType::FREE,
                  "Merged decision var must be free");
    static_assert(std::same_as<DecVarSpace, convex::dom_space_t<DecVarDom>>,
                  "Expected merged decision size [svec(Q); vec(Y); gamma]");

    auto make_vertex_lmis = [&]<std::size_t idx>() {
      return VertexFeedback()(poly(idx), Z_hat, margin);
    };

    auto form_intersection =
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
          return convex::Intersection(
              make_vertex_lmis.template operator()<Is>()...);
        };

    return form_intersection(std::make_index_sequence<N>{});
  }

  /**
   * @brief objective
   *
   * @param Z_hat (dec var)
   *
   * @return gamma
   */
  template <convex::AtomLike DecVar>
  constexpr auto objective(const DecVar& Z_hat) const {
    return VertexFeedback().objective(Z_hat);
  }

  /**
   * @brief K (feedback gain)
   *
   * @param Q (inverse BRL Lyapunov variable)
   * @param Y (Lifted convex object)
   *
   * @return K
   */
  constexpr auto K(const QMat& Q, const YMat& Y) const {
    return VertexFeedback().K(Q, Y);
  }
};

/**
 * @brief make polytopic Hinf state feedback
 *
 * @param poly (Polytope of plant models)
 * @param Z_hat (dec var)
 * @param margin (PDness)
 *
 * @return Hinf problem
 */
template <ContinuousPlantModelLike Plant, std::size_t N,
          convex::AtomLike DecVar>
constexpr auto make_hinf_polytopic_state_feedback_problem(
    const linalg::Vector<Plant, N>& poly, const DecVar& Z_hat,
    plant_type_t<Plant> margin = 1e-6) {
  const auto hinf = HinfPolytopicStateFeedback<Plant, N>();
  const auto lmis = hinf(poly, Z_hat, margin);
  const auto prob = convex::make_problem(Z_hat).in_set(lmis);
  const auto obj = hinf.objective(Z_hat);
  return prob.minimize(obj);
}

} // namespace control
