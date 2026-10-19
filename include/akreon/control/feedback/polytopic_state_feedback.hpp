/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "akreon/control/feedback/state_feedback.hpp"
#include "akreon/convex/composed/quad_form.hpp"
#include "akreon/convex/composed/weighted_sum.hpp"
#include "akreon/convex/convex_problem.hpp"
#include "akreon/convex/intersection.hpp"
#include "akreon/convex/variable.hpp"

namespace control {

/**
 * @brief PolytopicStateFeedback
 *
 * Polytopic State Feedback (PLDI)
 * Given the LTV system
 *   xdot = A(t) x + B_u(t) u
 *
 * We seek a static state feedback controller
 *   u = K x
 *
 * s.t. the CL system is quadratically stable.
 *
 * There exists S > 0, P = S^{-1}, and Y = K P so
 *   P A_i^T + A_i P + B_{ui} Y + Y^T B_{ui}^T < 0
 *
 * This set is represented over the merged decision variable
 *   z = [ svec(P); vec(Y) ].
 */
template <ContinuousPlantModelLike Plant, std::size_t N>
struct PolytopicStateFeedback {
  using Model = linalg::remove_cvref_t<Plant>;

  using T = plant_type_t<Model>;
  using VertexFeedback = StateFeedback<Model>;

  static constexpr std::size_t StateDim = VertexFeedback::StateDim;
  static constexpr std::size_t InputDim = VertexFeedback::InputDim;
  static constexpr std::size_t PDim = VertexFeedback::PDim;
  static constexpr std::size_t YDim = VertexFeedback::YDim;
  static constexpr std::size_t ZDim = VertexFeedback::ZDim;

  using PMat = typename VertexFeedback::PMat;
  using YMat = typename VertexFeedback::YMat;
  using KMat = typename VertexFeedback::KMat;
  using FMat = typename VertexFeedback::FMat;
  using ZSpace = typename VertexFeedback::ZSpace;
  using Fs = typename VertexFeedback::Fs;
  using Polytope = linalg::Vector<Plant, N>;

  /**
   * @brief operator overload (form polytopic state feedback)
   *
   * @param poly (Polytope)
   * @param Z (dec var [svec(P); vec(Y)])
   *
   * @return PLDI as intersection of LMI
   */
  template <convex::AtomLike ZVar>
  constexpr auto operator()(const Polytope& poly, const ZVar& Z,
                            T margin = 1e-6) const {
    return apply(poly, Z, margin);
  }

  /**
   * @brief apply (polytopic state feedback)
   *
   * @param poly (Polytope)
   * @param Z (dec var [svec(P); vec(Y)])
   *
   * @return PLDI as intersection of LMI
   */
  template <convex::AtomLike ZVar>
  constexpr auto apply(const Polytope& poly, const ZVar& Z,
                       T margin = 1e-6) const {
    static_assert(convex::is_var_type_v<ZVar>, "Input atom is not a var");
    using ZDom = convex::meta_dom_t<convex::atom_meta_t<ZVar>>;
    static_assert(convex::dom_vartype_v<ZDom> == convex::VarType::FREE,
                  "Merged decision var must be free");
    using ZVarSpace = convex::dom_space_t<ZDom>;
    static_assert(std::same_as<ZVarSpace, ZSpace>,
                  "Expected merged decision size [svec(P); vec(Y)]");

    const auto p_lmi = VertexFeedback().p_constraint(Z, margin);

    auto make_vertex_lmi = [&]<std::size_t idx>() {
      return VertexFeedback().stability_constraint(poly(idx), Z);
    };

    auto form_intersection = [&]<std::size_t... Is>(
                                 std::index_sequence<Is...>) {
      return convex::Intersection(p_lmi,
                                  make_vertex_lmi.template operator()<Is>()...);
    };

    return form_intersection(std::make_index_sequence<N>{});
  }

  /**
   * @brief K
   *
   * K = Y P^{-1}
   *
   * @param P (symmetric Lyapunov variable)
   * @param Y (lifted feedback product Y = KP)
   *
   * @return K
   */
  constexpr auto K(const PMat& P, const YMat& Y) const {
    return VertexFeedback().K(P, Y);
  }
};

/**
 * @brief make linear objective over merged PLDI variable
 *
 * obj(z) = p_w^T svec(P) + y_w^T vec(Y)
 *
 * @param p_weights weights over svec(P)
 * @param y_weights weights over vec(Y)
 * @param Z merged decision var
 *
 * @return linear objective atom
 */
template <ContinuousPlantModelLike Plant, std::size_t N, convex::AtomLike ZVar>
constexpr auto make_polytopic_state_feedback_linear_objective(
    const linalg::Vector<plant_type_t<Plant>,
                         PolytopicStateFeedback<Plant, N>::PDim>& p_weights,
    const linalg::Vector<plant_type_t<Plant>,
                         PolytopicStateFeedback<Plant, N>::YDim>& y_weights,
    const ZVar& Z) {
  using T = plant_type_t<Plant>;
  using PSF = PolytopicStateFeedback<Plant, N>;

  linalg::Vector<T, PSF::ZDim> c{};
  for (std::size_t i = 0; i < PSF::PDim; ++i) c(i) = p_weights(i);
  for (std::size_t i = 0; i < PSF::YDim; ++i) c(PSF::PDim + i) = y_weights(i);

  return convex::make_weighted_sum(c, Z);
}

/**
 * @brief make quadratic objective over merged PLDI variable
 *
 * obj(z) = 0.5 z^T P z + q^T z
 *
 * @param P quadratic weight over merged z
 * @param q linear weight over merged z
 * @param Z merged decision var
 *
 * @return quadratic objective atom
 */
template <ContinuousPlantModelLike Plant, std::size_t N, convex::AtomLike ZVar>
constexpr auto make_polytopic_state_feedback_quadratic_objective(
    const linalg::Matrix<plant_type_t<Plant>,
                         PolytopicStateFeedback<Plant, N>::ZDim,
                         PolytopicStateFeedback<Plant, N>::ZDim>& P,
    const linalg::Vector<plant_type_t<Plant>,
                         PolytopicStateFeedback<Plant, N>::ZDim>& q,
    const ZVar& Z) {
  return convex::make_quad_form(P, q, Z);
}

/**
 * @brief make PLDI problem with objective
 *
 * @param poly polytope vertices
 * @param Z merged decision var [svec(P); vec(Y)]
 * @param obj objective atom over merged var
 * @param margin PSD margin on P block
 *
 * @return PLDI optimization problem
 */
template <ContinuousPlantModelLike Plant, std::size_t N, convex::AtomLike ZVar>
constexpr auto make_polytopic_state_feedback_problem(
    const linalg::Vector<Plant, N>& poly, const ZVar& Z, const auto& obj,
    plant_type_t<Plant> margin = 1e-6) {
  const auto pldi = PolytopicStateFeedback<Plant, N>()(poly, Z, margin);
  return convex::make_problem(Z).in_set(pldi).minimize(obj);
}

/**
 * @brief make PLDI feasibility problem
 *
 * @param poly polytope vertices
 * @param Z merged decision var [svec(P); vec(Y)]
 * @param margin PSD margin on P block
 *
 * @return PLDI feasibility problem
 */
template <ContinuousPlantModelLike Plant, std::size_t N, convex::AtomLike ZVar>
constexpr auto make_polytopic_state_feedback_problem(
    const linalg::Vector<Plant, N>& poly, const ZVar& Z,
    plant_type_t<Plant> margin = 1e-6) {
  const auto pldi = PolytopicStateFeedback<Plant, N>()(poly, Z, margin);
  return convex::make_problem(Z).in_set(pldi).feasibility();
}

} // namespace control
