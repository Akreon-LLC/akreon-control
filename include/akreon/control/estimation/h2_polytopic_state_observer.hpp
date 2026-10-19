/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "akreon/control/estimation/h2_state_observer.hpp"
#include "akreon/convex/convex_problem.hpp"
#include "akreon/convex/intersection.hpp"
#include "akreon/convex/variable.hpp"

namespace control {

/**
 * @brief H2 Polytopic State Observer
 *
 * Uses a common H2 state-observer decision variable
 *   z = [svec(Q); vec(W); svec(Z)]
 *
 * across every plant vertex and intersects each vertex H2 certificate:
 *   Q : observability Gramian
 *   W : lifted observer product, W = Q L
 *   Z : shared H2 epigraph block
 */
template <ContinuousPlantModelLike Plant, std::size_t N>
struct H2PolytopicStateObserver {
  /// @brief extracting alias and types
  using Model = linalg::remove_cvref_t<Plant>;
  using T = plant_type_t<Model>;
  using VertexObserver = H2StateObserver<Model>;

  static constexpr std::size_t StateDim = VertexObserver::StateDim;
  static constexpr std::size_t ExogenousDim = VertexObserver::ExogenousDim;
  static constexpr std::size_t MeasurementDim = VertexObserver::MeasurementDim;
  static constexpr std::size_t PerformanceDim = VertexObserver::PerformanceDim;
  static constexpr std::size_t QDim = VertexObserver::QDim;
  static constexpr std::size_t WDim = VertexObserver::WDim;
  static constexpr std::size_t ZDim = VertexObserver::ZDim;
  static constexpr std::size_t DecVarDim = VertexObserver::DecVarDim;

  using QMat = typename VertexObserver::QMat;
  using WMat = typename VertexObserver::WMat;
  using LMat = typename VertexObserver::LMat;
  using ZMat = typename VertexObserver::ZMat;
  using DecVarSpace = typename VertexObserver::DecVarSpace;
  using Polytope = linalg::Vector<Plant, N>;

  /**
   * @brief operator overload to form H2 polytopic state observer
   *
   * @param poly (Polytope of plant models)
   * @param Z_hat (dec var)
   * @param margin (for PD)
   *
   * @return H2 LMIs (intersected)
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
   * @param margin (for PD)
   *
   * @return LMIs
   */
  template <convex::AtomLike DecVar>
  constexpr auto apply(const Polytope& poly, const DecVar& Z_hat,
                       T margin = 1e-6) const {
    static_assert(convex::is_var_type_v<DecVar>, "Input atom is not a var");
    using DecVarDom = convex::meta_dom_t<convex::atom_meta_t<DecVar>>;
    static_assert(convex::dom_vartype_v<DecVarDom> == convex::VarType::FREE,
                  "Merged decision var must be free");
    static_assert(std::same_as<DecVarSpace, convex::dom_space_t<DecVarDom>>,
                  "Expected merged decision size [svec(Q); vec(W); svec(Z)]");

    auto make_vertex_lmis = [&]<std::size_t idx>() {
      return VertexObserver()(poly(idx), Z_hat, margin);
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
   * @return Tr(Z)
   */
  template <convex::AtomLike DecVar>
  constexpr auto objective(const DecVar& Z_hat) const {
    return VertexObserver().objective(Z_hat);
  }

  /**
   * @brief L (observer gain)
   *
   * @param Q (Observability Gramian)
   * @param W (Lifted convex object)
   *
   * @return L
   */
  constexpr auto L(const QMat& Q, const WMat& W) const {
    return VertexObserver().L(Q, W);
  }
};

/**
 * @brief make polytopic state observer
 *
 * @param poly (Polytope of plant models)
 * @param Z_hat (Dec Var)
 * @param margin (for PD)
 *
 * @return convex problem
 */
template <ContinuousPlantModelLike Plant, std::size_t N,
          convex::AtomLike DecVar>
constexpr auto make_h2_polytopic_state_observer_problem(
    const linalg::Vector<Plant, N>& poly, const DecVar& Z_hat,
    plant_type_t<Plant> margin = 1e-6) {
  const auto h2 = H2PolytopicStateObserver<Plant, N>();
  const auto lmis = h2(poly, Z_hat, margin);
  const auto prob = convex::make_problem(Z_hat).in_set(lmis);
  const auto obj = h2.objective(Z_hat);
  return prob.minimize(obj);
}

} // namespace control
