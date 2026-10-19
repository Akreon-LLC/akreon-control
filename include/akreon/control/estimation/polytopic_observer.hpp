/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "akreon/linalg/assert.hpp"
#include "akreon/control/model/plant_model.hpp"
#include "akreon/convex/composed/quad_form.hpp"
#include "akreon/convex/convex_problem.hpp"
#include "akreon/convex/intersection.hpp"
#include "akreon/convex/sets/lmi.hpp"
#include "akreon/convex/variable.hpp"
#include "akreon/linalg/operator/matrix_inverse.hpp"
#include "akreon/linalg/operator/matrix_pencil.hpp"

namespace control {

/**
 * @brief Polytopic Luenberger Observer
 *
 * Given
 *   xdot = A(t) x + B(t) u
 *   y    = C(t) x
 *
 * design a state observer
 *   xhatdot = A(t) xhat + B(t) u + L (y - C(t) xhat)
 *
 * s.t. the estimation error
 *   e = x - xhat
 * satisfies
 *   edot = (A(t) - L C(t)) e
 *
 * There exists Q > 0 s.t.
 *   A_i^T Q + Q A_i - W C_i - C_i^T W^T < 0
 *
 * where W = Q L, and L = Q^{-1} W.
 */
template <ContinuousPlantModelLike Plant, std::size_t N>
struct PolytopicObserver {
  using Model = linalg::remove_cvref_t<Plant>;

  using T = plant_type_t<Model>;
  using State = plant_state_vec_t<Model>;

  static constexpr std::size_t StateDim = plant_traits<Model>::StateDim;
  static constexpr std::size_t OutputDim = plant_traits<Model>::MeasurementDim;

  using QMat = linalg::Matrix<T, StateDim, StateDim>;
  using WMat = linalg::Matrix<T, StateDim, OutputDim>;
  using LMat = linalg::Matrix<T, StateDim, OutputDim>;
  using FMat = QMat;

  static constexpr std::size_t QDim = StateDim * (StateDim + 1) / 2;
  static constexpr std::size_t WDim = OutputDim * StateDim;
  static constexpr std::size_t ZDim = QDim + WDim;

  using ZSpace = linalg::RowVector<T, ZDim>;
  using Fs = linalg::Vector<FMat, ZDim>;
  using Polytope = linalg::Vector<Plant, N>;

  /**
   * @brief operator overload (polytopic observer)
   *
   * @param poly (Polytope)
   * @param Z ([svec(Q); vec(W)])
   *
   * @return PLDI as intersection of LMIs
   */
  template <convex::AtomLike Var>
  constexpr auto operator()(const Polytope& poly, const Var& Z,
                            T margin = 1e-6) const {
    return apply(poly, Z, margin);
  }

  /**
   * @brief apply
   *
   * @param poly (Polytope)
   * @param Z ([svec(Q); vec(W)])
   *
   * @return PLDI as intersection of LMIs
   */
  template <convex::AtomLike Var>
  constexpr auto apply(const Polytope& poly, const Var& Z,
                       T margin = 1e-6) const {
    static_assert(convex::is_var_type_v<Var>, "Input atom is not a var");
    using ZDom = convex::meta_dom_t<convex::atom_meta_t<Var>>;
    static_assert(convex::dom_vartype_v<ZDom> == convex::VarType::FREE,
                  "Merged decision var must be free");
    using ZVarSpace = convex::dom_space_t<ZDom>;
    static_assert(std::same_as<ZVarSpace, ZSpace>,
                  "Expected merged decision size [svec(Q); vec(W)]");

    const auto q_lmi = q_constraint(Z, margin);

    auto make_vertex_lmi = [&]<std::size_t idx>() {
      return vertex_constraint(poly(idx), Z);
    };

    auto form_intersection = [&]<std::size_t... Is>(
                                 std::index_sequence<Is...>) {
      return convex::Intersection(q_lmi,
                                  make_vertex_lmi.template operator()<Is>()...);
    };

    return form_intersection(std::make_index_sequence<N>{});
  }

  /**
   * @brief q_constraint
   *
   * @param Z ([svec(Q); vec(W)])
   * @param margin (PDness)
   *
   * @return LMI for Q
   */
  template <convex::AtomLike ZVar>
  constexpr auto q_constraint(const ZVar& Z, T margin = 1e-6) const {
    const auto I = linalg::make_identity<T, StateDim>();
    const auto Fq =
        linalg::MatrixPencil<QMat, QDim>{}.template apply_sym_mat<QMat>(
            [](const QMat& E) -> QMat { return E; });

    Fs F_blks{};
    for (std::size_t i = 0; i < QDim; ++i) F_blks(i) = Fq(i);
    for (std::size_t i = 0; i < WDim; ++i) F_blks(QDim + i) = QMat{};

    return convex::LMI(F_blks, T(-1) * margin * I, Z);
  }

  /**
   * @brief vertex constraint
   *
   * @param plant (ModelPlant)
   * @param Z (dec var)
   *
   * @return LMI for the stability of each vertex
   */
  template <convex::AtomLike ZVar>
  constexpr auto vertex_constraint(const Plant& plant, const ZVar& Z) const {
    auto M_q = [&](const QMat& E) -> FMat {
      const auto A = plant.dynamics_matrix();
      return T(-1) * (linalg::transpose(A) * E + E * A);
    };

    auto M_w = [&](const WMat& E) -> FMat {
      const auto C_y = plant.measurement_matrix();
      return E * C_y + linalg::transpose(C_y) * linalg::transpose(E);
    };

    const auto Fq_local =
        linalg::MatrixPencil<FMat, QDim>{}.template apply_sym_mat<QMat>(M_q);
    const auto Fw_local =
        linalg::MatrixPencil<FMat, WDim>{}.template apply_full<WMat>(M_w);

    Fs F_blks{};
    for (std::size_t i = 0; i < QDim; ++i) F_blks(i) = Fq_local(i);
    for (std::size_t i = 0; i < WDim; ++i) F_blks(QDim + i) = Fw_local(i);

    return convex::LMI(F_blks, FMat{}, Z);
  }

  /**
   * @brief L
   *
   * L = Q^{-1} W
   *
   * @param Q (symmetric observability Gramian)
   * @param W (lifted luenberger gain W = Q L)
   *
   * @return L
   */
  constexpr auto L(const QMat& Q, const WMat& W) const {
    const auto [Q_inv, event] = linalg::SymmetricMatrixInverse<QMat>()(Q);
    if (event != linalg::NumericalEvent::NONE) return std::tuple(LMat{}, event);
    const auto L = Q_inv * W;
    return std::tuple(L, linalg::NumericalEvent::NONE);
  }
};

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
constexpr auto make_polytopic_observer_quadratic_objective(
    const linalg::Matrix<plant_type_t<Plant>, PolytopicObserver<Plant, N>::ZDim,
                         PolytopicObserver<Plant, N>::ZDim>& P,
    const linalg::Vector<plant_type_t<Plant>,
                         PolytopicObserver<Plant, N>::ZDim>& q,
    const ZVar& Z) {
  return convex::make_quad_form(P, q, Z);
}

/**
 * @brief make PLDI problem with objective
 *
 * @param poly polytope vertices
 * @param Z merged decision var [svec(Q); vec(W)]
 * @param obj objective atom over merged var
 * @param margin PSD margin on Q block
 *
 * @return PLDI optimization problem
 */
template <ContinuousPlantModelLike Plant, std::size_t N, convex::AtomLike ZVar>
constexpr auto make_polytopic_observer_problem(
    const linalg::Vector<Plant, N>& poly, const ZVar& Z, const auto& obj,
    plant_type_t<Plant> margin = 1e-6) {
  const auto pldi = PolytopicObserver<Plant, N>()(poly, Z, margin);
  return convex::make_problem(Z).in_set(pldi).minimize(obj);
}

/**
 * @brief make PLDI feasibility problem
 *
 * @param poly polytope vertices
 * @param Z merged decision var [svec(Q); vec(W)]
 * @param margin PSD margin on Q block
 *
 * @return PLDI feasibility problem
 */
template <ContinuousPlantModelLike Plant, std::size_t N, convex::AtomLike ZVar>
constexpr auto make_polytopic_observer_problem(
    const linalg::Vector<Plant, N>& poly, const ZVar& Z,
    plant_type_t<Plant> margin = 1e-6) {
  const auto pldi = PolytopicObserver<Plant, N>()(poly, Z, margin);
  return convex::make_problem(Z).in_set(pldi).feasibility();
}

} // namespace control
