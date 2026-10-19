/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "akreon/linalg/assert.hpp"
#include "akreon/control/feedback/gain_recovery.hpp"
#include "akreon/control/model/plant_model.hpp"
#include "akreon/control/model/state_space.hpp"
#include "akreon/convex/convex_problem.hpp"
#include "akreon/convex/intersection.hpp"
#include "akreon/convex/sets/lmi.hpp"
#include "akreon/convex/variable.hpp"
#include "akreon/linalg/operator/matrix_inverse.hpp"
#include "akreon/linalg/operator/matrix_pencil.hpp"

namespace control {

/**
 * @brief Luenberger Observer
 *
 * Given
 *   xdot = A x + B u
 *   y    = C x
 *
 * design a state observer
 *   xhatdot = A xhat + B u + L (y - C xhat)
 *
 * such that the estimation error
 *   e = x - xhat
 * satisfies
 *   edot = (A - L C) e
 * and the error dynamics are asymptotically stable
 * (or satisfy the chosen performance certificate).
 *
 * There exists Q > 0 s.t.
 *   A^T Q + Q A - W C - C^T W^T < 0
 *
 * where W = Q L, and L = Q^{-1} W.
 */
template <ContinuousPlantModelLike ObsModel>
struct LuenbergerObserver {
  /// @brief extracting alias and types
  using Model = linalg::remove_cvref_t<ObsModel>;

  using T = plant_type_t<Model>;
  using State = plant_state_vec_t<Model>;

  static constexpr std::size_t StateDim = plant_traits<Model>::StateDim;
  static constexpr std::size_t OutputDim = plant_traits<Model>::MeasurementDim;

  using QMat = linalg::Matrix<T, StateDim, StateDim>;
  using WMat = linalg::Matrix<T, StateDim, OutputDim>;
  using LMat = linalg::Matrix<T, StateDim, OutputDim>;
  using FMat = QMat;

  static constexpr std::size_t QDim = StateDim * (StateDim + 1) / 2;
  static constexpr std::size_t WDim = StateDim * OutputDim;
  static constexpr std::size_t ZDim = QDim + WDim;

  using ZSpace = linalg::RowVector<T, ZDim>;
  using Fs = linalg::Vector<FMat, ZDim>;

  /**
   * @brief operator overload (form luenberger observer)
   *
   * @param model (StateSpace)
   * @param Z (dec var [svec(Q); vec(W)])
   *
   * @return observer synthesis constraints
   */
  template <convex::AtomLike ZVar>
  constexpr auto operator()(const ObsModel& model, const ZVar& Z,
                            T margin = 1e-6) const {
    return apply(model, Z, margin);
  }

  /**
   * @brief apply
   *
   * @param model (StateSpace)
   * @param Z (dec var [svec(Q); vec(W)])
   *
   * @return observer synthesis constraints
   */
  template <convex::AtomLike ZVar>
  constexpr auto apply(const ObsModel& model, const ZVar& Z,
                       T margin = 1e-6) const {
    static_assert(convex::is_var_type_v<ZVar>, "Input atom is not a var");
    using ZDom = convex::meta_dom_t<convex::atom_meta_t<ZVar>>;
    static_assert(convex::dom_vartype_v<ZDom> == convex::VarType::FREE,
                  "Merged decision var must be free");
    using ZVarSpace = convex::dom_space_t<ZDom>;
    static_assert(std::same_as<ZVarSpace, ZSpace>,
                  "Expected merged decision size [svec(Q); vec(W)]");

    const auto stability_lmi = stability_constraint(model, Z);
    const auto q_lmi = q_constraint(Z, margin);
    return convex::Intersection(q_lmi, stability_lmi);
  }

  /**
   * @brief q_constraint
   *
   * @param Z (dec var [svec(Q); vec(W)])
   * @param margin (PD)
   *
   * @return Q constraint as LMI
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
   * @brief stability_constraint
   *
   * @param model (PlantModel)
   * @param Z (dec var [svec(Q); vec(W)])
   *
   * @return stability constraint as LMI
   */
  template <convex::AtomLike ZVar>
  constexpr auto stability_constraint(const ObsModel& model,
                                      const ZVar& Z) const {
    auto M_q = [&](const QMat& E) -> FMat {
      const auto& A = model.dynamics_matrix();
      return T(-1) * (linalg::transpose(A) * E + E * A);
    };

    auto M_w = [&](const WMat& E) -> FMat {
      const auto C_y = model.measurement_matrix();
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
   * @param Q (QMat)
   * @param W (lifted luenberger gain W = Q L)
   *
   * @return L
   */
  constexpr auto L(const QMat& Q, const WMat& W) const {
    return recover_state_observer_gain(Q, W);
  }
};

/**
 * @brief make observer problem
 *
 * @param model (StateSpace)
 * @param Q (Var)
 * @param obj (Objective)
 * @param margin (PSD margin)
 *
 * @return observer problem
 */
template <ContinuousPlantModelLike ObsModel, convex::AtomLike ZVar>
constexpr auto make_observer_problem(const ObsModel& model, const ZVar& Z,
                                     const auto& obj,
                                     plant_type_t<ObsModel> margin = 1e-6) {
  const auto obs = LuenbergerObserver<decltype(model)>();
  const auto constraints = obs(model, Z, margin);

  const auto prob_constraints = convex::make_problem(Z).in_set(constraints);

  return prob_constraints.minimize(obj);
}

/**
 * @brief make observer problem
 *
 * @param model (StateSpace)
 * @param Q (Var)
 * @param margin (PSD margin)
 *
 * @return observer problem
 */
template <ContinuousPlantModelLike ObsModel, convex::AtomLike ZVar>
constexpr auto make_observer_problem(const ObsModel& model, const ZVar& Z,
                                     plant_type_t<ObsModel> margin = 1e-6) {
  const auto obs = LuenbergerObserver<decltype(model)>();
  const auto constraints = obs(model, Z, margin);

  const auto prob_constraints = convex::make_problem(Z).in_set(constraints);

  return prob_constraints.feasibility();
}

} // namespace control
