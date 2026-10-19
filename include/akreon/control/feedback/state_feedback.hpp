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
 * @brief StateFeedback
 *
 * Linear Differential Inclusion (LDI) State Feedback
 * Given the LTI system
 *   xdot = A x + B_u u
 *
 * We seek a static state-feedback
 *   u = K x
 *
 * such that the closed-loop system is quadratically stable.
 *
 * There exists S > 0 such that
 *   (A + B_u K)^T S + S (A + B_u K) < 0
 *
 * Equivalently, letting P = S^{-1} > 0,
 *   P (A + B_u K)^T + (A + B_u K) P < 0
 *
 * This is bilinear in P and K, so define
 *   Y = K P
 * hence
 *   K = Y P^{-1}
 *
 * The problem is represented over the merged decision variable
 *   z = [ svec(P); vec(Y) ].
 */
template <ContinuousPlantModelLike SFModel>
struct StateFeedback {
  /// @brief extracting alias and types
  using Model = linalg::remove_cvref_t<SFModel>;

  using T = plant_type_t<Model>;
  using State = plant_state_vec_t<Model>;

  static constexpr std::size_t StateDim = plant_traits<Model>::StateDim;
  static constexpr std::size_t InputDim = plant_traits<Model>::ControlDim;

  using PMat = linalg::Matrix<T, StateDim, StateDim>;
  using YMat = linalg::Matrix<T, InputDim, StateDim>;
  using KMat = linalg::Matrix<T, InputDim, StateDim>;
  using FMat = PMat;

  static constexpr std::size_t PDim = StateDim * (StateDim + 1) / 2;
  static constexpr std::size_t YDim = InputDim * StateDim;
  static constexpr std::size_t ZDim = PDim + YDim;

  using ZSpace = linalg::RowVector<T, ZDim>;
  using Fs = linalg::Vector<FMat, ZDim>;

  /**
   * @brief operator overload (form state feedback)
   *
   * @param model (StateSpace)
   * @param Z (dec var [svec(P); vec(Y)])
   *
   * @return state-feedback synthesis constraints
   */
  template <convex::AtomLike ZVar>
  constexpr auto operator()(const SFModel& model, const ZVar& Z,
                            T margin = 1e-6) const {
    return apply(model, Z, margin);
  }

  /**
   * @brief apply
   *
   * @param model (StateSpace)
   * @param Z (dec var [svec(P); vec(Y)])
   * @param margin (for PD)
   *
   * @return state-feedback synthesis constraints
   */
  template <convex::AtomLike ZVar>
  constexpr auto apply(const SFModel& model, const ZVar& Z,
                       T margin = 1e-6) const {
    static_assert(convex::is_var_type_v<ZVar>, "Input atom is not a var");
    using ZDom = convex::meta_dom_t<convex::atom_meta_t<ZVar>>;
    static_assert(convex::dom_vartype_v<ZDom> == convex::VarType::FREE,
                  "Merged decision var must be free");
    using ZVarSpace = convex::dom_space_t<ZDom>;
    static_assert(std::same_as<ZVarSpace, ZSpace>,
                  "Expected merged decision size [svec(P); vec(Y)]");

    const auto stability_lmi = stability_constraint(model, Z);
    const auto p_lmi = p_constraint(Z, margin);
    return convex::Intersection(p_lmi, stability_lmi);
  }

  /**
   * @brief p_constraint
   *
   * @param Z (dec var)
   * @param margin (PD)
   *
   * @return P constraint as LMI
   */
  template <convex::AtomLike ZVar>
  constexpr auto p_constraint(const ZVar& Z, T margin = 1e-6) const {
    const auto I = linalg::make_identity<T, StateDim>();
    const auto Fp_psd =
        linalg::MatrixPencil<PMat, PDim>{}.template apply_sym_mat<PMat>(
            [](const PMat& E) -> PMat { return E; });

    Fs P_blks{};
    for (std::size_t i = 0; i < PDim; ++i) P_blks(i) = Fp_psd(i);
    for (std::size_t i = 0; i < YDim; ++i) P_blks(PDim + i) = PMat{};

    return convex::LMI(P_blks, T(-1) * margin * I, Z);
  }

  /**
   * @brief stability_constraint
   *
   * @param model (PlantModel)
   * @param Z (Dec Var)
   *
   * @return stability constraint as LMI
   */
  template <convex::AtomLike ZVar>
  constexpr auto stability_constraint(const SFModel& model,
                                      const ZVar& Z) const {
    auto M = [&](const PMat& E) -> FMat {
      const auto A = model.dynamics_matrix();
      return T(-1) * (A * E + E * linalg::transpose(A));
    };

    auto M_y = [&](const YMat& E) -> FMat {
      const auto B_u = model.control_matrix();
      return T(-1) * (B_u * E + linalg::transpose(E) * linalg::transpose(B_u));
    };

    const auto Fp_local =
        linalg::MatrixPencil<FMat, PDim>{}.template apply_sym_mat<PMat>(M);
    const auto Fy_local =
        linalg::MatrixPencil<FMat, YDim>{}.template apply_full<YMat>(M_y);

    Fs F_blks{};
    for (std::size_t i = 0; i < PDim; ++i) F_blks(i) = Fp_local(i);
    for (std::size_t i = 0; i < YDim; ++i) F_blks(PDim + i) = Fy_local(i);

    return convex::LMI(F_blks, FMat{}, Z);
  }

  /**
   * @brief K
   *
   * K = Y P^{-1}
   *
   * @param P (PMat)
   * @param Y (YMat)
   *
   * @return K
   */
  constexpr auto K(const PMat& P, const YMat& Y) const {
    return recover_state_feedback_gain(P, Y);
  }
};

/**
 * @brief make state feedback problem
 *
 * @param model (StateSpace)
 * @param Z (merged decision var [svec(P); vec(Y)])
 * @param obj (objective)
 * @param margin (PSD margin)
 *
 * @return state feedback problem
 */
template <ContinuousPlantModelLike SFModel, convex::AtomLike ZVar>
constexpr auto make_state_feedback_problem(
    const SFModel& model, const ZVar& Z, const auto& obj,
    plant_type_t<SFModel> margin = 1e-6) {
  const auto sf = StateFeedback<decltype(model)>();
  const auto constraints = sf(model, Z, margin);
  const auto prob_constraints = convex::make_problem(Z).in_set(constraints);
  return prob_constraints.minimize(obj);
}

/**
 * @brief make state feedback feasibility problem
 *
 * @param model (StateSpace)
 * @param Z (merged decision var [svec(P); vec(Y)])
 * @param margin (PSD margin)
 *
 * @return state feedback feasibility problem
 */
template <ContinuousPlantModelLike SFModel, convex::AtomLike ZVar>
constexpr auto make_state_feedback_problem(
    const SFModel& model, const ZVar& Z, plant_type_t<SFModel> margin = 1e-6) {
  const auto sf = StateFeedback<decltype(model)>();
  const auto constraints = sf(model, Z, margin);
  return convex::make_problem(Z).in_set(constraints).feasibility();
}

} // namespace control
