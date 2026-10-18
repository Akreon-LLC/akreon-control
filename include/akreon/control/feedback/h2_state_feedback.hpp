/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "akreon/linalg/assert.hpp"
#include "akreon/control/feedback/gain_recovery.hpp"
#include "akreon/control/model/plant_model.hpp"
#include "akreon/convex/composed/trace.hpp"
#include "akreon/convex/convex_problem.hpp"
#include "akreon/convex/intersection.hpp"
#include "akreon/convex/sets/lmi.hpp"
#include "akreon/convex/variable.hpp"
#include "akreon/linalg/operator/matrix_inverse.hpp"
#include "akreon/linalg/operator/matrix_pencil.hpp"

namespace control {

/**
 * @brief H2 State Feedback
 *
 * LDI governed by H2 norm
 *
 * Given the LTI System with additive disturbance
 *   xdot = A x + B_u u + B_w w
 *   y = C x + D_u u
 *
 * We seek a static feedback controller u = Kx s.t.
 *   xdot = (A + B_u K) x + B_w w... where (A + B_u K) is the cled loop Plant
 *   y = (C + D_u K) x... where (C + D_u K) is the closed loop performance
 * output
 *
 * The controllability gramian (Q) requirement
 *   A_cl Q + Q A_cl^T + Bw Bw^T < 0
 *
 * The H2 norm squared (total impulse response energy)
 *   ||T_{wz}||_2^2 = Tr(C_cl Q C_cl^T)
 *
 * Dimensions:
 *   n   = state dimension
 *   m_u = control input dimension
 *   m_w = disturbance input dimension
 *   p_z = performance output dimension
 *
 *   A   : n x n
 *   B_u : n x m_u
 *   B_w : n x m_w
 *   C_z : p_z x n
 *   D_u : p_z x m_u
 *
 * Decision blocks:
 *   Q : n x n, synthesis Lyapunov/Gramian variable
 *   Y  : m_u x n
 *   K  : m_u x n
 *   Z  : p_z x p_z
 *   DecVar = [svec(Q); vec(Y); svec(Z)]
 *
 * We have a bilinear term in K Q, so let Y = K Q where K = Y Q^-1
 *
 * H2 objective can be lifted via epigraph with Z
 *   Z >= (C Q + D_u Y) Q^{-1} (C Q + Du Y)^T
 *
 * Convex Problem is now
 *   min  Tr(Z)
 *   s.t. A Q + Q A^T + B_u Y + Y^T B_u^T + B_w B_w^T < 0
 *        [ Z                C Q + Du Y ] >= 0
 *        [ (C Q + Du Y)^T  Q          ]
 *
 * Shape checks:
 *   A Q, Q A^T, B_u Y, Y^T B_u^T, B_w B_w^T : n x n
 *   C_z Q + D_u Y : p_z x n
 *   Epigraph LMI   : (p_z + n) x (p_z + n)
 */
template <ContinuousPlantModelLike SFModel>
struct H2StateFeedback {
  /// @brief extracting alias and types
  using Model = linalg::remove_cvref_t<SFModel>;

  using T = plant_type_t<Model>;

  static constexpr std::size_t StateDim = plant_traits<Model>::StateDim;
  static constexpr std::size_t InputDim = plant_traits<Model>::ControlDim;
  static constexpr std::size_t OutputDim = plant_traits<Model>::PerformanceDim;

  using QMat = linalg::Matrix<T, StateDim, StateDim>;
  using YMat = linalg::Matrix<T, InputDim, StateDim>;
  using KMat = linalg::Matrix<T, InputDim, StateDim>;
  using ZMat = linalg::Matrix<T, OutputDim, OutputDim>;

  static constexpr std::size_t QDim = StateDim * (StateDim + 1) / 2;
  static constexpr std::size_t YDim = InputDim * StateDim;
  static constexpr std::size_t ZDim = OutputDim * (OutputDim + 1) / 2;
  static constexpr std::size_t DecVarDim = QDim + YDim + ZDim;

  using DecVarSpace = linalg::RowVector<T, DecVarDim>;

  /**
   * @brief operator overload to form H2 state feedback
   *
   * @param model (PlantModel)
   * @param Z_hat (DecVar)
   *
   * @return H2 LMIs
   */
  template <convex::AtomLike DecVar>
  constexpr auto operator()(const SFModel& model, const DecVar& Z_hat,
                            T margin = 1e-6) const {
    return apply(model, Z_hat, margin);
  }

  /**
   * @brief apply
   *
   * @param model (PlantModel)
   * @param Z_hat ([svec(Q); vec(Y); svec(Z)])
   *
   * @return H2 LMI
   */
  template <convex::AtomLike DecVar>
  constexpr auto apply(const SFModel& model, const DecVar& Z_hat,
                       T margin = 1e-6) const {
    static_assert(convex::is_var_type_v<DecVar>, "Input atom is not a var");
    using DecVarDom = convex::meta_dom_t<convex::atom_meta_t<DecVar>>;
    static_assert(convex::dom_vartype_v<DecVarDom> == convex::VarType::FREE,
                  "Merged decision var must be free");
    static_assert(std::same_as<DecVarSpace, convex::dom_space_t<DecVarDom>>,
                  "Expected merged decision size [svec(Q); vec(Y); svec(Z)]");

    const auto stability_lmi = stability_constraint(model, Z_hat);
    const auto z_lmi = z_constraint(model, Z_hat, margin);
    return convex::Intersection(z_lmi, stability_lmi);
  }

  /**
   * @brief stability_constraint
   *
   * Stability constraint on LMI:
   *   A Q + Q A^T + B_u Y + Y^T B_u^T + B_w B_w^T < 0
   *
   * @param model (PlantModel)
   * @param Z_hat (Dec Var)
   *
   * @return stability constraint as LMI
   */
  template <convex::AtomLike DecVar>
  constexpr auto stability_constraint(const SFModel& model,
                                      const DecVar& Z_hat) const {
    using FMat = QMat;
    using Fs = linalg::Vector<FMat, DecVarDim>;

    auto M_q = [&](const QMat& E) -> FMat {
      const auto& A = model.dynamics_matrix();
      return T(-1) * (A * E + E * linalg::transpose(A));
    };

    auto M_y = [&](const YMat& E) -> FMat {
      const auto& B_u = model.control_matrix();
      return T(-1) * (B_u * E + linalg::transpose(E) * linalg::transpose(B_u));
    };

    const auto B_w = model.exogenous_matrix();
    const auto F_0 = T(-1) * (B_w * linalg::transpose(B_w));

    const auto F_q_local =
        linalg::MatrixPencil<FMat, QDim>{}.template apply_sym_mat<QMat>(M_q);
    const auto F_y_local =
        linalg::MatrixPencil<FMat, YDim>{}.template apply_full<YMat>(M_y);

    Fs F_blks{};
    for (std::size_t i = 0; i < QDim; ++i) F_blks(i) = F_q_local(i);
    for (std::size_t i = 0; i < YDim; ++i) F_blks(QDim + i) = F_y_local(i);

    return convex::LMI(F_blks, F_0, Z_hat);
  }

  /**
   * @brief z_constraint
   *
   * PSD constraint on Z (epigraph)
   *   [ Z                C Q + Du Y ] >= 0
   *   [ (C Q + Du Y)^T  Q          ]
   *
   * FMat Dim (p+n x p+n)
   *   [ pxp pxn ]
   *   [ nxp nxn ]
   *
   * @param model (PlantModel)
   * @param Z_hat (DecVar)
   * @param margin for PD
   *
   * @return Z constraint as LMI
   */
  template <convex::AtomLike DecVar>
  constexpr auto z_constraint(const SFModel& model, const DecVar& Z_hat,
                              T margin = 1e-6) const {
    constexpr std::size_t LmiDim = OutputDim + StateDim;
    using FMat = linalg::Matrix<T, LmiDim, LmiDim>;
    using Fs = linalg::Vector<FMat, DecVarDim>;
    auto M_z = [](const ZMat& E) -> FMat {
      FMat F{};
      return linalg::set_block<0, OutputDim, 0, OutputDim>(F, E);
    };
    auto M_q = [&](const QMat& E) -> FMat {
      FMat F{};
      const auto C = model.performance_matrix();
      F = linalg::set_block<OutputDim, LmiDim, OutputDim, LmiDim>(F, E);
      F = linalg::set_block<0, OutputDim, OutputDim, LmiDim>(F, C * E);
      F = linalg::set_block<OutputDim, LmiDim, 0, OutputDim>(
          F, E * linalg::transpose(C));
      return F;
    };
    auto M_y = [&](const YMat& E) -> FMat {
      FMat F{};
      const auto D_u =
          model
              .template feedthrough<PortRole::PERFORMANCE, PortRole::CONTROL>();
      F = linalg::set_block<0, OutputDim, OutputDim, LmiDim>(F, D_u * E);
      F = linalg::set_block<OutputDim, LmiDim, 0, OutputDim>(
          F, linalg::transpose(E) * linalg::transpose(D_u));
      return F;
    };

    const auto F_z_local =
        linalg::MatrixPencil<FMat, ZDim>{}.template apply_sym_mat<ZMat>(M_z);
    const auto F_q_local =
        linalg::MatrixPencil<FMat, QDim>{}.template apply_sym_mat<QMat>(M_q);
    const auto F_y_local =
        linalg::MatrixPencil<FMat, YDim>{}.template apply_full<YMat>(M_y);

    Fs F_blks{};
    for (std::size_t i = 0; i < QDim; ++i) F_blks(i) = F_q_local(i);
    for (std::size_t i = 0; i < YDim; ++i) F_blks(QDim + i) = F_y_local(i);
    for (std::size_t i = 0; i < ZDim; ++i)
      F_blks(QDim + YDim + i) = F_z_local(i);

    const auto I = linalg::make_identity<T, LmiDim>();
    return convex::LMI(F_blks, T(-1) * margin * I, Z_hat);
  }

  /**
   * @brief objective
   *
   * min Tr(Z)
   *
   * @param Z_hat (DecVar)
   *
   * @return Tr(Z)
   */
  template <convex::AtomLike DecVar>
  constexpr auto objective(const DecVar& Z_hat) const {
    return convex::make_trace_block<QDim + YDim, linalg::SymmetricMatrix<ZMat>>(
        Z_hat);
  }

  /**
   * @brief K
   *
   * K = Y Q^{-1}
   *
   * @param Q controllability gramian
   * @param Y lifted feedback product Y = K Q
   *
   * @return K
   */
  constexpr auto K(const QMat& Q, const YMat& Y) const {
    return recover_state_feedback_gain(Q, Y);
  }
};

/**
 * @brief make h2 state feedback problem
 *
 * @param model (PlantModel)
 * @param Z (DecVar)
 * @param margin (PSD Margin)
 *
 * @return h2 state feedback problem
 */
template <ContinuousPlantModelLike SFModel, convex::AtomLike DecVar>
constexpr auto make_h2_state_feedback_problem(
    const SFModel& model, const DecVar& Z_hat,
    plant_type_t<SFModel> margin = 1e-6) {
  const auto h2 = H2StateFeedback<decltype(model)>();
  const auto lmis = h2(model, Z_hat, margin);
  const auto prob = convex::make_problem(Z_hat).in_set(lmis);
  const auto obj = h2.objective(Z_hat);
  return prob.minimize(obj);
}

} // namespace control
