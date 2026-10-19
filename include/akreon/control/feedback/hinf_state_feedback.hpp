/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "akreon/linalg/assert.hpp"
#include "akreon/control/feedback/gain_recovery.hpp"
#include "akreon/control/model/plant_model.hpp"
#include "akreon/convex/composed/weighted_sum.hpp"
#include "akreon/convex/convex_problem.hpp"
#include "akreon/convex/intersection.hpp"
#include "akreon/convex/sets/lmi.hpp"
#include "akreon/convex/variable.hpp"
#include "akreon/linalg/operator/matrix_inverse.hpp"
#include "akreon/linalg/operator/matrix_pencil.hpp"

namespace control {

/**
 * @brief Hinf State Feedback
 *
 * We seek a state-feedback gain K s.t. L2 gain
 *   sup_{||w||_2 = 1} = ||z||_2 = sup_{||w||_2!=0}frac{||z||_2}{||w||_2}
 *
 * of the CL system is less than an arbitrary gamma
 *
 * L2 gain for LTI is equal to H_infinity norm of the corresponding TF. exists K
 * s.t. L2 gain of an LTI system is less than gamma if the inverse BRL
 * Lyapunov variable Q > 0.
 *   xdot = A x + B_xw w + B_xu u
 *   z    = C_zx x + D_zw w + D_zu u
 *   y    = C_yx x + D_yw w + D_yu u
 *
 * Plant =
 *   [ A    B_xw B_xu ]
 *   [ C_zx D_zw D_zu ]
 *   [ C_yx D_yw D_yu ]
 *
 * let u = K x
 *
 * xdot = (A + B_xu K) x + B_xw w
 * z    = (C_zx + D_zu K) x + D_zw w
 *
 * A_cl = A + B_xu K
 * B_cl = B_xw
 * C_cl = C_zx + D_zu K
 * D_cl = D_zw
 *
 * Certifies ||T_zw||_infinity < gamma where T_zw is the complementary
 * sensitivity TF from w to z
 *
 * General BRL form with Lyapunov variable S > 0:
 *   [ (A_cl^T S + S A_cl)  (S B_xw)    C_cl^T    ] < 0
 *   [ (B_xw^T S)           (-gamma I) D_zw^T    ]
 *   [ C_cl                 D_zw        -gamma I ]
 *
 * Hinf synthesis decision variables:
 *   Q     : n x n, inverse BRL Lyapunov variable Q = S^{-1}
 *   Y     : m_u x n, lifted feedback product Y = K Q
 *   K     : m_u x n
 *   gamma : scalar induced L2-gain bound
 *   DecVar = [svec(Q); vec(Y); gamma]
 *
 * Objective:
 *   min gamma
 *
 * We have a bilinear term in K Q, so let Y = K Q where K = Y Q^-1
 *   [ A Q + Q A^T + B_xu Y + Y^T B_xu^T  B_xw         (C_zx Q + D_zu Y)^T ] < 0
 *   [ B_xw^T                             (-gamma I)  D_zw^T              ]
 *   [ C_zx Q + D_zu Y                    D_zw         (-gamma I)         ]
 *
 * Convex problem:
 *   min  gamma
 *   s.t. Q > 0
 *        BRL(Q, Y, gamma) < 0
 *
 * Full decision space:
 *   QDim       = n(n + 1) / 2
 *   YDim       = m_u n
 *   GammaDim   = 1
 *   DecVarDim  = QDim + YDim + GammaDim
 *   DecVar     in R^DecVarDim
 *
 * Shape checks:
 *   A Q, Q A^T, B_xu Y, Y^T B_xu^T : n x n
 *   B_xw                           : n x m_w
 *   C_zx Q + D_zu Y                : p_z x n
 *   D_zw                           : p_z x m_w
 *   gamma I_w                      : m_w x m_w
 *   gamma I_z                      : p_z x p_z
 *   BRL LMI                        : (n + m_w + p_z) x (n + m_w + p_z)
 */
template <ContinuousPlantModelLike SFModel>
struct HinfStateFeedback {
  /// @brief extracting alias and types
  using Model = linalg::remove_cvref_t<SFModel>;

  using T = plant_type_t<Model>;

  static constexpr std::size_t StateDim = plant_traits<Model>::StateDim;
  static constexpr PortRole ExogenousPortRole = PortRole::EXOGENOUS;
  static constexpr std::size_t InputDim = plant_traits<Model>::ControlDim;
  static constexpr std::size_t ExogenousDim = plant_traits<Model>::ExogenousDim;
  static constexpr std::size_t OutputDim = plant_traits<Model>::PerformanceDim;

  using QMat = linalg::Matrix<T, StateDim, StateDim>;
  using YMat = linalg::Matrix<T, InputDim, StateDim>;
  using KMat = linalg::Matrix<T, InputDim, StateDim>;

  static constexpr std::size_t QDim = StateDim * (StateDim + 1) / 2;
  static constexpr std::size_t YDim = InputDim * StateDim;
  static constexpr std::size_t GammaDim = 1;
  static constexpr std::size_t DecVarDim = QDim + YDim + GammaDim;

  using DecVarSpace = linalg::RowVector<T, DecVarDim>;

  /**
   * @brief operator overload to form Hinf state feedback
   *
   * @param model (PlantModel)
   * @param Z_hat (DecVar)
   * @param margin (PDness)
   *
   * @return Hinf LMIs
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
   * @param Z_hat ([svec(Q); vec(Y); gamma])
   * @param margin (PDness)
   *
   * @return Hinf synthesis constraints
   */
  template <convex::AtomLike DecVar>
  constexpr auto apply(const SFModel& model, const DecVar& Z_hat,
                       T margin = 1e-6) const {
    static_assert(convex::is_var_type_v<DecVar>, "Input atom is not a var");
    using DecVarDom = convex::meta_dom_t<convex::atom_meta_t<DecVar>>;
    static_assert(convex::dom_vartype_v<DecVarDom> == convex::VarType::FREE,
                  "Merged decision var must be free");
    static_assert(std::same_as<DecVarSpace, convex::dom_space_t<DecVarDom>>,
                  "Expected merged decision size [svec(Q); vec(Y); gamma]");

    const auto q_lmi = q_constraint(Z_hat, margin);
    const auto brl_lmi = brl_constraint(model, Z_hat, margin);
    return convex::Intersection(q_lmi, brl_lmi);
  }

  /**
   * @brief q_constraint
   *
   * PSD constraint on the synthesis Lyapunov block:
   *   Q >= margin I
   *
   * @param Z_hat (DecVar)
   * @param margin (PDness)
   *
   * @return Q constraint as LMI
   */
  template <convex::AtomLike DecVar>
  constexpr auto q_constraint(const DecVar& Z_hat, T margin = 1e-6) const {
    using FMat = QMat;
    using Fs = linalg::Vector<FMat, DecVarDim>;

    const auto F_q =
        linalg::MatrixPencil<FMat, QDim>{}.template apply_sym_mat<QMat>(
            [](const QMat& E) -> FMat { return E; });

    Fs F_blks{};
    for (std::size_t i = 0; i < QDim; ++i) F_blks(i) = F_q(i);

    const auto I = linalg::make_identity<T, StateDim>();
    return convex::LMI(F_blks, T(-1) * margin * I, Z_hat);
  }

  /**
   * @brief brl_constraint
   *
   * PSD constraint on the negated inverse-BRL block:
   *   -BRL(Q, Y, gamma) >= margin I
   *
   * FMat Dim (n+m_w+p_z x n+m_w+p_z)
   *
   * @param model (PlantModel)
   * @param Z_hat (DecVar)
   * @param margin (PDness)
   *
   * @return BRL constraint as LMI
   */
  template <convex::AtomLike DecVar>
  constexpr auto brl_constraint(const SFModel& model, const DecVar& Z_hat,
                                T margin = 1e-6) const {
    constexpr std::size_t StateEnd = StateDim;
    constexpr std::size_t DisturbanceEnd = StateDim + ExogenousDim;
    constexpr std::size_t LmiDim = StateDim + ExogenousDim + OutputDim;

    using FMat = linalg::Matrix<T, LmiDim, LmiDim>;
    using Fs = linalg::Vector<FMat, DecVarDim>;

    auto M_q = [&](const QMat& E) -> FMat {
      FMat F{};
      const auto A = model.dynamics_matrix();
      const auto C_z = model.performance_matrix();
      F = linalg::set_block<0, StateEnd, 0, StateEnd>(
          F, T(-1) * (A * E + E * linalg::transpose(A)));
      F = linalg::set_block<0, StateEnd, DisturbanceEnd, LmiDim>(
          F, T(-1) * linalg::transpose(C_z * E));
      F = linalg::set_block<DisturbanceEnd, LmiDim, 0, StateEnd>(
          F, T(-1) * C_z * E);
      return F;
    };

    auto M_y = [&](const YMat& E) -> FMat {
      FMat F{};
      const auto B_u = model.control_matrix();
      const auto D_u =
          model
              .template feedthrough<PortRole::PERFORMANCE, PortRole::CONTROL>();
      F = linalg::set_block<0, StateEnd, 0, StateEnd>(
          F, T(-1) * (B_u * E + linalg::transpose(E) * linalg::transpose(B_u)));
      F = linalg::set_block<0, StateEnd, DisturbanceEnd, LmiDim>(
          F, T(-1) * linalg::transpose(D_u * E));
      F = linalg::set_block<DisturbanceEnd, LmiDim, 0, StateEnd>(
          F, T(-1) * D_u * E);
      return F;
    };

    auto M_gamma = [](const linalg::Matrix<T, 1, 1>& E) -> FMat {
      FMat F{};
      const auto gamma = E(0, 0);
      const auto I_w = linalg::make_identity<T, ExogenousDim>();
      const auto I_z = linalg::make_identity<T, OutputDim>();
      F = linalg::set_block<StateEnd, DisturbanceEnd, StateEnd, DisturbanceEnd>(
          F, gamma * I_w);
      F = linalg::set_block<DisturbanceEnd, LmiDim, DisturbanceEnd, LmiDim>(
          F, gamma * I_z);
      return F;
    };

    const auto F_q =
        linalg::MatrixPencil<FMat, QDim>{}.template apply_sym_mat<QMat>(M_q);
    const auto F_y =
        linalg::MatrixPencil<FMat, YDim>{}.template apply_full<YMat>(M_y);
    const auto F_gamma =
        linalg::MatrixPencil<FMat, GammaDim>{}
            .template apply_full<linalg::Matrix<T, 1, 1>>(M_gamma);

    Fs F_blks{};
    for (std::size_t i = 0; i < QDim; ++i) F_blks(i) = F_q(i);
    for (std::size_t i = 0; i < YDim; ++i) F_blks(QDim + i) = F_y(i);
    F_blks(QDim + YDim) = F_gamma(0);

    FMat F_0{};
    const auto B_w = model.exogenous_matrix();
    const auto D_w =
        model
            .template feedthrough<PortRole::PERFORMANCE, PortRole::EXOGENOUS>();
    F_0 = linalg::set_block<0, StateEnd, StateEnd, DisturbanceEnd>(F_0,
                                                                   T(-1) * B_w);
    F_0 = linalg::set_block<StateEnd, DisturbanceEnd, 0, StateEnd>(
        F_0, T(-1) * linalg::transpose(B_w));
    F_0 = linalg::set_block<StateEnd, DisturbanceEnd, DisturbanceEnd, LmiDim>(
        F_0, T(-1) * linalg::transpose(D_w));
    F_0 = linalg::set_block<DisturbanceEnd, LmiDim, StateEnd, DisturbanceEnd>(
        F_0, T(-1) * D_w);

    const auto I = linalg::make_identity<T, LmiDim>();
    return convex::LMI(F_blks, F_0 + T(-1) * margin * I, Z_hat);
  }

  /**
   * @brief objective
   *
   * min gamma
   *
   * @param Z_hat (DecVar)
   *
   * @return gamma objective
   */
  template <convex::AtomLike DecVar>
  constexpr auto objective(const DecVar& Z_hat) const {
    linalg::Vector<T, DecVarDim> weights{};
    weights(QDim + YDim) = T(1);
    return convex::make_weighted_sum(weights, Z_hat);
  }

  /**
   * @brief K
   *
   * K = Y Q^{-1}
   *
   * @param Q inverse BRL Lyapunov variable
   * @param Y lifted feedback product Y = K Q
   *
   * @return K
   */
  constexpr auto K(const QMat& Q, const YMat& Y) const {
    return recover_state_feedback_gain(Q, Y);
  }
};

/**
 * @brief make hinf state feedback problem
 *
 * @param model (PlantModel)
 * @param Z_hat (DecVar)
 * @param margin (PSD margin)
 *
 * @return hinf state feedback problem
 */
template <ContinuousPlantModelLike SFModel, convex::AtomLike DecVar>
constexpr auto make_hinf_state_feedback_problem(
    const SFModel& model, const DecVar& Z_hat,
    plant_type_t<SFModel> margin = 1e-6) {
  const auto hinf = HinfStateFeedback<decltype(model)>();
  const auto lmis = hinf(model, Z_hat, margin);
  const auto prob = convex::make_problem(Z_hat).in_set(lmis);
  const auto obj = hinf.objective(Z_hat);
  return prob.minimize(obj);
}

} // namespace control
