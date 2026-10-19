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
 * @brief Hinf State Observer
 *
 * Given the LTI System with additive disturbance
 *   xdot = A x + B_u u + B_w w
 *   y = C_y x + D_yw w
 *   z_e = C_e e
 *
 * Observer design
 *   hat{xdot} = A hat{x} + B_u u + L(y - C_y hat{x})
 *   e = x - hat{x}
 *   edot = (A - L C_y) e + (B_w - L D_yw) w
 *
 * A_cl = A - L C_y
 * B_cl = B_w - L D_yw
 * C_cl = C_e
 *
 * Certifies ||T_{wz_e}||_infinity < gamma.
 *
 * General BRL form with observability Lyapunov variable Q > 0:
 *   [ A_cl^T Q + Q A_cl  Q B_cl   C_cl^T   ] < 0
 *   [ B_cl^T Q           -gamma I D_ew^T   ]
 *   [ C_cl               D_ew     -gamma I ]
 *
 * Hinf observer synthesis decision variables:
 *   Q     : n x n, observability Lyapunov variable
 *   W     : n x p_y, lifted observer product W = Q L
 *   L     : n x p_y
 *   gamma : scalar induced L2-gain bound
 *   DecVar = [svec(Q); vec(W); gamma]
 *
 * Objective:
 *   min gamma
 *
 * We have a bilinear term in Q L, so let W = Q L where L = Q^-1 W.
 *
 * Convex problem:
 *   min  gamma
 *   s.t. Q > 0
 *        [ A^T Q + Q A - C_y^T W^T - W C_y  Q B_w - W D_yw  C_e^T    ] < 0
 *        [ (Q B_w - W D_yw)^T               -gamma I        D_ew^T   ]
 *        [ C_e                              D_ew            -gamma I ]
 *
 * Full decision space:
 *   QDim       = n(n + 1) / 2
 *   WDim       = n p_y
 *   GammaDim   = 1
 *   DecVarDim  = QDim + WDim + GammaDim
 *   DecVar     in R^DecVarDim
 *
 * Shape checks:
 *   A^T Q, Q A, C_y^T W^T, W C_y : n x n
 *   Q B_w - W D_yw               : n x m_w
 *   C_e                           : p_z x n
 *   gamma I_w                     : m_w x m_w
 *   gamma I_z                     : p_z x p_z
 *   BRL LMI                       : (n + m_w + p_z) x (n + m_w + p_z)
 */
template <ContinuousPlantModelLike ObsModel>
struct HinfStateObserver {
  /// @brief extracting alias and types
  using Model = linalg::remove_cvref_t<ObsModel>;

  using T = plant_type_t<Model>;

  static constexpr std::size_t StateDim = plant_traits<Model>::StateDim;
  static constexpr std::size_t ExogenousDim = plant_traits<Model>::ExogenousDim;
  static constexpr std::size_t MeasurementDim =
      plant_traits<Model>::MeasurementDim;
  static constexpr std::size_t PerformanceDim =
      plant_traits<Model>::PerformanceDim;

  using QMat = linalg::Matrix<T, StateDim, StateDim>;
  using WMat = linalg::Matrix<T, StateDim, MeasurementDim>;
  using LMat = linalg::Matrix<T, StateDim, MeasurementDim>;

  static constexpr std::size_t QDim = StateDim * (StateDim + 1) / 2;
  static constexpr std::size_t WDim = StateDim * MeasurementDim;
  static constexpr std::size_t GammaDim = 1;
  static constexpr std::size_t DecVarDim = QDim + WDim + GammaDim;

  using DecVarSpace = linalg::RowVector<T, DecVarDim>;

  /**
   * @brief operator overload to form Hinf state observer
   *
   * @param model (PlantModel)
   * @param Z_hat (DecVar)
   * @param margin (PDness)
   *
   * @return Hinf LMIs
   */
  template <convex::AtomLike DecVar>
  constexpr auto operator()(const ObsModel& model, const DecVar& Z_hat,
                            T margin = 1e-6) const {
    return apply(model, Z_hat, margin);
  }

  /**
   * @brief apply
   *
   * @param model (PlantModel)
   * @param Z_hat ([svec(Q); vec(W); gamma])
   * @param margin (PDness)
   *
   * @return Hinf synthesis constraints
   */
  template <convex::AtomLike DecVar>
  constexpr auto apply(const ObsModel& model, const DecVar& Z_hat,
                       T margin = 1e-6) const {
    static_assert(convex::is_var_type_v<DecVar>, "Input atom is not a var");
    using DecVarDom = convex::meta_dom_t<convex::atom_meta_t<DecVar>>;
    static_assert(convex::dom_vartype_v<DecVarDom> == convex::VarType::FREE,
                  "Merged decision var must be free");
    static_assert(std::same_as<DecVarSpace, convex::dom_space_t<DecVarDom>>,
                  "Expected merged decision size [svec(Q); vec(W); gamma]");

    const auto q_lmi = q_constraint(Z_hat, margin);
    const auto brl_lmi = brl_constraint(model, Z_hat, margin);
    return convex::Intersection(q_lmi, brl_lmi);
  }

  /**
   * @brief q_constraint
   *
   * PSD constraint on the observability Lyapunov block:
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
   * PSD constraint on the negated observer BRL block:
   *   -BRL(Q, W, gamma) >= margin I
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
  constexpr auto brl_constraint(const ObsModel& model, const DecVar& Z_hat,
                                T margin = 1e-6) const {
    constexpr std::size_t StateEnd = StateDim;
    constexpr std::size_t DisturbanceEnd = StateDim + ExogenousDim;
    constexpr std::size_t LmiDim = StateDim + ExogenousDim + PerformanceDim;

    using FMat = linalg::Matrix<T, LmiDim, LmiDim>;
    using Fs = linalg::Vector<FMat, DecVarDim>;

    auto M_q = [&](const QMat& E) -> FMat {
      FMat F{};
      const auto A = model.dynamics_matrix();
      const auto B_w = model.exogenous_matrix();
      F = linalg::set_block<0, StateEnd, 0, StateEnd>(
          F, T(-1) * (linalg::transpose(A) * E + E * A));
      F = linalg::set_block<0, StateEnd, StateEnd, DisturbanceEnd>(
          F, T(-1) * E * B_w);
      F = linalg::set_block<StateEnd, DisturbanceEnd, 0, StateEnd>(
          F, T(-1) * linalg::transpose(B_w) * E);
      return F;
    };

    auto M_w = [&](const WMat& E) -> FMat {
      FMat F{};
      const auto C_y = model.measurement_matrix();
      const auto D_yw = model.template feedthrough<PortRole::MEASUREMENT,
                                                   PortRole::EXOGENOUS>();
      F = linalg::set_block<0, StateEnd, 0, StateEnd>(
          F, linalg::transpose(C_y) * linalg::transpose(E) + E * C_y);
      F = linalg::set_block<0, StateEnd, StateEnd, DisturbanceEnd>(F, E * D_yw);
      F = linalg::set_block<StateEnd, DisturbanceEnd, 0, StateEnd>(
          F, linalg::transpose(D_yw) * linalg::transpose(E));
      return F;
    };

    auto M_gamma = [](const linalg::Matrix<T, 1, 1>& E) -> FMat {
      FMat F{};
      const auto gamma = E(0, 0);
      const auto I_w = linalg::make_identity<T, ExogenousDim>();
      const auto I_z = linalg::make_identity<T, PerformanceDim>();
      F = linalg::set_block<StateEnd, DisturbanceEnd, StateEnd, DisturbanceEnd>(
          F, gamma * I_w);
      F = linalg::set_block<DisturbanceEnd, LmiDim, DisturbanceEnd, LmiDim>(
          F, gamma * I_z);
      return F;
    };

    const auto F_q =
        linalg::MatrixPencil<FMat, QDim>{}.template apply_sym_mat<QMat>(M_q);
    const auto F_w =
        linalg::MatrixPencil<FMat, WDim>{}.template apply_full<WMat>(M_w);
    const auto F_gamma =
        linalg::MatrixPencil<FMat, GammaDim>{}
            .template apply_full<linalg::Matrix<T, 1, 1>>(M_gamma);

    Fs F_blks{};
    for (std::size_t i = 0; i < QDim; ++i) F_blks(i) = F_q(i);
    for (std::size_t i = 0; i < WDim; ++i) F_blks(QDim + i) = F_w(i);
    F_blks(QDim + WDim) = F_gamma(0);

    FMat F_0{};
    const auto C_e = model.performance_matrix();
    const auto D_ew =
        model
            .template feedthrough<PortRole::PERFORMANCE, PortRole::EXOGENOUS>();
    F_0 = linalg::set_block<0, StateEnd, DisturbanceEnd, LmiDim>(
        F_0, T(-1) * linalg::transpose(C_e));
    F_0 = linalg::set_block<DisturbanceEnd, LmiDim, 0, StateEnd>(F_0,
                                                                 T(-1) * C_e);
    F_0 = linalg::set_block<StateEnd, DisturbanceEnd, DisturbanceEnd, LmiDim>(
        F_0, T(-1) * linalg::transpose(D_ew));
    F_0 = linalg::set_block<DisturbanceEnd, LmiDim, StateEnd, DisturbanceEnd>(
        F_0, T(-1) * D_ew);

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
    weights(QDim + WDim) = T(1);
    return convex::make_weighted_sum(weights, Z_hat);
  }

  /**
   * @brief L
   *
   * L = Q^{-1} W
   *
   * @param Q observability Lyapunov variable
   * @param W lifted observer product W = Q L
   *
   * @return L
   */
  constexpr auto L(const QMat& Q, const WMat& W) const {
    return recover_state_observer_gain(Q, W);
  }
};

/**
 * @brief make hinf state observer problem
 *
 * @param model (PlantModel)
 * @param Z_hat (DecVar)
 * @param margin (PSD margin)
 *
 * @return hinf state observer problem
 */
template <ContinuousPlantModelLike ObsModel, convex::AtomLike DecVar>
constexpr auto make_hinf_state_observer_problem(
    const ObsModel& model, const DecVar& Z_hat,
    plant_type_t<ObsModel> margin = 1e-6) {
  const auto hinf = HinfStateObserver<decltype(model)>();
  const auto lmis = hinf(model, Z_hat, margin);
  const auto prob = convex::make_problem(Z_hat).in_set(lmis);
  const auto obj = hinf.objective(Z_hat);
  return prob.minimize(obj);
}

} // namespace control
