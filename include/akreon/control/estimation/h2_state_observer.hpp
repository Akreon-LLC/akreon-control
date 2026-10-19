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
 * @brief H2 State Observer
 *
 * LDI governed by H2 norm
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
 *
 * The observability gramian (Q) requirement
 *   A_cl^T Q + Q A_cl + C_e^T C_e < 0
 *
 * The H2 norm squared (total impulse response energy)
 *   ||T_{wz_e}||_2^2 = Tr(B_cl^T Q B_cl)
 *
 * Dimensions:
 *   n   = state dimension
 *   m_w = disturbance input dimension
 *   p_y = measurement output dimension
 *   p_e = performance output dimension on the estimation error
 *
 *   A    : n x n
 *   B_w  : n x m_w
 *   C_y  : p_y x n
 *   D_yw : p_y x m_w
 *   C_e  : p_e x n
 *
 * Decision blocks:
 *   Q : n x n
 *   W   : n x p_y
 *   L   : n x p_y
 *   Z   : m_w x m_w
 *   DecVar = [svec(Q); vec(W); svec(Z)]
 *
 * We have a bilinear term in Q L, so let W = Q L
 * where L = Q^{-1} W
 *
 * H2 objective can be lifted via epigraph with Z
 *   Z >= (Q B_w - W D_yw)^T Q^{-1} (Q B_w - W D_yw)
 *
 * Convex Problem is now
 *   min  Tr(Z)
 *   s.t. A^T Q + Q A - C_y^T W^T - W C_y + C_e^T C_e < 0
 *        [ Z                         (Q B_w - W D_yw)^T ] >= 0
 *        [ Q B_w - W D_yw          Q                  ]
 *
 * Shape checks:
 *   A^T Q, Q A, C_y^T W^T, W C_y, C_e^T C_e : n x n
 *   Q B_w - W D_yw : n x m_w
 *   Epigraph LMI   : (m_w + n) x (m_w + n)
 */
template <ContinuousPlantModelLike ObsModel>
struct H2StateObserver {
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
  using ZMat = linalg::Matrix<T, ExogenousDim, ExogenousDim>;

  static constexpr std::size_t QDim = StateDim * (StateDim + 1) / 2;
  static constexpr std::size_t WDim = StateDim * MeasurementDim;
  static constexpr std::size_t ZDim = ExogenousDim * (ExogenousDim + 1) / 2;
  static constexpr std::size_t DecVarDim = QDim + WDim + ZDim;

  using DecVarSpace = linalg::RowVector<T, DecVarDim>;

  /**
   * @brief operator overload to form H2 state observer
   *
   * @param model (PlantModel)
   * @param Z_hat (DecVar)
   *
   * @return H2 LMIs
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
   * @param Z_hat ([svec(Q); vec(W); svec(Z)])
   *
   * @return H2 LMI
   */
  template <convex::AtomLike DecVar>
  constexpr auto apply(const ObsModel& model, const DecVar& Z_hat,
                       T margin = 1e-6) const {
    static_assert(convex::is_var_type_v<DecVar>, "Input atom is not a var");
    using DecVarDom = convex::meta_dom_t<convex::atom_meta_t<DecVar>>;
    static_assert(convex::dom_vartype_v<DecVarDom> == convex::VarType::FREE,
                  "Merged decision var must be free");
    static_assert(std::same_as<DecVarSpace, convex::dom_space_t<DecVarDom>>,
                  "Expected merged decision size [svec(Q); vec(W); svec(Z)]");

    const auto stability_lmi = stability_constraint(model, Z_hat);
    const auto z_lmi = z_constraint(model, Z_hat, margin);
    return convex::Intersection(z_lmi, stability_lmi);
  }

  /**
   * @brief stability_constraint
   *
   * Stability constraint on LMI:
   *   A^T Q + Q A - C_y^T W^T - W C_y + C_e^T C_e < 0
   *
   * @param model (PlantModel)
   * @param Z_hat (Dec Var)
   *
   * @return stability constraint as LMI
   */
  template <convex::AtomLike DecVar>
  constexpr auto stability_constraint(const ObsModel& model,
                                      const DecVar& Z_hat) const {
    using FMat = QMat;
    using Fs = linalg::Vector<FMat, DecVarDim>;

    auto M_q = [&](const QMat& E) -> FMat {
      const auto& A = model.dynamics_matrix();
      return T(-1) * (linalg::transpose(A) * E + E * A);
    };

    auto M_w = [&](const WMat& E) -> FMat {
      const auto& C = model.measurement_matrix();
      return linalg::transpose(C) * linalg::transpose(E) + E * C;
    };

    const auto C_e = model.performance_matrix();
    const auto F_0 = T(-1) * (linalg::transpose(C_e) * C_e);

    const auto F_q_local =
        linalg::MatrixPencil<FMat, QDim>{}.template apply_sym_mat<QMat>(M_q);
    const auto F_w_local =
        linalg::MatrixPencil<FMat, WDim>{}.template apply_full<WMat>(M_w);

    Fs F_blks{};
    for (std::size_t i = 0; i < QDim; ++i) F_blks(i) = F_q_local(i);
    for (std::size_t i = 0; i < WDim; ++i) F_blks(QDim + i) = F_w_local(i);

    return convex::LMI(F_blks, F_0, Z_hat);
  }

  /**
   * @brief z_constraint
   *
   * PSD constraint on Z (epigraph)
   *   [ Z                 (Q B_w - W D_yw)^T ] >= 0
   *   [ Q B_w - W D_yw  Q                  ]
   *
   * FMat Dim (m+n x m+n)
   *   [ mxm mxn ]
   *   [ nxm nxn ]
   *
   * @param model (PlantModel)
   * @param Z_hat (DecVar)
   * @param margin for PD
   *
   * @return Z constraint as LMI
   */
  template <convex::AtomLike DecVar>
  constexpr auto z_constraint(const ObsModel& model, const DecVar& Z_hat,
                              T margin = 1e-6) const {
    constexpr std::size_t LmiDim = ExogenousDim + StateDim;
    using FMat = linalg::Matrix<T, LmiDim, LmiDim>;
    using Fs = linalg::Vector<FMat, DecVarDim>;
    auto M_z = [](const ZMat& E) -> FMat {
      FMat F{};
      return linalg::set_block<0, ExogenousDim, 0, ExogenousDim>(F, E);
    };
    auto M_q = [&](const QMat& E) -> FMat {
      FMat F{};
      const auto B_w = model.exogenous_matrix();
      F = linalg::set_block<ExogenousDim, LmiDim, ExogenousDim, LmiDim>(F, E);
      F = linalg::set_block<0, ExogenousDim, ExogenousDim, LmiDim>(
          F, linalg::transpose(B_w) * E);
      F = linalg::set_block<ExogenousDim, LmiDim, 0, ExogenousDim>(F, E * B_w);
      return F;
    };
    auto M_w = [&](const WMat& E) -> FMat {
      FMat F{};
      const auto D_w = model.template feedthrough<PortRole::MEASUREMENT,
                                                  PortRole::EXOGENOUS>();
      F = linalg::set_block<0, ExogenousDim, ExogenousDim, LmiDim>(
          F, T(-1) * linalg::transpose(D_w) * linalg::transpose(E));
      F = linalg::set_block<ExogenousDim, LmiDim, 0, ExogenousDim>(
          F, T(-1) * E * D_w);
      return F;
    };

    const auto F_z_local =
        linalg::MatrixPencil<FMat, ZDim>{}.template apply_sym_mat<ZMat>(M_z);
    const auto F_q_local =
        linalg::MatrixPencil<FMat, QDim>{}.template apply_sym_mat<QMat>(M_q);
    const auto F_w_local =
        linalg::MatrixPencil<FMat, WDim>{}.template apply_full<WMat>(M_w);

    Fs F_blks{};
    for (std::size_t i = 0; i < QDim; ++i) F_blks(i) = F_q_local(i);
    for (std::size_t i = 0; i < WDim; ++i) F_blks(QDim + i) = F_w_local(i);
    for (std::size_t i = 0; i < ZDim; ++i)
      F_blks(QDim + WDim + i) = F_z_local(i);

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
    return convex::make_trace_block<QDim + WDim, linalg::SymmetricMatrix<ZMat>>(
        Z_hat);
  }

  /**
   * @brief L
   *
   * L = Q^{-1} W
   *
   * @param Q observability gramian
   * @param W lifted observer product (W = Q L)
   *
   * @return L
   */
  constexpr auto L(const QMat& Q, const WMat& W) const {
    return recover_state_observer_gain(Q, W);
  }
};

/**
 * @brief make h2 state observer problem
 *
 * @param model (PlantModel)
 * @param Z (DecVar)
 * @param margin (PSD Margin)
 *
 * @return h2 state observer problem
 */
template <ContinuousPlantModelLike ObsModel, convex::AtomLike DecVar>
constexpr auto make_h2_state_observer_problem(
    const ObsModel& model, const DecVar& Z_hat,
    plant_type_t<ObsModel> margin = 1e-6) {
  const auto h2 = H2StateObserver<decltype(model)>();
  const auto lmis = h2(model, Z_hat, margin);
  const auto prob = convex::make_problem(Z_hat).in_set(lmis);
  const auto obj = h2.objective(Z_hat);
  return prob.minimize(obj);
}

} // namespace control
