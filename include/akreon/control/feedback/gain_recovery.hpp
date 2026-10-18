/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "akreon/linalg/operator/matrix_inverse.hpp"

namespace control {

/**
 * @brief recover state-feedback gain from a lifted gain variable
 *
 * For standard state feedback and H2 state feedback, the lifted variable has
 * the same form:
 *
 *   Y = K P
 *   K = Y P^{-1}
 *
 * @param P symmetric positive-definite controllability Gramian block
 * @param Y lifted feedback product
 *
 * @return tuple(K, numerical event)
 */
template <linalg::SquareMatrix PMat, linalg::MatrixContainer YMat>
constexpr auto recover_state_feedback_gain(const PMat& P, const YMat& Y) {
  using T = linalg::matrix_scalar_t<PMat>;
  static constexpr std::size_t InputDim = linalg::mat_rows_v<YMat>;
  static constexpr std::size_t StateDim = linalg::mat_cols_v<YMat>;
  static_assert(linalg::mat_rows_v<PMat> == StateDim,
                "P row dimension must match Y column dimension");
  static_assert(linalg::mat_cols_v<PMat> == StateDim,
                "P column dimension must match Y column dimension");

  using KMat = linalg::Matrix<T, InputDim, StateDim>;
  const auto [P_inv, event] = linalg::SymmetricMatrixInverse<PMat>()(P);
  if (event != linalg::NumericalEvent::NONE) return std::tuple(KMat{}, event);
  const auto K = Y * P_inv;
  return std::tuple(K, linalg::NumericalEvent::NONE);
}

/**
 * @brief recover state-observer gain from a lifted gain variable
 *
 * For H2 state observers, the lifted variable has the form:
 *
 *   W = Q L
 *   L = Q^{-1} W
 *
 * @param Q symmetric positive-definite observability Gramian block
 * @param W lifted observer product
 *
 * @return tuple(L, numerical event)
 */
template <linalg::SquareMatrix QMat, linalg::MatrixContainer WMat>
constexpr auto recover_state_observer_gain(const QMat& Q, const WMat& W) {
  using T = linalg::matrix_scalar_t<QMat>;
  static constexpr std::size_t StateDim = linalg::mat_rows_v<WMat>;
  static constexpr std::size_t MeasurementDim = linalg::mat_cols_v<WMat>;
  static_assert(linalg::mat_rows_v<QMat> == StateDim,
                "Q row dimension must match W row dimension");
  static_assert(linalg::mat_cols_v<QMat> == StateDim,
                "Q column dimension must match W row dimension");

  using LMat = linalg::Matrix<T, StateDim, MeasurementDim>;
  const auto [Q_inv, event] = linalg::SymmetricMatrixInverse<QMat>()(Q);
  if (event != linalg::NumericalEvent::NONE) return std::tuple(LMat{}, event);
  const auto L = Q_inv * W;
  return std::tuple(L, linalg::NumericalEvent::NONE);
}

} // namespace control
