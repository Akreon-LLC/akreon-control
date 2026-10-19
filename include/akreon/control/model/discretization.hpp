/*
 * Copyright © 2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "akreon/control/model/state_space.hpp"
#include "akreon/linalg/operator/matrix_inverse.hpp"
#include "akreon/linalg/spectral/matrix_exp.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace control {
/**
 * @brief Convert a real continuous state-space model using zero-order hold.
 *
 * For xdot = A*x + B*u, hold u(t) = u_k over [k*dt, (k+1)*dt).
 * Integrating the dynamics gives x_(k+1) = Ad*x_k + Bd*u_k, where
 *   Ad = exp(A*dt),
 *   Bd = integral_0^dt exp(A*tau)*B d(tau).
 * Compute both through exp(dt * [A B; 0 0]) = [Ad Bd; 0 I], avoiding
 * an inverse of A so integrators and other singular A are supported.
 * For y = C*x + D*u, sampled outputs retain Cd = C and Dd = D;
 * an absent D remains absent.
 * Inputs are held constant over each sample interval; dt is in the same time
 * units as the continuous dynamics. The returned StateSpace does not store dt.
 * Accuracy follows linalg::matrix_exp's generalized-eigenvector algorithm.
 *
 * @throws std::invalid_argument Nonfinite model or nonpositive/nonfinite dt.
 * @throws std::runtime_error Nonfinite, nonreal, or structurally invalid
 * matrix exponential (including its zero-matrix failure sentinel).
 */
template <ContinuousStateSpaceLike SS>
  requires linalg::Real<ss_type_t<SS>>
constexpr auto c2d_zoh(const SS& continuous, ss_type_t<SS> dt) {
  using T = ss_type_t<SS>;
  constexpr std::size_t N = linalg::mat_rows_v<ss_plant_t<SS>>;
  constexpr std::size_t M = linalg::mat_cols_v<ss_control_t<SS>>;
  static_assert(N > 0 && M > 0);
  static_assert(std::same_as<T, linalg::matrix_scalar_t<ss_control_t<SS>>> &&
                std::same_as<T, linalg::matrix_scalar_t<ss_output_t<SS>>>);
  using Plant = linalg::Matrix<T, N, N>;
  using Input = linalg::Matrix<T, N, M>;
  using Discrete =
      StateSpace<TimeDomainType::DISCRETE, Plant, Input, ss_output_t<SS>>;
  if (!std::isfinite(dt) || dt <= T(0))
    throw std::invalid_argument(
        "ZOH sample period must be finite and positive");
  const auto check_finite = [](const auto& matrix) {
    for (std::size_t i = 0; i < linalg::mat_rows_v<decltype(matrix)>; ++i)
      for (std::size_t j = 0; j < linalg::mat_cols_v<decltype(matrix)>; ++j)
        if (!std::isfinite(matrix(i, j)))
          throw std::invalid_argument("ZOH model matrices must be finite");
  };
  check_finite(continuous.A);
  check_finite(continuous.B);
  check_finite(continuous.C);
  if (continuous.D) check_finite(*continuous.D);

  linalg::Matrix<T, N + M, N + M> augmented{};
  for (std::size_t i = 0; i < N; ++i) {
    for (std::size_t j = 0; j < N; ++j)
      augmented(i, j) = dt * continuous.A(i, j);
    for (std::size_t j = 0; j < M; ++j)
      augmented(i, N + j) = dt * continuous.B(i, j);
  }
  // Check for overflow in dt*A or dt*B before entering the eigensolver.
  check_finite(augmented);
  const auto exponential = linalg::matrix_exp(augmented);
  constexpr T tolerance =
      std::max(T(1e-8), T(100) * std::numeric_limits<T>::epsilon());
  for (std::size_t i = 0; i < N + M; ++i) {
    for (std::size_t j = 0; j < N + M; ++j) {
      const auto value = exponential(i, j);
      if (!std::isfinite(value.real()) || !std::isfinite(value.imag()) ||
          linalg::constexpr_abs(value.imag()) >
              tolerance * std::max(T(1), linalg::constexpr_abs(value.real())))
        throw std::runtime_error(
            "ZOH matrix exponential is nonfinite or nonreal");
      if (i >= N) {
        const T expected = i == j ? T(1) : T(0);
        if (linalg::constexpr_abs(value.real() - expected) > tolerance)
          throw std::runtime_error(
              "ZOH matrix exponential has an invalid lower block");
      }
    }
  }
  Plant Ad{};
  Input Bd{};
  for (std::size_t i = 0; i < N; ++i) {
    for (std::size_t j = 0; j < N; ++j) Ad(i, j) = exponential(i, j).real();
    for (std::size_t j = 0; j < M; ++j) Bd(i, j) = exponential(i, N + j).real();
  }
  if (continuous.D) return Discrete(Ad, Bd, continuous.C, *continuous.D);
  return Discrete(Ad, Bd, continuous.C);
}
} // namespace control
