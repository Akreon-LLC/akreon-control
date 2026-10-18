/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../control_traits.hpp"

#pragma once

namespace control {

/**
 * @brief StateSpace
 *
 * @f$\dot{x} = A x(t) + B u(t)@f$
 * @f$y = C x(t) + D u(t)@f$
 */
template <TimeDomainType Type, linalg::SquareMatrix PlantIn,
          linalg::MatrixContainer ControlIn, linalg::MatrixContainer OutputIn>
struct StateSpace {
  /// @brief extract type and alias
  using Plant = linalg::remove_cvref_t<PlantIn>;
  using Control = linalg::remove_cvref_t<ControlIn>;
  using Output = linalg::remove_cvref_t<OutputIn>;

  using T = linalg::matrix_scalar_t<Plant>;
  static constexpr std::size_t PlantRows = linalg::mat_rows_v<Plant>;
  static constexpr std::size_t PlantCols = linalg::mat_cols_v<Plant>;
  static constexpr std::size_t ControlRows = linalg::mat_rows_v<Control>;
  static constexpr std::size_t ControlCols = linalg::mat_cols_v<Control>;
  static constexpr std::size_t OutputRows = linalg::mat_rows_v<Output>;
  static constexpr std::size_t OutputCols = linalg::mat_cols_v<Output>;

  using Feedthrough = linalg::Matrix<T, OutputRows, ControlCols>;

  static_assert(PlantRows == ControlRows,
                "state dim is not aligned between plant and control matrices");

  /// @brief state vector
  using State = linalg::Vector<T, PlantCols>;
  using Input = linalg::Vector<T, ControlCols>;

  /**
   * @brief CTOR
   *
   * @param A (Plant)
   * @param B (Control)
   * @param C (Output)
   */
  constexpr explicit StateSpace(const Plant& A_in, const Control& B_in,
                                const Output& C_in)
      : A(A_in), B(B_in), C(C_in) {}

  /**
   * @brief CTOR
   *
   * @param A (Plant)
   * @param B (Control)
   * @param C (Output)
   * @param D (Feedthrough)
   */
  constexpr explicit StateSpace(const Plant& A_in, const Control& B_in,
                                const Output& C_in, const Feedthrough& D_in)
      : A(A_in), B(B_in), C(C_in), D(D_in) {}

  /**
   * @brief state_update
   *
   * Discrete
   * x_next = A x_k + B u_k
   *
   * Continuous
   * xdot = A x + B u
   *
   * @param x (State)
   * @param u (Input)
   *
   * @return xdot or x_next
   */
  constexpr auto state_update(const State& x, const Input& u) const {
    return A * x + B * u;
  }

  /**
   * @brief output_map
   *
   * similar as plant_dynamics
   * y = C x + D u
   *
   * @param x (State)
   * @param u (Input)
   *
   * @return y
   */
  constexpr auto output_map(const State& x, const Input& u) const {
    return D.has_value() ? C * x + (*D) * u : C * x;
  }

  /** @brief member var */
  Plant A;                      ///< Plant matrix
  Control B;                    ///< Control matrix
  Output C;                     ///< Output matrix
  std::optional<Feedthrough> D; ///< Feedthrough matrix
};

} // namespace control
