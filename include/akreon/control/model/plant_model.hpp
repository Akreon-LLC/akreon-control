/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <utility>

#include "akreon/control/control_traits.hpp"
#include "akreon/linalg/linalg_util.hpp"

namespace control {

/**
 * @brief State-space model with compile-time input and output port roles.
 *
 * The underlying equations are xdot = A*x + B*v (continuous) or
 * x[k+1] = A*x[k] + B*v[k] (discrete), with o = C*x + D*v.
 * Input ports identify exogenous channels w and control channels u in v;
 * output ports identify performance channels z and measurements y in o.
 * The full matrices retain their original channel ordering.
 *
 * Each Port<Role, Offset, Size> selects a contiguous, zero-based range:
 * inputs select B/D columns and outputs select C/D rows. Ranges must be
 * nonempty, in bounds, and nonoverlapping within each layout; these checks
 * occur at compile time. Layouts need not cover every channel. Callers choose
 * appropriate input/output roles and satisfy each synthesis formulation's
 * additional assumptions; this wrapper does not check stabilizability,
 * detectability, or physical units.
 *
 * Accessors concatenate matching ports in layout declaration order, not
 * ascending matrix offset. A missing role produces a zero-width input matrix
 * or zero-height output matrix. All returned matrices are values, not views.
 * Layouts are types and require no runtime storage; state_space owns a copy
 * of the supplied model, including its time domain and optional D matrix.
 *
 * For a model with inputs [w, u] and outputs [z, y], one channel each:
 * @code
 * using Inputs = PortLayout<Port<PortRole::EXOGENOUS, 0, 1>,
 *                           Port<PortRole::CONTROL, 1, 1>>;
 * using Outputs = PortLayout<Port<PortRole::PERFORMANCE, 0, 1>,
 *                            Port<PortRole::MEASUREMENT, 1, 1>>;
 * // SS has two input columns and two output rows.
 * const auto plant = PlantModel<SS, Inputs, Outputs>(ss);
 * const auto B_u = plant.control_matrix();
 * const auto D_yu = plant.feedthrough<PortRole::MEASUREMENT,
 *                                    PortRole::CONTROL>();
 * @endcode
 *
 * @tparam SS Continuous or discrete StateSpaceLike model.
 * @tparam InputPorts PortLayout describing the full input axis.
 * @tparam OutputPorts PortLayout describing the full output axis.
 */
template <StateSpaceLike SS, typename InputPorts, typename OutputPorts>
struct PlantModel {
  using StateSpace = linalg::remove_cvref_t<SS>;
  using InputPortsT = linalg::remove_cvref_t<InputPorts>;
  using OutputPortsT = linalg::remove_cvref_t<OutputPorts>;
  using Model = PlantModel<StateSpace, InputPortsT, OutputPortsT>;
  using T = plant_type_t<Model>;

  static constexpr std::size_t StateDim = plant_traits<Model>::StateDim;
  static constexpr std::size_t InputDim = plant_traits<Model>::InputDim;
  static constexpr std::size_t OutputDim = plant_traits<Model>::OutputDim;

  static_assert(InputPortsT::template validates<InputDim>(),
                "Input port layout is invalid for the state-space input");
  static_assert(OutputPortsT::template validates<OutputDim>(),
                "Output port layout is invalid for the state-space output");

  /** @brief Copy a state-space model using the template-specified layouts.
   * @param ss Full model whose matrix ordering matches the port offsets.
   */
  constexpr explicit PlantModel(const StateSpace& ss) : state_space(ss) {}

  /** @brief Copy a model with layout arguments supplied for type deduction.
   * @param ss Full state-space model.
   * The layout arguments carry types only; no layout objects are stored.
   */
  constexpr PlantModel(const StateSpace& ss, const InputPortsT&,
                       const OutputPortsT&)
      : state_space(ss) {}

  /// @brief Return A (StateDim x StateDim), without changing the time domain.
  constexpr auto dynamics_matrix() const { return state_space.A; }

  /// @brief Return B_w (StateDim x ExogenousDim), for
  /// disturbance/reference/noise inputs.
  constexpr auto exogenous_matrix() const {
    return input_matrix<PortRole::EXOGENOUS>();
  }

  /// @brief Return B_u (StateDim x ControlDim), for actuator inputs.
  constexpr auto control_matrix() const {
    return input_matrix<PortRole::CONTROL>();
  }

  /// @brief Return C_z (PerformanceDim x StateDim), for regulated outputs.
  constexpr auto performance_matrix() const {
    return output_matrix<PortRole::PERFORMANCE>();
  }

  /// @brief Return C_y (MeasurementDim x StateDim), for measured outputs.
  constexpr auto measurement_matrix() const {
    return output_matrix<PortRole::MEASUREMENT>();
  }

  /**
   * @brief Extract D blocks from the selected input role to output role.
   * @tparam OutRole Output role selecting rows in C/D.
   * @tparam InRole Input role selecting columns in B/D.
   * @return Matrix with total_size<OutRole>() rows and total_size<InRole>()
   * columns. Matching row/column blocks follow their respective layout order.
   * If the model has no D, return zeros of the same shape. A missing role
   * yields the corresponding zero dimension.
   */
  template <PortRole OutRole, PortRole InRole>
  constexpr auto feedthrough() const {
    constexpr std::size_t Rows = OutputPortsT::template total_size<OutRole>();
    constexpr std::size_t Cols = InputPortsT::template total_size<InRole>();
    linalg::Matrix<T, Rows, Cols> result{};
    if (state_space.D.has_value()) {
      copy_feedthrough_ports<OutRole, InRole, 0, 0>(result);
    }
    return result;
  }

  SS state_space; ///< Owned full model; accessors reflect its current matrices.

private:
  /// @brief aggregate all input ports of a role from the full B matrix
  template <PortRole Role>
  constexpr auto input_matrix() const {
    constexpr std::size_t Cols = InputPortsT::template total_size<Role>();
    linalg::Matrix<T, StateDim, Cols> result{};
    copy_input_ports<Role, 0, 0>(result);
    return result;
  }

  /// @brief aggregate all output ports of a role from the full C matrix
  template <PortRole Role>
  constexpr auto output_matrix() const {
    constexpr std::size_t Rows = OutputPortsT::template total_size<Role>();
    linalg::Matrix<T, Rows, StateDim> result{};
    copy_output_ports<Role, 0, 0>(result);
    return result;
  }

  /// @brief copy selected B columns into an aggregated input matrix
  template <PortRole Role, std::size_t Index, std::size_t DestOffset,
            typename Mat>
  constexpr void copy_input_ports(Mat& result) const {
    constexpr std::size_t Count = InputPortsT::template count<Role>();
    if constexpr (Index < Count) {
      constexpr std::size_t Offset =
          InputPortsT::template offset<Role, Index>();
      constexpr std::size_t Size = InputPortsT::template size<Role, Index>();
      const auto block =
          linalg::get_block<0, StateDim, Offset, Offset + Size>(state_space.B);
      result = linalg::set_block<0, StateDim, DestOffset, DestOffset + Size>(
          result, block);
      copy_input_ports<Role, Index + 1, DestOffset + Size>(result);
    }
  }

  /// @brief copy selected C rows into an aggregated output matrix
  template <PortRole Role, std::size_t Index, std::size_t DestOffset,
            typename Mat>
  constexpr void copy_output_ports(Mat& result) const {
    constexpr std::size_t Count = OutputPortsT::template count<Role>();
    if constexpr (Index < Count) {
      constexpr std::size_t Offset =
          OutputPortsT::template offset<Role, Index>();
      constexpr std::size_t Size = OutputPortsT::template size<Role, Index>();
      const auto block =
          linalg::get_block<Offset, Offset + Size, 0, StateDim>(state_space.C);
      result = linalg::set_block<DestOffset, DestOffset + Size, 0, StateDim>(
          result, block);
      copy_output_ports<Role, Index + 1, DestOffset + Size>(result);
    }
  }

  /// @brief copy selected D row blocks for each matching output port
  template <PortRole OutRole, PortRole InRole, std::size_t OutIndex,
            std::size_t OutDestOffset, typename Mat>
  constexpr void copy_feedthrough_ports(Mat& result) const {
    constexpr std::size_t OutCount = OutputPortsT::template count<OutRole>();
    if constexpr (OutIndex < OutCount) {
      constexpr std::size_t OutOffset =
          OutputPortsT::template offset<OutRole, OutIndex>();
      constexpr std::size_t OutSize =
          OutputPortsT::template size<OutRole, OutIndex>();
      copy_feedthrough_input_ports<OutRole, InRole, OutIndex, OutOffset,
                                   OutSize, 0, 0>(result);
      copy_feedthrough_ports<OutRole, InRole, OutIndex + 1,
                             OutDestOffset + OutSize>(result);
    }
  }

  /// @brief copy selected D column blocks for one selected output port
  template <PortRole OutRole, PortRole InRole, std::size_t OutIndex,
            std::size_t OutOffset, std::size_t OutSize, std::size_t InIndex,
            std::size_t InDestOffset, typename Mat>
  constexpr void copy_feedthrough_input_ports(Mat& result) const {
    constexpr std::size_t InCount = InputPortsT::template count<InRole>();
    if constexpr (InIndex < InCount) {
      constexpr std::size_t InOffset =
          InputPortsT::template offset<InRole, InIndex>();
      constexpr std::size_t InSize =
          InputPortsT::template size<InRole, InIndex>();
      constexpr std::size_t OutDestOffset =
          feedthrough_output_destination<OutRole, OutIndex>();
      const auto block =
          linalg::get_block<OutOffset, OutOffset + OutSize, InOffset,
                            InOffset + InSize>(*state_space.D);
      result =
          linalg::set_block<OutDestOffset, OutDestOffset + OutSize,
                            InDestOffset, InDestOffset + InSize>(result, block);
      copy_feedthrough_input_ports<OutRole, InRole, OutIndex, OutOffset,
                                   OutSize, InIndex + 1, InDestOffset + InSize>(
          result);
    }
  }

  /// @brief aggregated row offset for an output port selected from D
  template <PortRole Role, std::size_t Index>
  static consteval std::size_t feedthrough_output_destination() {
    std::size_t dest = 0;
    [&]<std::size_t... Is>(std::index_sequence<Is...>) consteval {
      ((Is < Index ? dest += OutputPortsT::template size<Role, Is>() : 0), ...);
    }(std::make_index_sequence<OutputPortsT::template count<Role>()>{});
    return dest;
  }
};

} // namespace control
