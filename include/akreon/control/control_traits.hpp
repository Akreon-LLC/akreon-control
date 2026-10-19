/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <concepts>
#include <cstdint>
#include <tuple>

#include "akreon/linalg/space/space_concepts.hpp"

namespace control {

/// @brief Plant port role
enum struct PortRole : uint8_t {
  EXOGENOUS,   ///< w: disturbance, reference, noise, or other external input
  CONTROL,     ///< u: controller input to the plant
  PERFORMANCE, ///< z: regulated/performance output
  MEASUREMENT  ///< y: measured output available for feedback/estimation
};

/// @brief Port specification
template <PortRole RoleT, std::size_t OffsetT, std::size_t SizeT>
struct Port {
  static constexpr PortRole Role = RoleT;
  static constexpr std::size_t Offset = OffsetT;
  static constexpr std::size_t Size = SizeT;
};

template <typename Channel>
concept PortLike = requires {
  { Channel::Role } -> std::convertible_to<PortRole>;
  { Channel::Offset } -> std::convertible_to<std::size_t>;
  { Channel::Size } -> std::convertible_to<std::size_t>;
};

/// @brief Port layout over either the state-space input or output axis
template <PortLike... Channels>
struct PortLayout {
  using ChannelsTuple = std::tuple<Channels...>;

  /// @brief count channels matching a port role
  template <PortRole Role>
  static consteval std::size_t count() {
    return (std::size_t{0} + ... +
            (Channels::Role == Role ? std::size_t{1} : std::size_t{0}));
  }

  /// @brief total dimension for channels matching a port role
  template <PortRole Role>
  static consteval std::size_t total_size() {
    return (std::size_t{0} + ... +
            (Channels::Role == Role ? Channels::Size : 0));
  }

  /// @brief offset of the Index-th channel matching a port role
  template <PortRole Role, std::size_t Index>
  static consteval std::size_t offset() {
    std::size_t current = 0;
    std::size_t result = 0;
    // Walk matching channels left-to-right. current is incremented only for
    // matching channels, so current == Index is true for exactly one match.
    ((Channels::Role == Role
          ? ((current == Index ? (result = Channels::Offset, 0) : 0), ++current,
             0)
          : 0),
     ...);
    static_assert(Index < count<Role>(), "Requested port is missing");
    return result;
  }

  /// @brief size of the Index-th channel matching a port role
  template <PortRole Role, std::size_t Index>
  static consteval std::size_t size() {
    std::size_t current = 0;
    std::size_t result = 0;
    // Same matching-channel walk as offset(), but records channel size.
    ((Channels::Role == Role
          ? ((current == Index ? (result = Channels::Size, 0) : 0), ++current,
             0)
          : 0),
     ...);
    static_assert(Index < count<Role>(), "Requested port is missing");
    return result;
  }

  /// @brief validate port bounds and overlap against a full axis dimension
  template <std::size_t FullDim>
  static consteval bool validates() {
    // First fold checks every channel has positive size and in-bounds extent.
    bool ok =
        ((Channels::Size > 0 && Channels::Offset + Channels::Size <= FullDim) &&
         ...);
    bool occupied[FullDim == 0 ? 1 : FullDim]{};
    auto mark = [&]<typename Channel>() consteval {
      if constexpr (Channel::Offset + Channel::Size > FullDim) { return; }
      // Mark each occupied index; seeing an index twice means overlap.
      for (std::size_t i = Channel::Offset; i < Channel::Offset + Channel::Size;
           ++i) {
        if (occupied[i]) ok = false;
        occupied[i] = true;
      }
    };
    // Invoke mark once per channel type in declaration order.
    (mark.template operator()<Channels>(), ...);
    return ok;
  }
};

/// @brief time domain type
enum struct TimeDomainType : uint8_t { CONTINUOUS = 0, DISCRETE };

/// @brief forward decl of state space
template <TimeDomainType Type, linalg::SquareMatrix PlantIn,
          linalg::MatrixContainer ControlIn, linalg::MatrixContainer OutputIn>
struct StateSpace;

/// @brief base type traits for state space
template <class StateSpace>
struct ss_traits;

/// @brief explicit type traits for state space
template <TimeDomainType Type, linalg::SquareMatrix PlantIn,
          linalg::MatrixContainer ControlIn, linalg::MatrixContainer OutputIn>
struct ss_traits<StateSpace<Type, PlantIn, ControlIn, OutputIn>> {
  using Plant = linalg::remove_cvref_t<PlantIn>;
  using Control = linalg::remove_cvref_t<ControlIn>;
  using Output = linalg::remove_cvref_t<OutputIn>;

  using T = linalg::matrix_scalar_t<Plant>;
  using State = linalg::Vector<T, linalg::mat_cols_v<Plant>>;
  using Input = linalg::Vector<T, linalg::mat_cols_v<Control>>;
  using OutVec = linalg::Vector<T, linalg::mat_rows_v<Output>>;
  static constexpr TimeDomainType DType = Type;
};

/// @brief extract scalar type from ss
template <typename StateSpace>
using ss_type_t = typename ss_traits<linalg::remove_cvref_t<StateSpace>>::T;

/// @brief ss plant type
template <typename StateSpace>
using ss_plant_t =
    typename ss_traits<linalg::remove_cvref_t<StateSpace>>::Plant;

/// @brief ss control type
template <typename StateSpace>
using ss_control_t =
    typename ss_traits<linalg::remove_cvref_t<StateSpace>>::Control;

/// @brief ss output type
template <typename StateSpace>
using ss_output_t =
    typename ss_traits<linalg::remove_cvref_t<StateSpace>>::Output;

/// @brief extract state vector
template <typename StateSpace>
using state_vec_t =
    typename ss_traits<linalg::remove_cvref_t<StateSpace>>::State;

/// @brief extract input vector
template <typename StateSpace>
using input_vec_t =
    typename ss_traits<linalg::remove_cvref_t<StateSpace>>::Input;

/// @brief extract output vector
template <typename StateSpace>
using output_vec_t =
    typename ss_traits<linalg::remove_cvref_t<StateSpace>>::OutVec;

/// @brief check if continuous or not
template <typename StateSpace>
inline constexpr bool is_continuous_v =
    ss_traits<linalg::remove_cvref_t<StateSpace>>::DType ==
    TimeDomainType::CONTINUOUS;

/// @brief StateSpace concept
template <typename StateSpace>
concept StateSpaceLike = requires(const StateSpace& ss) {
  { ss.A };
  { ss.B };
  { ss.C };
};

/// @brief ContinuousStateSpace concept
template <typename StateSpace>
concept ContinuousStateSpaceLike =
    StateSpaceLike<StateSpace> && is_continuous_v<StateSpace>;

/// @brief ContinuousStateSpace concept
template <typename StateSpace>
concept DiscreteStateSpaceLike =
    StateSpaceLike<StateSpace> && !is_continuous_v<StateSpace>;

/// @brief Forward declaration of PlantModel
template <StateSpaceLike SS, typename InputPorts, typename OutputPorts>
struct PlantModel;

/// @brief base type traits for plant model
template <class Plant>
struct plant_traits;

/// @brief explicit type traits for plant model
template <StateSpaceLike SS, typename InputPortsT, typename OutputPortsT>
struct plant_traits<PlantModel<SS, InputPortsT, OutputPortsT>> {
  using StateSpace = linalg::remove_cvref_t<SS>;
  using InputPorts = linalg::remove_cvref_t<InputPortsT>;
  using OutputPorts = linalg::remove_cvref_t<OutputPortsT>;

  using T = ss_type_t<StateSpace>;
  using State = state_vec_t<StateSpace>;
  using Input = input_vec_t<StateSpace>;
  using Output = output_vec_t<StateSpace>;
  static constexpr TimeDomainType DType = ss_traits<StateSpace>::DType;

  static constexpr std::size_t StateDim = linalg::vector_dim_v<State>;
  static constexpr std::size_t InputDim = linalg::vector_dim_v<Input>;
  static constexpr std::size_t OutputDim = linalg::vector_dim_v<Output>;

  static constexpr std::size_t ExogenousDim =
      InputPorts::template total_size<PortRole::EXOGENOUS>();
  static constexpr std::size_t ControlDim =
      InputPorts::template total_size<PortRole::CONTROL>();

  static constexpr std::size_t PerformanceDim =
      OutputPorts::template total_size<PortRole::PERFORMANCE>();
  static constexpr std::size_t MeasurementDim =
      OutputPorts::template total_size<PortRole::MEASUREMENT>();
};

/// @brief extract scalar type from plant
template <typename Plant>
using plant_type_t = typename plant_traits<linalg::remove_cvref_t<Plant>>::T;

/// @brief extract state-space type from plant
template <typename Plant>
using plant_state_space_t =
    typename plant_traits<linalg::remove_cvref_t<Plant>>::StateSpace;

/// @brief extract input port layout type from plant
template <typename Plant>
using plant_input_ports_t =
    typename plant_traits<linalg::remove_cvref_t<Plant>>::InputPorts;

/// @brief extract output port layout type from plant
template <typename Plant>
using plant_output_ports_t =
    typename plant_traits<linalg::remove_cvref_t<Plant>>::OutputPorts;

/// @brief extract state vector from plant
template <typename Plant>
using plant_state_vec_t =
    typename plant_traits<linalg::remove_cvref_t<Plant>>::State;

/// @brief extract full input vector from plant
template <typename Plant>
using plant_input_vec_t =
    typename plant_traits<linalg::remove_cvref_t<Plant>>::Input;

/// @brief extract full output vector from plant
template <typename Plant>
using plant_output_vec_t =
    typename plant_traits<linalg::remove_cvref_t<Plant>>::Output;

/// @brief PlantModel concept
template <typename Plant>
concept PlantModelLike = requires(const Plant& plant) {
  typename plant_traits<linalg::remove_cvref_t<Plant>>::StateSpace;
  { plant.state_space };
  { plant.exogenous_matrix() };
  { plant.control_matrix() };
  { plant.performance_matrix() };
  { plant.measurement_matrix() };
};

/// @brief check if plant model is continuous
template <typename Plant>
inline constexpr bool plant_is_continuous_v =
    plant_traits<linalg::remove_cvref_t<Plant>>::DType ==
    TimeDomainType::CONTINUOUS;

/// @brief ContinuousPlantModel concept
template <typename Plant>
concept ContinuousPlantModelLike =
    PlantModelLike<Plant> && plant_is_continuous_v<Plant>;

/// @brief DiscretePlantModel concept
template <typename Plant>
concept DiscretePlantModelLike =
    PlantModelLike<Plant> && !plant_is_continuous_v<Plant>;

} // namespace control
