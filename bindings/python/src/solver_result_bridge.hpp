/*
 * Copyright © 2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <limits>
#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "akreon_native/akreon_native.h"
#include "ndarray_bridge.hpp"
#include "reflection_bridge.hpp"

namespace akreon_native::python {

struct SolverDiagnostics {
  std::string status;
  std::string event;
  bool stalled{};
  size_t num_iter{};
  double normalized_duality_gap{std::numeric_limits<double>::quiet_NaN()};
  double primal_objective_value{std::numeric_limits<double>::quiet_NaN()};
  double dual_objective_value{std::numeric_limits<double>::quiet_NaN()};
  double dual_residual_norm{std::numeric_limits<double>::quiet_NaN()};
  double primal_residual_norm{std::numeric_limits<double>::quiet_NaN()};
  double equality_residual_norm{std::numeric_limits<double>::quiet_NaN()};
  double objective_value{std::numeric_limits<double>::quiet_NaN()};
  double gamma{std::numeric_limits<double>::quiet_NaN()};
  nb::object canonical_trace;
  nb::object cone_primal_values;
  nb::object cone_dual_values;
  nb::object cone_residual_values;
  nb::object equality_dual_values;
  nb::object equality_residual_values;
};

struct StateFeedbackResult : SolverDiagnostics {
  nb::object K;
};

struct StateObserverResult : SolverDiagnostics {
  nb::object L;
};

constexpr int64_t trace_i64(size_t value) {
  constexpr auto kInt64Max =
      static_cast<size_t>(std::numeric_limits<int64_t>::max());
  return value > kInt64Max ? int64_t{-1} : static_cast<int64_t>(value);
}

inline nb::object make_trace_array(
    const ak_native_solver_result_f64& native_result) {
  auto object =
      make_numpy_empty_trace(native_result.canonical_trace_record_count);
  const auto trace = nb::cast<MutableTraceArray>(object);
  auto* out = trace.data();
  for (size_t i = 0; i < native_result.canonical_trace_record_count; ++i) {
    const auto& record = native_result.canonical_trace_records[i];
    const size_t row = 9 * i;
    out[row + 0] = static_cast<int64_t>(record.record_kind);
    out[row + 1] = static_cast<int64_t>(record.atom_type);
    out[row + 2] = static_cast<int64_t>(record.cone_kind);
    out[row + 3] = trace_i64(record.node_id);
    out[row + 4] = trace_i64(record.parent_id);
    out[row + 5] = trace_i64(record.local_index);
    out[row + 6] = trace_i64(record.global_index);
    out[row + 7] = trace_i64(record.offset);
    out[row + 8] = trace_i64(record.dim);
  }
  return object;
}

inline nb::object make_value_array(const double* values, size_t count) {
  auto object = make_numpy_empty_1d(count);
  const auto array = nb::cast<MutableArray1D>(object);
  auto* out = array.data();
  for (size_t i = 0; i < count; ++i) out[i] = values[i];
  return object;
}

inline void fill_common_result(const ak_native_solver_result_f64& native_result,
                               SolverDiagnostics& result) {
  result.status = std::string(
      ak_native_solver_status_name(native_result.solver_status_code));
  result.event = std::string(
      ak_native_numerical_event_name(native_result.numerical_event_code));
  result.stalled = native_result.stalled != 0;
  result.num_iter = native_result.num_iter;
  result.normalized_duality_gap = native_result.normalized_duality_gap;
  result.primal_objective_value = native_result.primal_objective_value;
  result.dual_objective_value = native_result.dual_objective_value;
  result.dual_residual_norm = native_result.dual_residual_norm;
  result.primal_residual_norm = native_result.primal_residual_norm;
  result.equality_residual_norm = native_result.equality_residual_norm;
  result.objective_value = native_result.objective_value;
  result.gamma = native_result.gamma;
  result.canonical_trace = make_trace_array(native_result);
  result.cone_primal_values = make_value_array(native_result.cone_primal_values,
                                               native_result.cone_value_count);
  result.cone_dual_values = make_value_array(native_result.cone_dual_values,
                                             native_result.cone_value_count);
  result.cone_residual_values = make_value_array(
      native_result.cone_residual_values, native_result.cone_value_count);
  result.equality_dual_values = make_value_array(
      native_result.equality_dual_values, native_result.equality_value_count);
  result.equality_residual_values =
      make_value_array(native_result.equality_residual_values,
                       native_result.equality_value_count);
}

template <typename Binding, typename Result>
Binding& bind_diagnostics(Binding& cls) {
  return bind_readonly_fields(
      cls, Field<&Result::status>{"status"}, Field<&Result::event>{"event"},
      Field<&Result::stalled>{"stalled"}, Field<&Result::num_iter>{"num_iter"},
      Field<&Result::normalized_duality_gap>{"normalized_duality_gap"},
      Field<&Result::primal_objective_value>{"primal_objective_value"},
      Field<&Result::dual_objective_value>{"dual_objective_value"},
      Field<&Result::dual_residual_norm>{"dual_residual_norm"},
      Field<&Result::primal_residual_norm>{"primal_residual_norm"},
      Field<&Result::equality_residual_norm>{"equality_residual_norm"},
      Field<&Result::objective_value>{"objective_value"},
      Field<&Result::gamma>{"gamma"},
      Field<&Result::canonical_trace>{"canonical_trace"},
      Field<&Result::cone_primal_values>{"cone_primal_values"},
      Field<&Result::cone_dual_values>{"cone_dual_values"},
      Field<&Result::cone_residual_values>{"cone_residual_values"},
      Field<&Result::equality_dual_values>{"equality_dual_values"},
      Field<&Result::equality_residual_values>{"equality_residual_values"});
}

inline void bind_result_classes(nb::module_& m) {
  auto feedback = nb::class_<StateFeedbackResult>(m, "StateFeedbackResult")
                      .def_ro("K", &StateFeedbackResult::K);
  bind_diagnostics<decltype(feedback), StateFeedbackResult>(feedback);

  auto observer = nb::class_<StateObserverResult>(m, "StateObserverResult")
                      .def_ro("L", &StateObserverResult::L);
  bind_diagnostics<decltype(observer), StateObserverResult>(observer);
}

} // namespace akreon_native::python
