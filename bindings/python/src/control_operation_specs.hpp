/*
 * Copyright © 2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "akreon_native/akreon_native.h"

namespace akreon_native::python {

/**
 * @brief Nanobind-facing operation descriptors for C ABI facades.
 *
 * These specs keep Python registration boring: public Python name, workflow
 * label, and the native C ABI entrypoint. They are intentionally not a schema
 * for C++ internals.
 */
struct H2StateFeedbackOp {
  static constexpr const char* python_name = "solve_h2_state_feedback";
  static constexpr const char* workflow_name = "H2 state feedback";
  static constexpr auto call = ak_h2_state_feedback_solve_f64;
};

struct HinfStateFeedbackOp {
  static constexpr const char* python_name = "solve_hinf_state_feedback";
  static constexpr const char* workflow_name = "Hinf state feedback";
  static constexpr auto call = ak_hinf_state_feedback_solve_f64;
};

struct H2StateObserverOp {
  static constexpr const char* python_name = "solve_h2_state_observer";
  static constexpr const char* workflow_name = "H2 state observer";
  static constexpr auto call = ak_h2_state_observer_solve_f64;
};

struct HinfStateObserverOp {
  static constexpr const char* python_name = "solve_hinf_state_observer";
  static constexpr const char* workflow_name = "Hinf state observer";
  static constexpr auto call = ak_hinf_state_observer_solve_f64;
};

struct H2PolytopicStateFeedbackOp {
  static constexpr const char* python_name =
      "solve_h2_polytopic_state_feedback";
  static constexpr const char* workflow_name = "H2 polytopic state feedback";
  static constexpr auto call = ak_h2_polytopic_state_feedback_solve_f64;
};

struct HinfPolytopicStateFeedbackOp {
  static constexpr const char* python_name =
      "solve_hinf_polytopic_state_feedback";
  static constexpr const char* workflow_name = "Hinf polytopic state feedback";
  static constexpr auto call = ak_hinf_polytopic_state_feedback_solve_f64;
};

struct H2PolytopicStateObserverOp {
  static constexpr const char* python_name =
      "solve_h2_polytopic_state_observer";
  static constexpr const char* workflow_name = "H2 polytopic state observer";
  static constexpr auto call = ak_h2_polytopic_state_observer_solve_f64;
};

struct HinfPolytopicStateObserverOp {
  static constexpr const char* python_name =
      "solve_hinf_polytopic_state_observer";
  static constexpr const char* workflow_name = "Hinf polytopic state observer";
  static constexpr auto call = ak_hinf_polytopic_state_observer_solve_f64;
};

} // namespace akreon_native::python
