/*
 * Copyright © 2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../controls_common.hpp"
#include "state_feedback.hpp"

using namespace akreon_native::controls_detail;

/**
 * @brief Direct state feedback keeps the C ABI validation local because its
 * plant shape is small: A is square, B is control-only, and K is sized M x N.
 */
ak_native_status ak_state_feedback_solve_f64(
    ak_native_matrix_view_f64 A, ak_native_matrix_view_f64 B,
    const ak_native_solver_config_f64* config, ak_native_matrix_mut_f64 K,
    ak_native_solver_result_f64* result) {
  /** @brief Every exit path should leave deterministic output metadata. */
  if (result == nullptr) return AK_NATIVE_INVALID_ARGUMENT;
  reset_result(result);

  if (invalid_config(config)) return AK_NATIVE_INVALID_ARGUMENT;
  const auto valid = validate_state_feedback(A, B, K);
  if (valid != AK_NATIVE_OK) return valid;

  try {
    /** @brief Runtime dimensions are converted into Akreon Control template
     * dimensions here. */
    return dispatch_state_feedback(A, B, config, K, result);
  } catch (const std::exception&) {
    return AK_NATIVE_INTERNAL_ERROR;
  } catch (...) { return AK_NATIVE_INTERNAL_ERROR; }
}
