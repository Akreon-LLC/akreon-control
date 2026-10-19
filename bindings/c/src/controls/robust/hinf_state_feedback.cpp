/*
 * Copyright © 2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../controls_common.hpp"
#include "hinf.hpp"

using namespace akreon_native::controls_detail;

/**
 * @brief Hinf state-feedback uses a generalized plant with inputs laid out as
 * [exogenous, control], with the split supplied by the facade.
 */
ak_native_status ak_hinf_state_feedback_solve_f64(
    ak_native_matrix_view_f64 A, ak_native_matrix_view_f64 B,
    ak_native_matrix_view_f64 C, ak_native_matrix_view_f64 D,
    size_t exogenous_dim, const ak_native_solver_config_f64* config,
    ak_native_matrix_mut_f64 K, ak_native_solver_result_f64* result) {
  /** @brief Normalize caller-owned outputs before validation. */
  if (result == nullptr) return AK_NATIVE_INVALID_ARGUMENT;
  reset_result(result);
  /**
   * @brief The shared validator checks the ABI plant layout;
   * HinfStateFeedbackProblem owns the controller-specific Akreon Control solve.
   */
  if (invalid_config(config)) return AK_NATIVE_INVALID_ARGUMENT;
  const auto valid =
      validate_generalized_feedback(A, B, C, D, exogenous_dim, K);
  if (valid != AK_NATIVE_OK) return valid;
  try {
    /**
     * @brief Dispatch only selects compiled dimensions; it does not own Hinf
     * semantics.
     */
    return dispatch_generalized_feedback<HinfStateFeedbackProblem>(
        A, B, C, D, exogenous_dim, config, K, result);
  } catch (const std::exception&) {
    return AK_NATIVE_INTERNAL_ERROR;
  } catch (...) { return AK_NATIVE_INTERNAL_ERROR; }
}
