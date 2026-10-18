/*
 * Copyright © 2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../controls_common.hpp"
#include "polytopic_feedback.hpp"

using namespace akreon_native::controls_detail;

/**
 * @brief Polytopic feedback accepts vertex arrays for A/B only.
 * Objective matrices are optional ABI inputs and are validated against the
 * compiled polytopic decision layout.
 */
ak_native_status ak_polytopic_state_feedback_solve_f64(
    const ak_native_matrix_view_f64* A_vertices,
    const ak_native_matrix_view_f64* B_vertices, size_t vertex_count,
    const ak_native_polytopic_state_feedback_objective_f64* objective,
    const ak_native_solver_config_f64* config, ak_native_matrix_mut_f64 K,
    ak_native_solver_result_f64* result) {
  /** @brief Normalize caller-owned outputs before validating vertex arrays. */
  if (result == nullptr) return AK_NATIVE_INVALID_ARGUMENT;
  reset_result(result);
  /**
   * @brief Polytopic validation checks vertex consistency plus optional
   * objective shape before dispatch enters templated Akreon Control code.
   */
  if (invalid_config(config)) return AK_NATIVE_INVALID_ARGUMENT;
  const auto valid = validate_polytopic_feedback(A_vertices, B_vertices,
                                                 vertex_count, objective, K);
  if (valid != AK_NATIVE_OK) return valid;
  try {
    /** @brief Dispatch selects vertex/state/input dimensions compiled from
     * YAML. */
    return dispatch_polytopic_feedback(A_vertices, B_vertices, vertex_count,
                                       objective, config, K, result);
  } catch (const std::exception&) {
    return AK_NATIVE_INTERNAL_ERROR;
  } catch (...) { return AK_NATIVE_INTERNAL_ERROR; }
}
