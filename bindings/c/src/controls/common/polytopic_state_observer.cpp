/*
 * Copyright © 2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../controls_common.hpp"
#include "polytopic_observer.hpp"

using namespace akreon_native::controls_detail;

/**
 * @brief Polytopic observer accepts vertex arrays for A/C only.
 * Objective matrices are optional ABI inputs and are validated against the
 * compiled polytopic observer decision layout.
 */
ak_native_status ak_polytopic_state_observer_solve_f64(
    const ak_native_matrix_view_f64* A_vertices,
    const ak_native_matrix_view_f64* C_vertices, size_t vertex_count,
    const ak_native_polytopic_state_observer_objective_f64* objective,
    const ak_native_solver_config_f64* config, ak_native_matrix_mut_f64 L,
    ak_native_solver_result_f64* result) {
  /** @brief Normalize caller-owned outputs before validating vertex arrays. */
  if (result == nullptr) return AK_NATIVE_INVALID_ARGUMENT;
  reset_result(result);
  /**
   * @brief Polytopic validation checks vertex consistency plus optional
   * objective shape before dispatch enters templated Akreon Control code.
   */
  if (invalid_config(config)) return AK_NATIVE_INVALID_ARGUMENT;
  const auto valid = validate_polytopic_observer(A_vertices, C_vertices,
                                                 vertex_count, objective, L);
  if (valid != AK_NATIVE_OK) return valid;
  try {
    /** @brief Dispatch selects vertex/state/measurement dimensions compiled
     * from YAML. */
    return dispatch_polytopic_observer(A_vertices, C_vertices, vertex_count,
                                       objective, config, L, result);
  } catch (const std::exception&) {
    return AK_NATIVE_INTERNAL_ERROR;
  } catch (...) { return AK_NATIVE_INTERNAL_ERROR; }
}
