/*
 * Copyright © 2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../controls_common.hpp"
#include "hinf.hpp"

using namespace akreon_native::controls_detail;

/**
 * @brief Hinf polytopic observer uses vertex arrays for a generalized plant
 * with [exogenous] inputs and [performance, measurement] outputs.
 */
ak_native_status ak_hinf_polytopic_state_observer_solve_f64(
    const ak_native_matrix_view_f64* A_vertices,
    const ak_native_matrix_view_f64* B_vertices,
    const ak_native_matrix_view_f64* C_vertices,
    const ak_native_matrix_view_f64* D_vertices, size_t vertex_count,
    size_t performance_dim, const ak_native_solver_config_f64* config,
    ak_native_matrix_mut_f64 L, ak_native_solver_result_f64* result) {
  /** @brief Normalize caller-owned outputs before validating vertex arrays. */
  if (result == nullptr) return AK_NATIVE_INVALID_ARGUMENT;
  reset_result(result);
  /**
   * @brief Shared validation checks vertex consistency and generalized plant
   * splits; HinfPolytopicStateObserverProblem owns the controller solve.
   */
  if (invalid_config(config)) return AK_NATIVE_INVALID_ARGUMENT;
  const auto valid =
      validate_polytopic_observer(A_vertices, B_vertices, C_vertices,
                                  D_vertices, vertex_count, performance_dim, L);
  if (valid != AK_NATIVE_OK) return valid;
  try {
    /** @brief Dispatch selects compiled vertex and port dimensions only. */
    return dispatch_polytopic_observer<HinfPolytopicStateObserverProblem>(
        A_vertices, B_vertices, C_vertices, D_vertices, vertex_count,
        performance_dim, config, L, result);
  } catch (const std::exception&) {
    return AK_NATIVE_INTERNAL_ERROR;
  } catch (...) { return AK_NATIVE_INTERNAL_ERROR; }
}
