/*
 * Copyright © 2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "akreon_native/core.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Solve continuous-time f64 state-feedback synthesis.
 *
 * Inputs are `A: (n, n)` and `B: (n, m)`. Output `K` must have at least
 * `(m, n)` writable elements.
 *
 * @param A borrowed dynamics matrix view
 * @param B borrowed control matrix view
 * @param config optional solver config, or null for defaults
 * @param K borrowed mutable gain output view
 * @param result solver diagnostics output
 *
 * @return native status code
 */
ak_native_status ak_state_feedback_solve_f64(
    ak_native_matrix_view_f64 A, ak_native_matrix_view_f64 B,
    const ak_native_solver_config_f64* config, ak_native_matrix_mut_f64 K,
    ak_native_solver_result_f64* result);

/**
 * @brief Solve continuous-time f64 Luenberger state-observer synthesis.
 *
 * Inputs are `A: (n, n)` and `C: (y, n)`. Output `L` must have at least
 * `(n, y)` writable elements.
 *
 * @param A borrowed dynamics matrix view
 * @param C borrowed measurement matrix view
 * @param config optional solver config, or null for defaults
 * @param L borrowed mutable observer gain output view
 * @param result solver diagnostics output
 *
 * @return native status code
 */
ak_native_status ak_state_observer_solve_f64(
    ak_native_matrix_view_f64 A, ak_native_matrix_view_f64 C,
    const ak_native_solver_config_f64* config, ak_native_matrix_mut_f64 L,
    ak_native_solver_result_f64* result);

/**
 * @brief Solve continuous-time f64 H2 state-feedback synthesis.
 *
 * Inputs are generalized plant matrices with input layout `[w, u]` and
 * performance-output rows only.
 *
 * @param A borrowed dynamics matrix view, shape `(n, n)`
 * @param B borrowed input matrix view, shape `(n, w + m)`
 * @param C borrowed performance output matrix view, shape `(z, n)`
 * @param D borrowed performance feedthrough matrix view, shape `(z, w + m)`
 * @param exogenous_dim number of leading exogenous input columns
 * @param config optional solver config, or null for defaults
 * @param K borrowed mutable gain output view, shape at least `(m, n)`
 * @param result solver diagnostics output
 *
 * @return native status code
 */
ak_native_status ak_h2_state_feedback_solve_f64(
    ak_native_matrix_view_f64 A, ak_native_matrix_view_f64 B,
    ak_native_matrix_view_f64 C, ak_native_matrix_view_f64 D,
    size_t exogenous_dim, const ak_native_solver_config_f64* config,
    ak_native_matrix_mut_f64 K, ak_native_solver_result_f64* result);

/**
 * @brief Solve continuous-time f64 Hinf state-feedback synthesis.
 *
 * Inputs are generalized plant matrices with input layout `[w, u]` and
 * performance-output rows only.
 *
 * @param A borrowed dynamics matrix view, shape `(n, n)`
 * @param B borrowed input matrix view, shape `(n, w + m)`
 * @param C borrowed performance output matrix view, shape `(z, n)`
 * @param D borrowed performance feedthrough matrix view, shape `(z, w + m)`
 * @param exogenous_dim number of leading exogenous input columns
 * @param config optional solver config, or null for defaults
 * @param K borrowed mutable gain output view, shape at least `(m, n)`
 * @param result solver diagnostics output
 *
 * @return native status code
 */
ak_native_status ak_hinf_state_feedback_solve_f64(
    ak_native_matrix_view_f64 A, ak_native_matrix_view_f64 B,
    ak_native_matrix_view_f64 C, ak_native_matrix_view_f64 D,
    size_t exogenous_dim, const ak_native_solver_config_f64* config,
    ak_native_matrix_mut_f64 K, ak_native_solver_result_f64* result);

/**
 * @brief Solve continuous-time f64 H2 state-observer synthesis.
 *
 * Inputs are generalized plant matrices with input layout `[w]` and output
 * layout `[z, y]`.
 *
 * @param A borrowed dynamics matrix view, shape `(n, n)`
 * @param B borrowed exogenous input matrix view, shape `(n, w)`
 * @param C borrowed output matrix view, shape `(z + y, n)`
 * @param D borrowed feedthrough matrix view, shape `(z + y, w)`
 * @param performance_dim number of leading performance output rows
 * @param config optional solver config, or null for defaults
 * @param L borrowed mutable observer gain output view, shape at least `(n, y)`
 * @param result solver diagnostics output
 *
 * @return native status code
 */
ak_native_status ak_h2_state_observer_solve_f64(
    ak_native_matrix_view_f64 A, ak_native_matrix_view_f64 B,
    ak_native_matrix_view_f64 C, ak_native_matrix_view_f64 D,
    size_t performance_dim, const ak_native_solver_config_f64* config,
    ak_native_matrix_mut_f64 L, ak_native_solver_result_f64* result);

/**
 * @brief Solve continuous-time f64 Hinf state-observer synthesis.
 *
 * Inputs are generalized plant matrices with input layout `[w]` and output
 * layout `[z, y]`.
 *
 * @param A borrowed dynamics matrix view, shape `(n, n)`
 * @param B borrowed exogenous input matrix view, shape `(n, w)`
 * @param C borrowed output matrix view, shape `(z + y, n)`
 * @param D borrowed feedthrough matrix view, shape `(z + y, w)`
 * @param performance_dim number of leading performance output rows
 * @param config optional solver config, or null for defaults
 * @param L borrowed mutable observer gain output view, shape at least `(n, y)`
 * @param result solver diagnostics output
 *
 * @return native status code
 */
ak_native_status ak_hinf_state_observer_solve_f64(
    ak_native_matrix_view_f64 A, ak_native_matrix_view_f64 B,
    ak_native_matrix_view_f64 C, ak_native_matrix_view_f64 D,
    size_t performance_dim, const ak_native_solver_config_f64* config,
    ak_native_matrix_mut_f64 L, ak_native_solver_result_f64* result);

/**
 * @brief Optional objective for direct polytopic state-feedback synthesis.
 */
typedef struct ak_native_polytopic_state_feedback_objective_f64 {
  /** @brief Optional linear weights for the symmetric P decision block. */
  ak_native_matrix_view_f64 p_weights;
  /** @brief Optional linear weights for the Y decision block. */
  ak_native_matrix_view_f64 y_weights;
  /** @brief Optional quadratic objective matrix over the full decision vector.
   */
  ak_native_matrix_view_f64 quadratic_P;
  /** @brief Optional quadratic objective vector over the full decision vector.
   */
  ak_native_matrix_view_f64 quadratic_q;
} ak_native_polytopic_state_feedback_objective_f64;

/**
 * @brief Optional objective for direct polytopic observer synthesis.
 */
typedef struct ak_native_polytopic_state_observer_objective_f64 {
  /** @brief Optional quadratic objective matrix over the full decision vector.
   */
  ak_native_matrix_view_f64 quadratic_P;
  /** @brief Optional quadratic objective vector over the full decision vector.
   */
  ak_native_matrix_view_f64 quadratic_q;
} ak_native_polytopic_state_observer_objective_f64;

/**
 * @brief Solve continuous-time f64 direct polytopic state-feedback synthesis.
 *
 * @param A_vertices borrowed array of `vertex_count` dynamics matrix views
 * @param B_vertices borrowed array of `vertex_count` control matrix views
 * @param vertex_count number of polytope vertices
 * @param objective optional linear or quadratic objective, or null for default
 * @param config optional solver config, or null for defaults
 * @param K borrowed mutable gain output view, shape at least `(m, n)`
 * @param result solver diagnostics output
 *
 * @return native status code
 */
ak_native_status ak_polytopic_state_feedback_solve_f64(
    const ak_native_matrix_view_f64* A_vertices,
    const ak_native_matrix_view_f64* B_vertices, size_t vertex_count,
    const ak_native_polytopic_state_feedback_objective_f64* objective,
    const ak_native_solver_config_f64* config, ak_native_matrix_mut_f64 K,
    ak_native_solver_result_f64* result);

/**
 * @brief Solve continuous-time f64 direct polytopic state-observer synthesis.
 *
 * @param A_vertices borrowed array of `vertex_count` dynamics matrix views
 * @param C_vertices borrowed array of `vertex_count` measurement matrix views
 * @param vertex_count number of polytope vertices
 * @param objective optional quadratic objective, or null for default
 * @param config optional solver config, or null for defaults
 * @param L borrowed mutable observer gain output view, shape at least `(n, y)`
 * @param result solver diagnostics output
 *
 * @return native status code
 */
ak_native_status ak_polytopic_state_observer_solve_f64(
    const ak_native_matrix_view_f64* A_vertices,
    const ak_native_matrix_view_f64* C_vertices, size_t vertex_count,
    const ak_native_polytopic_state_observer_objective_f64* objective,
    const ak_native_solver_config_f64* config, ak_native_matrix_mut_f64 L,
    ak_native_solver_result_f64* result);

/**
 * @brief Solve continuous-time f64 H2 polytopic state-feedback synthesis.
 *
 * @param A_vertices borrowed array of dynamics matrix views, each shape `(n,
 * n)`
 * @param B_vertices borrowed array of input matrix views, each shape `(n, w +
 * m)`
 * @param C_vertices borrowed array of performance matrix views, each shape `(z,
 * n)`
 * @param D_vertices borrowed array of feedthrough views, each shape `(z, w +
 * m)`
 * @param vertex_count number of polytope vertices
 * @param exogenous_dim number of leading exogenous input columns
 * @param config optional solver config, or null for defaults
 * @param K borrowed mutable gain output view, shape at least `(m, n)`
 * @param result solver diagnostics output
 *
 * @return native status code
 */
ak_native_status ak_h2_polytopic_state_feedback_solve_f64(
    const ak_native_matrix_view_f64* A_vertices,
    const ak_native_matrix_view_f64* B_vertices,
    const ak_native_matrix_view_f64* C_vertices,
    const ak_native_matrix_view_f64* D_vertices, size_t vertex_count,
    size_t exogenous_dim, const ak_native_solver_config_f64* config,
    ak_native_matrix_mut_f64 K, ak_native_solver_result_f64* result);

/**
 * @brief Solve continuous-time f64 Hinf polytopic state-feedback synthesis.
 *
 * @param A_vertices borrowed array of dynamics matrix views, each shape `(n,
 * n)`
 * @param B_vertices borrowed array of input matrix views, each shape `(n, w +
 * m)`
 * @param C_vertices borrowed array of performance matrix views, each shape `(z,
 * n)`
 * @param D_vertices borrowed array of feedthrough views, each shape `(z, w +
 * m)`
 * @param vertex_count number of polytope vertices
 * @param exogenous_dim number of leading exogenous input columns
 * @param config optional solver config, or null for defaults
 * @param K borrowed mutable gain output view, shape at least `(m, n)`
 * @param result solver diagnostics output
 *
 * @return native status code
 */
ak_native_status ak_hinf_polytopic_state_feedback_solve_f64(
    const ak_native_matrix_view_f64* A_vertices,
    const ak_native_matrix_view_f64* B_vertices,
    const ak_native_matrix_view_f64* C_vertices,
    const ak_native_matrix_view_f64* D_vertices, size_t vertex_count,
    size_t exogenous_dim, const ak_native_solver_config_f64* config,
    ak_native_matrix_mut_f64 K, ak_native_solver_result_f64* result);

/**
 * @brief Solve continuous-time f64 H2 polytopic state-observer synthesis.
 *
 * @param A_vertices borrowed array of dynamics matrix views, each shape `(n,
 * n)`
 * @param B_vertices borrowed array of exogenous input views, each shape `(n,
 * w)`
 * @param C_vertices borrowed array of output matrix views, each shape `(z + y,
 * n)`
 * @param D_vertices borrowed array of feedthrough views, each shape `(z + y,
 * w)`
 * @param vertex_count number of polytope vertices
 * @param performance_dim number of leading performance output rows
 * @param config optional solver config, or null for defaults
 * @param L borrowed mutable observer gain output view, shape at least `(n, y)`
 * @param result solver diagnostics output
 *
 * @return native status code
 */
ak_native_status ak_h2_polytopic_state_observer_solve_f64(
    const ak_native_matrix_view_f64* A_vertices,
    const ak_native_matrix_view_f64* B_vertices,
    const ak_native_matrix_view_f64* C_vertices,
    const ak_native_matrix_view_f64* D_vertices, size_t vertex_count,
    size_t performance_dim, const ak_native_solver_config_f64* config,
    ak_native_matrix_mut_f64 L, ak_native_solver_result_f64* result);

/**
 * @brief Solve continuous-time f64 Hinf polytopic state-observer synthesis.
 *
 * @param A_vertices borrowed array of dynamics matrix views, each shape `(n,
 * n)`
 * @param B_vertices borrowed array of exogenous input views, each shape `(n,
 * w)`
 * @param C_vertices borrowed array of output matrix views, each shape `(z + y,
 * n)`
 * @param D_vertices borrowed array of feedthrough views, each shape `(z + y,
 * w)`
 * @param vertex_count number of polytope vertices
 * @param performance_dim number of leading performance output rows
 * @param config optional solver config, or null for defaults
 * @param L borrowed mutable observer gain output view, shape at least `(n, y)`
 * @param result solver diagnostics output
 *
 * @return native status code
 */
ak_native_status ak_hinf_polytopic_state_observer_solve_f64(
    const ak_native_matrix_view_f64* A_vertices,
    const ak_native_matrix_view_f64* B_vertices,
    const ak_native_matrix_view_f64* C_vertices,
    const ak_native_matrix_view_f64* D_vertices, size_t vertex_count,
    size_t performance_dim, const ak_native_solver_config_f64* config,
    ak_native_matrix_mut_f64 L, ak_native_solver_result_f64* result);

#ifdef __cplusplus
}
#endif
