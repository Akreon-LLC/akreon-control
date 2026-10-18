/*
 * Copyright © 2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Native C ABI major version. */
#define AK_NATIVE_VERSION_MAJOR 0
/** @brief Native C ABI minor version. */
#define AK_NATIVE_VERSION_MINOR 1
/** @brief Native C ABI patch version. */
#define AK_NATIVE_VERSION_PATCH 0

/** @brief Maximum canonical trace records copied into native result objects. */
#define AK_NATIVE_CANONICAL_TRACE_MAX_RECORDS 128
/** @brief Maximum canonical cone/equality vector values copied into results. */
#define AK_NATIVE_CANONICAL_VECTOR_MAX_VALUES 512

/**
 * @brief Status code returned by Akreon native C ABI entrypoints.
 */
typedef enum ak_native_status {
  /** @brief Operation completed successfully. */
  AK_NATIVE_OK = 0,
  /** @brief Input argument, shape, stride, value, or config is invalid. */
  AK_NATIVE_INVALID_ARGUMENT = 1,
  /** @brief Requested template dimensions are not compiled into this binary. */
  AK_NATIVE_UNSUPPORTED_DIMENSION = 2,
  /** @brief Caller-provided mutable output buffer is too small. */
  AK_NATIVE_OUTPUT_BUFFER_TOO_SMALL = 3,
  /** @brief Solver completed but did not produce an acceptable optimal result.
   */
  AK_NATIVE_SOLVER_FAILED = 4,
  /** @brief Fatal numerical event occurred during solve or gain recovery. */
  AK_NATIVE_NUMERICAL_ERROR = 5,
  /** @brief Unexpected native exception or internal failure. */
  AK_NATIVE_INTERNAL_ERROR = 6
} ak_native_status;

/**
 * @brief Semantic version triple for the native C ABI.
 */
typedef struct ak_native_version_info {
  /** @brief Major ABI version. */
  uint32_t major;
  /** @brief Minor ABI version. */
  uint32_t minor;
  /** @brief Patch ABI version. */
  uint32_t patch;
} ak_native_version_info;

/**
 * @brief Borrowed immutable row-major f64 matrix view.
 *
 * `stride` is measured in scalar elements, not bytes. The view does not own the
 * pointed-to memory.
 */
typedef struct ak_native_matrix_view_f64 {
  /** @brief Pointer to the first scalar element. */
  const double* data;
  /** @brief Number of logical matrix rows. */
  size_t rows;
  /** @brief Number of logical matrix columns. */
  size_t cols;
  /** @brief Row stride in scalar elements. */
  size_t stride;
} ak_native_matrix_view_f64;

/**
 * @brief Borrowed mutable row-major f64 matrix view.
 *
 * `stride` is measured in scalar elements, not bytes. The view does not own the
 * pointed-to memory.
 */
typedef struct ak_native_matrix_mut_f64 {
  /** @brief Pointer to the first writable scalar element. */
  double* data;
  /** @brief Number of logical writable rows. */
  size_t rows;
  /** @brief Number of logical writable columns. */
  size_t cols;
  /** @brief Row stride in scalar elements. */
  size_t stride;
} ak_native_matrix_mut_f64;

/**
 * @brief Native f64 primal-dual solver configuration.
 */
typedef struct ak_native_solver_config_f64 {
  /** @brief Compiled primal-dual IPM iteration budget; must equal PD_IPM_STEPS.
   */
  size_t steps;
  /** @brief Maximum backtracking iterations per primal-dual step. */
  size_t backtrack_steps;
  /** @brief Primal feasibility tolerance. */
  double primal_feas_tol;
  /** @brief Dual feasibility tolerance. */
  double dual_feas_tol;
  /** @brief Duality gap tolerance. */
  double duality_gap_tol;
  /** @brief Global Tikhonov regularization for the Schur/KKT system. */
  double global_reg_eps;
  /** @brief Equality-direction regularization for weak equality directions. */
  double eq_reg_eps;
} ak_native_solver_config_f64;

/**
 * @brief Structural canonicalization record kind.
 */
typedef enum ak_native_canonical_record_kind {
  /** @brief Canonical node record. */
  AK_NATIVE_CANONICAL_RECORD_NODE = 0,
  /** @brief Cone slice record aligned to primal/dual cone vectors. */
  AK_NATIVE_CANONICAL_RECORD_CONE = 1,
  /** @brief Equality slice record aligned to equality dual vectors. */
  AK_NATIVE_CANONICAL_RECORD_EQUALITY = 2
} ak_native_canonical_record_kind;

/**
 * @brief Canonical cone kind.
 */
typedef enum ak_native_canonical_cone_kind {
  /** @brief Nonnegative orthant cone. */
  AK_NATIVE_CANONICAL_CONE_NONNEG_ORTHANT = 0,
  /** @brief Second-order cone. */
  AK_NATIVE_CANONICAL_CONE_SECOND_ORDER = 1,
  /** @brief Positive-semidefinite cone. */
  AK_NATIVE_CANONICAL_CONE_POSITIVE_SEMIDEFINITE = 2,
  /** @brief Exponential cone. */
  AK_NATIVE_CANONICAL_CONE_EXP = 3,
  /** @brief Power cone. */
  AK_NATIVE_CANONICAL_CONE_POWER = 4,
  /** @brief Unknown or not applicable. */
  AK_NATIVE_CANONICAL_CONE_UNKNOWN = 255
} ak_native_canonical_cone_kind;

/**
 * @brief Flattened canonical trace record.
 *
 * For node records, `atom_type` is the Akreon Convex AtomType value and
 * `offset`/`dim` are zero. For cone records, `offset`/`dim` slice `x`, `y`, and
 * `rp`. For equality records, `offset`/`dim` slice `nu` and `req`.
 */
typedef struct ak_native_canonical_record {
  /** @brief One of ak_native_canonical_record_kind. */
  uint32_t record_kind;
  /** @brief Akreon Convex AtomType value for node records; otherwise zero. */
  uint32_t atom_type;
  /** @brief One of ak_native_canonical_cone_kind for cone records. */
  uint32_t cone_kind;
  /** @brief Canonical node id associated with this record. */
  size_t node_id;
  /** @brief Parent node id, or SIZE_MAX when no parent is recorded. */
  size_t parent_id;
  /** @brief Local cone/equality index inside the associated node. */
  size_t local_index;
  /** @brief Global cone index for cone records; otherwise zero. */
  size_t global_index;
  /** @brief Global vector offset for cone/equality records. */
  size_t offset;
  /** @brief Global slice dimension for cone/equality records. */
  size_t dim;
} ak_native_canonical_record;

/**
 * @brief Native f64 solver diagnostics returned by solve entrypoints.
 */
typedef struct ak_native_solver_result_f64 {
  /** @brief Akreon Convex solver status code. */
  int32_t solver_status_code;
  /** @brief Akreon Linalg numerical event code. */
  int32_t numerical_event_code;
  /** @brief Nonzero when line search or solver progress stalled. */
  int32_t stalled;
  /** @brief Number of solver iterations consumed. */
  size_t num_iter;
  /** @brief Final normalized primal-dual gap `mu`. */
  double normalized_duality_gap;
  /** @brief Final Akreon Convex primal objective `c^T z`. */
  double primal_objective_value;
  /** @brief Final Akreon Convex dual objective `-g^T y`. */
  double dual_objective_value;
  /** @brief Final dual residual norm `||rd||_2`. */
  double dual_residual_norm;
  /** @brief Final primal residual norm `||rp||_2`. */
  double primal_residual_norm;
  /** @brief Final equality residual norm `||req||_2`. */
  double equality_residual_norm;
  /** @brief Objective value or H2 epigraph value when available. */
  double objective_value;
  /** @brief Hinf gamma value when available. */
  double gamma;
  /** @brief Number of populated canonical trace records. */
  size_t canonical_trace_record_count;
  /** @brief Nonzero when canonical trace records were truncated. */
  int32_t canonical_trace_truncated;
  /** @brief Flattened canonical trace records. */
  ak_native_canonical_record
      canonical_trace_records[AK_NATIVE_CANONICAL_TRACE_MAX_RECORDS];
  /** @brief Number of copied cone-vector values. */
  size_t cone_value_count;
  /** @brief Nonzero when cone-vector values were truncated. */
  int32_t cone_values_truncated;
  /** @brief Final primal cone vector `x`. */
  double cone_primal_values[AK_NATIVE_CANONICAL_VECTOR_MAX_VALUES];
  /** @brief Final dual cone vector `y`. */
  double cone_dual_values[AK_NATIVE_CANONICAL_VECTOR_MAX_VALUES];
  /** @brief Final primal cone residual vector `rp`. */
  double cone_residual_values[AK_NATIVE_CANONICAL_VECTOR_MAX_VALUES];
  /** @brief Number of copied equality-vector values. */
  size_t equality_value_count;
  /** @brief Nonzero when equality-vector values were truncated. */
  int32_t equality_values_truncated;
  /** @brief Final equality dual vector `nu`. */
  double equality_dual_values[AK_NATIVE_CANONICAL_VECTOR_MAX_VALUES];
  /** @brief Final equality residual vector `req`. */
  double equality_residual_values[AK_NATIVE_CANONICAL_VECTOR_MAX_VALUES];
} ak_native_solver_result_f64;

/**
 * @brief Return the native C ABI version.
 *
 * @return semantic version triple
 */
ak_native_version_info ak_native_version(void);

/**
 * @brief Return a stable status name for a native status code.
 *
 * @param status native status code
 *
 * @return null-terminated static string
 */
const char* ak_native_status_name(ak_native_status status);

/**
 * @brief Return a stable Akreon Convex solver status name.
 *
 * @param solver_status_code Akreon Convex solver status code
 *
 * @return null-terminated static string
 */
const char* ak_native_solver_status_name(int32_t solver_status_code);

/**
 * @brief Return a stable Akreon Linalg numerical event name.
 *
 * @param numerical_event_code Akreon Linalg numerical event code
 *
 * @return null-terminated static string
 */
const char* ak_native_numerical_event_name(int32_t numerical_event_code);

/**
 * @brief Return the default f64 solver config used by native bindings.
 *
 * @return default primal-dual IPM config
 */
ak_native_solver_config_f64 ak_native_default_solver_config_f64(void);

#ifdef __cplusplus
}
#endif
