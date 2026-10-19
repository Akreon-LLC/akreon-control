/*
 * Copyright © 2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#include "akreon_native/core.h"

#include <cstdint>

#include <akreon/convex/solver_config.hpp>

/**
 * @brief Return the native C ABI version.
 *
 * @return semantic version triple
 */
ak_native_version_info ak_native_version(void) {
  return ak_native_version_info{AK_NATIVE_VERSION_MAJOR,
                                AK_NATIVE_VERSION_MINOR,
                                AK_NATIVE_VERSION_PATCH};
}

/**
 * @brief Return a stable status name for a native status code.
 *
 * @param status native status code
 *
 * @return null-terminated static string
 */
const char* ak_native_status_name(ak_native_status status) {
  switch (status) {
    case AK_NATIVE_OK:
      return "OK";
    case AK_NATIVE_INVALID_ARGUMENT:
      return "INVALID_ARGUMENT";
    case AK_NATIVE_UNSUPPORTED_DIMENSION:
      return "UNSUPPORTED_DIMENSION";
    case AK_NATIVE_OUTPUT_BUFFER_TOO_SMALL:
      return "OUTPUT_BUFFER_TOO_SMALL";
    case AK_NATIVE_SOLVER_FAILED:
      return "SOLVER_FAILED";
    case AK_NATIVE_NUMERICAL_ERROR:
      return "NUMERICAL_ERROR";
    case AK_NATIVE_INTERNAL_ERROR:
      return "INTERNAL_ERROR";
  }
  return "UNKNOWN";
}

/**
 * @brief Return a stable Akreon Convex solver status name.
 *
 * @param solver_status_code Akreon Convex solver status code
 *
 * @return null-terminated static string
 */
const char* ak_native_solver_status_name(int32_t solver_status_code) {
  switch (static_cast<config::SolverResult>(solver_status_code)) {
    case config::SolverResult::INFEASIBLE_OR_UNBOUNDED:
      return "INFEASIBLE_OR_UNBOUNDED";
    case config::SolverResult::OPTIMAL:
      return "OPTIMAL";
    case config::SolverResult::PRIMAL_FEASIBLE:
      return "PRIMAL_FEASIBLE";
    case config::SolverResult::DUAL_FEASIBLE:
      return "DUAL_FEASIBLE";
    case config::SolverResult::PRIMAL_DUAL_FEASIBLE:
      return "PRIMAL_DUAL_FEASIBLE";
    case config::SolverResult::PRIMAL_OPTIMAL:
      return "PRIMAL_OPTIMAL";
    case config::SolverResult::DUAL_OPTIMAL:
      return "DUAL_OPTIMAL";
    case config::SolverResult::INFEASIBLE_OPTIMAL:
      return "INFEASIBLE_OPTIMAL";
  }
  return "UNKNOWN";
}

/**
 * @brief Return a stable Akreon Linalg numerical event name.
 *
 * @param numerical_event_code Akreon Linalg numerical event code
 *
 * @return null-terminated static string
 */
const char* ak_native_numerical_event_name(int32_t numerical_event_code) {
  switch (static_cast<linalg::NumericalEvent>(numerical_event_code)) {
    case linalg::NumericalEvent::NONE:
      return "NONE";
    case linalg::NumericalEvent::RANK_DEFICIENCY:
      return "RANK_DEFICIENCY";
    case linalg::NumericalEvent::INDEFINITE_FACTORIZATION:
      return "INDEFINITE_FACTORIZATION";
    case linalg::NumericalEvent::NEAR_SINGULAR_TRIANGULAR_SOLVE:
      return "NEAR_SINGULAR_TRIANGULAR_SOLVE";
    case linalg::NumericalEvent::NUMERICAL_SAFEGUARD:
      return "NUMERICAL_SAFEGUARD";
    case linalg::NumericalEvent::DB_MAX_ITERS_REACHED:
      return "DB_MAX_ITERS_REACHED";
    case linalg::NumericalEvent::FACTORIZATION_FAILED:
      return "FACTORIZATION_FAILED";
    case linalg::NumericalEvent::EVD_FAILED:
      return "EVD_FAILED";
  }
  return "UNKNOWN";
}

/**
 * @brief Return the default f64 solver config used by native bindings.
 *
 * @return default primal-dual IPM config
 */
ak_native_solver_config_f64 ak_native_default_solver_config_f64(void) {
  return ak_native_solver_config_f64{
      .steps = PD_IPM_STEPS,
      .backtrack_steps = 64,
      .primal_feas_tol = 1e-5,
      .dual_feas_tol = 1e-5,
      .duality_gap_tol = 1e-5,
      .global_reg_eps = 1e-12,
      .eq_reg_eps = 1e-8,
  };
}
