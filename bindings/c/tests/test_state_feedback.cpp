/*
 * Copyright © 2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>

#include "akreon_native/akreon_native.h"

namespace {

int require(bool condition, int code) {
  if (!condition) std::fprintf(stderr, "native C ABI check failed: %d\n", code);
  return condition ? 0 : code;
}

} // namespace

int main() {
  if (require(ak_native_status_name(AK_NATIVE_OK) != nullptr, 1)) return 1;

  const double A[16] = {
      -1.0, 0.0,  0.0,  0.0, //
      0.0,  -1.4, 0.0,  0.0, //
      0.0,  0.0,  -1.8, 0.0, //
      0.0,  0.0,  0.0,  -2.2,
  };
  const double B[4] = {1.0, 0.5, 0.25, 0.1};
  const double B_gen[8] = {
      0.2, 1.0,  //
      0.0, 0.5,  //
      0.1, 0.25, //
      0.0, 0.1,
  };
  const double C_perf[20] = {
      1.0, 0.0, 0.0, 0.0, //
      0.0, 1.0, 0.0, 0.0, //
      0.0, 0.0, 1.0, 0.0, //
      0.0, 0.0, 0.0, 1.0, //
      0.0, 0.0, 0.0, 0.0,
  };
  const double D_h2_perf[10] = {
      0.0, 0.0, //
      0.0, 0.0, //
      0.0, 0.0, //
      0.0, 0.0, //
      0.0, 1.0,
  };
  const double D_hinf_perf[10] = {
      0.1, 0.0, //
      0.0, 0.0, //
      0.0, 0.0, //
      0.0, 0.0, //
      0.0, 1.0,
  };
  const double C_measurement[4] = {1.0, 0.0, 0.0, 0.0};
  const double C_obs[8] = {
      1.0, 0.0, 0.0, 0.0, //
      0.0, 1.0, 0.0, 0.0,
  };
  const double B_gen_obs[12] = {
      0.2, 0.0, 0.0, //
      0.0, 1.0, 0.0, //
      0.1, 0.0, 0.0, //
      0.0, 0.0, 1.0,
  };
  const double C_gen_obs[24] = {
      1.0, 0.0, 0.0, 0.0, //
      0.0, 1.0, 0.0, 0.0, //
      0.0, 0.0, 1.0, 0.0, //
      0.0, 0.0, 0.0, 1.0, //
      1.0, 0.0, 0.0, 0.0, //
      0.0, 1.0, 0.0, 0.0,
  };
  const double D_h2_obs[18] = {
      0.0, 0.0, 0.0, //
      0.0, 0.0, 0.0, //
      0.0, 0.0, 0.0, //
      0.0, 0.0, 0.0, //
      0.0, 1.0, 0.0, //
      0.0, 0.0, 1.0,
  };
  const double D_hinf_obs[18] = {
      0.0, 0.0, 0.0, //
      0.0, 0.0, 0.0, //
      0.0, 0.0, 0.0, //
      0.0, 0.0, 0.0, //
      0.0, 1.0, 0.0, //
      0.0, 0.0, 1.0,
  };

  double K[4] = {};
  double L[16] = {};
  ak_native_solver_result_f64 result{};
  auto cfg = ak_native_default_solver_config_f64();

  auto status = ak_state_feedback_solve_f64(
      ak_native_matrix_view_f64{A, 4, 4, 4},
      ak_native_matrix_view_f64{B, 4, 1, 1}, &cfg,
      ak_native_matrix_mut_f64{K, 1, 4, 4}, &result);
  if (status != AK_NATIVE_OK) {
    std::fprintf(stderr,
                 "state feedback status=%d solver=%d event=%d stalled=%d "
                 "iterations=%zu mu=%.17g\n",
                 static_cast<int>(status), result.solver_status_code,
                 result.numerical_event_code, result.stalled, result.num_iter,
                 result.normalized_duality_gap);
  }
  if (require(status == AK_NATIVE_OK, 2)) return 2;
  if (require(result.solver_status_code == 1, 3)) return 3;
  if (require(std::isfinite(result.normalized_duality_gap), 4)) return 4;

  status = ak_h2_state_feedback_solve_f64(
      ak_native_matrix_view_f64{A, 4, 4, 4},
      ak_native_matrix_view_f64{B_gen, 4, 2, 2},
      ak_native_matrix_view_f64{C_perf, 5, 4, 4},
      ak_native_matrix_view_f64{D_h2_perf, 5, 2, 2}, 1, nullptr,
      ak_native_matrix_mut_f64{K, 1, 4, 4}, &result);
  if (require(status == AK_NATIVE_OK, 8)) return 8;
  if (require(result.objective_value > 0.0, 9)) return 9;

  status = ak_hinf_state_feedback_solve_f64(
      ak_native_matrix_view_f64{A, 4, 4, 4},
      ak_native_matrix_view_f64{B_gen, 4, 2, 2},
      ak_native_matrix_view_f64{C_perf, 5, 4, 4},
      ak_native_matrix_view_f64{D_hinf_perf, 5, 2, 2}, 1, nullptr,
      ak_native_matrix_mut_f64{K, 1, 4, 4}, &result);
  if (require(status == AK_NATIVE_OK, 10)) return 10;
  if (require(result.gamma > 0.0, 11)) return 11;

  status = ak_state_observer_solve_f64(
      ak_native_matrix_view_f64{A, 4, 4, 4},
      ak_native_matrix_view_f64{C_obs, 2, 4, 4}, nullptr,
      ak_native_matrix_mut_f64{L, 4, 4, 4}, &result);
  if (require(status == AK_NATIVE_OK, 12)) return 12;

  double L_go[8] = {};
  status = ak_h2_state_observer_solve_f64(
      ak_native_matrix_view_f64{A, 4, 4, 4},
      ak_native_matrix_view_f64{B_gen_obs, 4, 3, 3},
      ak_native_matrix_view_f64{C_gen_obs, 6, 4, 4},
      ak_native_matrix_view_f64{D_h2_obs, 6, 3, 3}, 4, nullptr,
      ak_native_matrix_mut_f64{L_go, 4, 2, 2}, &result);
  if (require(status == AK_NATIVE_OK, 13)) return 13;
  if (require(result.objective_value > 0.0, 14)) return 14;

  status = ak_hinf_state_observer_solve_f64(
      ak_native_matrix_view_f64{A, 4, 4, 4},
      ak_native_matrix_view_f64{B_gen_obs, 4, 3, 3},
      ak_native_matrix_view_f64{C_gen_obs, 6, 4, 4},
      ak_native_matrix_view_f64{D_hinf_obs, 6, 3, 3}, 4, nullptr,
      ak_native_matrix_mut_f64{L_go, 4, 2, 2}, &result);
  if (require(status == AK_NATIVE_OK, 15)) return 15;
  if (require(result.gamma > 0.0, 16)) return 16;

  const double A_poly[32] = {
      -1.0, 0.0, 0.0, 0.0, 0.0,  -1.4, 0.0, 0.0, 0.0, 0.0,  -1.8,
      0.0,  0.0, 0.0, 0.0, -2.2, -1.1, 0.0, 0.0, 0.0, 0.0,  -1.5,
      0.0,  0.0, 0.0, 0.0, -1.9, 0.0,  0.0, 0.0, 0.0, -2.3,
  };
  const double B_poly[8] = {1.0, 0.5, 0.25, 0.1, 1.0, 0.5, 0.25, 0.1};
  const ak_native_matrix_view_f64 A_poly_views[2] = {
      ak_native_matrix_view_f64{A_poly, 4, 4, 4},
      ak_native_matrix_view_f64{A_poly + 16, 4, 4, 4},
  };
  const ak_native_matrix_view_f64 B_poly_views[2] = {
      ak_native_matrix_view_f64{B_poly, 4, 1, 1},
      ak_native_matrix_view_f64{B_poly + 4, 4, 1, 1},
  };
  double K_poly[4] = {};
  status = ak_polytopic_state_feedback_solve_f64(
      A_poly_views, B_poly_views, 2, nullptr, nullptr,
      ak_native_matrix_mut_f64{K_poly, 1, 4, 4}, &result);
  if (require(status == AK_NATIVE_OK, 17)) return 17;

  const ak_native_matrix_view_f64 C_obs_views[2] = {
      ak_native_matrix_view_f64{C_obs, 2, 4, 4},
      ak_native_matrix_view_f64{C_obs, 2, 4, 4},
  };
  double L_poly[16] = {};
  status = ak_polytopic_state_observer_solve_f64(
      A_poly_views, C_obs_views, 2, nullptr, nullptr,
      ak_native_matrix_mut_f64{L_poly, 4, 4, 4}, &result);
  if (require(status == AK_NATIVE_OK, 19)) return 19;

  auto invalid = ak_state_feedback_solve_f64(
      ak_native_matrix_view_f64{A, 4, 4, 4},
      ak_native_matrix_view_f64{B_gen, 4, 2, 2}, &cfg,
      ak_native_matrix_mut_f64{K, 1, 4, 4}, &result);
  if (require(invalid == AK_NATIVE_OUTPUT_BUFFER_TOO_SMALL ||
                  invalid == AK_NATIVE_UNSUPPORTED_DIMENSION,
              27)) {
    return 27;
  }

  auto mismatched_steps_cfg = ak_native_default_solver_config_f64();
  mismatched_steps_cfg.steps += 1;
  invalid = ak_state_feedback_solve_f64(
      ak_native_matrix_view_f64{A, 4, 4, 4},
      ak_native_matrix_view_f64{B, 4, 1, 1}, &mismatched_steps_cfg,
      ak_native_matrix_mut_f64{K, 1, 4, 4}, &result);
  if (require(invalid == AK_NATIVE_INVALID_ARGUMENT, 28)) return 28;

  invalid = ak_polytopic_state_feedback_solve_f64(
      A_poly_views, B_poly_views, 4, nullptr, nullptr,
      ak_native_matrix_mut_f64{K_poly, 1, 4, 4}, &result);
  if (require(invalid == AK_NATIVE_UNSUPPORTED_DIMENSION, 29)) return 29;

  return 0;
}
