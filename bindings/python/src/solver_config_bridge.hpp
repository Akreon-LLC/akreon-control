/*
 * Copyright © 2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>

#include <nanobind/nanobind.h>

#include "akreon_native/akreon_native.h"
#include "reflection_bridge.hpp"

namespace akreon_native::python {

struct SolverConfig {
  size_t steps{};
  size_t backtrack_steps{};
  double primal_feas_tol{};
  double dual_feas_tol{};
  double duality_gap_tol{};
  double global_reg_eps{};
  double eq_reg_eps{};

  SolverConfig() {
    const auto defaults = ak_native_default_solver_config_f64();
    steps = defaults.steps;
    backtrack_steps = defaults.backtrack_steps;
    primal_feas_tol = defaults.primal_feas_tol;
    dual_feas_tol = defaults.dual_feas_tol;
    duality_gap_tol = defaults.duality_gap_tol;
    global_reg_eps = defaults.global_reg_eps;
    eq_reg_eps = defaults.eq_reg_eps;
  }

  ak_native_solver_config_f64 native() const {
    return ak_native_solver_config_f64{
        .steps = steps,
        .backtrack_steps = backtrack_steps,
        .primal_feas_tol = primal_feas_tol,
        .dual_feas_tol = dual_feas_tol,
        .duality_gap_tol = duality_gap_tol,
        .global_reg_eps = global_reg_eps,
        .eq_reg_eps = eq_reg_eps,
    };
  }

  bool is_default() const {
    const auto defaults = ak_native_default_solver_config_f64();
    return steps == defaults.steps &&
           backtrack_steps == defaults.backtrack_steps &&
           primal_feas_tol == defaults.primal_feas_tol &&
           dual_feas_tol == defaults.dual_feas_tol &&
           duality_gap_tol == defaults.duality_gap_tol &&
           global_reg_eps == defaults.global_reg_eps &&
           eq_reg_eps == defaults.eq_reg_eps;
  }
};

inline const ak_native_solver_config_f64* native_config_ptr(
    const SolverConfig& config, ak_native_solver_config_f64& storage) {
  storage = config.native();
  return config.is_default() ? nullptr : &storage;
}

inline void bind_solver_config_class(nb::module_& m) {
  auto cls = nb::class_<SolverConfig>(m, "SolverConfig").def(nb::init<>());
  bind_readwrite_fields(
      cls, Field<&SolverConfig::steps>{"steps"},
      Field<&SolverConfig::backtrack_steps>{"backtrack_steps"},
      Field<&SolverConfig::primal_feas_tol>{"primal_feas_tol"},
      Field<&SolverConfig::dual_feas_tol>{"dual_feas_tol"},
      Field<&SolverConfig::duality_gap_tol>{"duality_gap_tol"},
      Field<&SolverConfig::global_reg_eps>{"global_reg_eps"},
      Field<&SolverConfig::eq_reg_eps>{"eq_reg_eps"});
}

} // namespace akreon_native::python
