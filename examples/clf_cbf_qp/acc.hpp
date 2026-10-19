/*
 * Copyright © 2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <akreon/convex/convex.hpp>

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>

namespace example::acc {
struct Config {
  double desired_speed{25.0};
  double standstill_gap{5.0};
  double headway{1.8};
  double barrier_reserve{0.1};
  double min_acceleration{-3.0};
  double max_acceleration{2.0};
  double clf_rate{1.0};
  double cbf_rate{0.5};
  double slack_weight{0.01};
  double nominal_gain{0.5};
};
struct State {
  double speed;
  double lead_speed;
  double gap;
};
struct Command {
  double acceleration;
  double slack;
  double barrier;
  double clf_residual; // <= 0
  double cbf_residual; // >= 0, includes reserve
  std::size_t iterations;
};
struct InfeasibleQP : std::runtime_error {
  using std::runtime_error::runtime_error;
};

constexpr void validate(const Config& c) {
  for (double value :
       {c.desired_speed, c.standstill_gap, c.headway, c.barrier_reserve,
        c.min_acceleration, c.max_acceleration, c.clf_rate, c.cbf_rate,
        c.slack_weight, c.nominal_gain}) {
    if (!std::isfinite(value))
      throw std::invalid_argument("ACC configuration must be finite");
  }
  if (c.headway <= 0 || c.clf_rate <= 0 || c.cbf_rate <= 0 ||
      c.slack_weight <= 0 || c.desired_speed < 0 || c.standstill_gap < 0 ||
      c.barrier_reserve < 0 || c.nominal_gain < 0 || c.min_acceleration >= 0 ||
      c.max_acceleration <= 0)
    throw std::invalid_argument("Invalid ACC configuration");
}
constexpr void validate(const State& s) {
  for (double value : {s.speed, s.lead_speed, s.gap}) {
    if (!std::isfinite(value) || value < 0)
      throw std::invalid_argument(
          "ACC speeds and gap must be finite and nonnegative");
  }
}

// All quantities are SI. a is net acceleration, not wheel force.
// Assemble [a, delta] with a soft CLF and a hard CBF in the native DSL.
inline auto make_problem(const State& s, const Config& c) {
  using Space = linalg::Matrix<double, 1, 2>;
  using Vec = linalg::Vector<double, 2>;
  validate(c);
  validate(s);
  const double e = s.speed - c.desired_speed;
  const double h = s.gap - c.standstill_gap - c.headway * s.speed;
  const double cbf_bound =
      s.lead_speed - s.speed + c.cbf_rate * (h - c.barrier_reserve);
  if (cbf_bound / c.headway < c.min_acceleration)
    throw InfeasibleQP("CBF and actuator limits are incompatible");

  constexpr auto x = convex::Variable<convex::VarMeta<Space>>();
  linalg::Matrix<double, 5, 2> A{Vec{e, -1.0}, Vec{c.headway, 0.0},
                                 Vec{-1.0, 0.0}, Vec{1.0, 0.0}, Vec{0.0, -1.0}};
  linalg::Vector<double, 5> b{-c.clf_rate * 0.5 * e * e, cbf_bound,
                              -c.min_acceleration, c.max_acceleration, 0.0};
  // Positive row scaling preserves the feasible set and balances the cone map.
  for (std::size_t row = 0; row < 5; ++row) {
    const double scale = std::max(
        {1.0, std::abs(b(row)), std::abs(A(row, 0)), std::abs(A(row, 1))});
    b(row) /= scale;
    A(row, 0) /= scale;
    A(row, 1) /= scale;
  }
  const double nominal = -c.nominal_gain * e;
  const linalg::Matrix<double, 2, 2> P{Vec{1.0, 0.0}, Vec{0.0, c.slack_weight}};
  const auto cost =
      convex::make_quad_form(P, Vec{-nominal, 0.0}, 0.5 * nominal * nominal, x);
  return convex::make_problem(x)
      .in_set(convex::Polyhedron(A, b, x))
      .minimize(cost);
}

using Problem = decltype(make_problem(State{}, Config{}));
using Solver = convex::PrimalDualIPM<Problem>;
using Iterate =
    convex::PrimalDualIterate<Solver::Var, Solver::ConeVar, Solver::NuVar>;

inline Command control(const State& s, const Config& c,
                       std::optional<Iterate>& previous) {
  const auto problem = make_problem(s, c);
  config::PrimalDualIPMConfig<double> solver_config;
  solver_config.steps = 128;
  // A wider proximity neighborhood avoids premature backtracking stalls in
  // this small conic QP. Keep the native residual and gap tolerances.
  solver_config.Delta = 10.0;
  solver_config.duality_gap_tol = 1e-5;
  // Move the reused cone variables slightly into the interior. Previous
  // optima can sit too close to the boundary for the next problem's steps.
  if (previous) {
    std::size_t offset = 0;
    std::apply(
        [&](const auto&... cones) {
          auto recenter = [&](const auto& cone) {
            const auto x0 = cone.x0();
            const auto y0 = cone.y0();
            for (std::size_t j = 0; j < linalg::vector_dim_v<decltype(x0)>;
                 ++j) {
              previous->x(offset) = 0.99 * previous->x(offset) + 0.01 * x0(j);
              previous->y(offset) = 0.99 * previous->y(offset) + 0.01 * y0(j);
              ++offset;
            }
          };
          (recenter(cones), ...);
        },
        problem.constraint_cones());
  }
  // Cold start once, then reuse all primal-dual and epigraph coordinates.
  const auto solver = previous ? Solver(problem, *previous, solver_config)
                               : Solver(problem, solver_config);
  const auto& result = solver.res;
  if (result.tag != config::SolverResult::OPTIMAL ||
      linalg::is_fatal_event(result.event) || result.stalled)
    throw std::runtime_error("ACC native QP solve failed: status=" +
                             std::to_string(static_cast<int>(result.tag)) +
                             ", iterations=" + std::to_string(result.num_iter) +
                             ", speed=" + std::to_string(s.speed) + ", event=" +
                             std::to_string(static_cast<int>(result.event)) +
                             ", stalled=" + std::to_string(result.stalled) +
                             ", a=" + std::to_string(result.iterate.z(0)) +
                             ", delta=" + std::to_string(result.iterate.z(1)) +
                             ", mu=" + std::to_string(result.mu));
  // Original decision variables precede the conic epigraph auxiliaries.
  const double a = result.iterate.z(0);
  const double delta = result.iterate.z(1);
  const double e = s.speed - c.desired_speed;
  const double h = s.gap - c.standstill_gap - c.headway * s.speed;
  const double clf = e * a + c.clf_rate * 0.5 * e * e - delta;
  const double cbf = s.lead_speed - s.speed - c.headway * a +
                     c.cbf_rate * (h - c.barrier_reserve);
  // Validate physical residuals too, rather than relying on solver status
  // alone.
  constexpr double tolerance = 1e-3;
  if (!std::isfinite(a) || !std::isfinite(delta) || !std::isfinite(clf) ||
      !std::isfinite(cbf) || clf > tolerance || cbf < -tolerance ||
      delta < -tolerance || a < c.min_acceleration - tolerance ||
      a > c.max_acceleration + tolerance)
    throw std::runtime_error("ACC native QP solution failed residual checks");
  previous = result.iterate;
  return {a, delta, h, clf, cbf, result.num_iter};
}

constexpr State step(const State& s, double a, double al, double dt) {
  return {s.speed + a * dt, s.lead_speed + al * dt,
          s.gap + (s.lead_speed - s.speed) * dt + 0.5 * (al - a) * dt * dt};
}
} // namespace example::acc
