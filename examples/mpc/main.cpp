/*
 * Copyright © 2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */
#include "mpc.hpp"

#include <iostream>
#include <optional>

int main() {
  using namespace example::mpc;
  try {
    constexpr Config config{.dt = 0.2,
                            .position_weight = 1.0,
                            .velocity_weight = 0.1,
                            .input_weight = 0.1,
                            .max_acceleration = 1.0,
                            .max_velocity = 2.0};
    constexpr std::size_t prediction_steps = 20;
    constexpr int simulation_steps = 60;
    State state{4.0, 0.0};
    const auto system = make_system(config);
    const auto [P, K] = terminal_cost(system, config);
    const Horizon<prediction_steps> horizon(system, config, P);
    config::PrimalDualIPMConfig<double> settings;
    settings.steps = 128;
    settings.Delta = 10.0;
    using Problem = decltype(horizon.problem(state));
    using Solver = convex::PrimalDualIPM<Problem>;
    using Iterate =
        convex::PrimalDualIterate<Solver::Var, Solver::ConeVar, Solver::NuVar>;
    std::optional<Iterate> previous;
    double tail_input = 0;
    double max_input = 0, max_velocity = 0;
    std::size_t cold_iterations = 0, warm_iterations = 0,
                maximum_warm_iterations = 0;
    for (int k = 0; k < simulation_steps; ++k) {
      const auto problem = horizon.problem(state);
      // Cold first; later samples seed the solver from the shifted input plan.
      const auto solver =
          previous ? Solver(problem,
                            horizon.warm_start(*previous, tail_input, problem),
                            settings)
                   : Solver(problem, settings);
      check_solve(solver.res, "MPC step " + std::to_string(k));
      if (k == 0) {
        cold_iterations = solver.res.num_iter;
      } else {
        warm_iterations += solver.res.num_iter;
        maximum_warm_iterations =
            std::max(maximum_warm_iterations, solver.res.num_iter);
      }
      const auto& z = solver.res.iterate.z;
      Horizon<prediction_steps>::Sequence inputs{};
      for (std::size_t i = 0; i < prediction_steps; ++i) inputs(i) = z(i);
      const auto stacked = horizon.Sx * state + horizon.Su * inputs;
      State predicted = state;
      // Check the whole predicted trajectory; apply only its first input.
      for (std::size_t i = 0; i < prediction_steps; ++i) {
        const double u = z(i);
        predicted =
            system.state_update(predicted, linalg::Vector<double, 1>{u});
        if (std::abs(predicted(0) - stacked(2 * i)) > 1e-9 ||
            std::abs(predicted(1) - stacked(2 * i + 1)) > 1e-9)
          throw std::runtime_error(
              "Condensed prediction disagrees with plant dynamics");
        if (!std::isfinite(u) || !std::isfinite(predicted(0)) ||
            !std::isfinite(predicted(1)) ||
            std::abs(u) > config.max_acceleration + 1e-3 ||
            std::abs(predicted(1)) > config.max_velocity + 1e-3)
          throw std::runtime_error("MPC prediction violated constraints");
      }
      tail_input = std::clamp((K * predicted)(0), -config.max_acceleration,
                              config.max_acceleration);
      previous = solver.res.iterate;
      max_input = std::max(max_input, std::abs(z(0)));
      state = system.state_update(state, linalg::Vector<double, 1>{z(0)});
      max_velocity = std::max(max_velocity, std::abs(state(1)));
    }
    if (std::abs(state(0)) > 1e-2 || std::abs(state(1)) > 1e-2)
      throw std::runtime_error("MPC did not regulate to the origin");
    std::cout << "H2 gain: [" << K(0, 0) << ", " << K(0, 1) << "]\n"
              << "Terminal P: [" << P(0, 0) << ", " << P(0, 1) << "; "
              << P(1, 0) << ", " << P(1, 1) << "]\n"
              << "Final position / velocity: " << state(0) << " / " << state(1)
              << '\n'
              << "Peak acceleration / velocity: " << max_input << " / "
              << max_velocity << '\n'
              << "Cold iterations: " << cold_iterations << '\n'
              << "Warm iterations (mean / max): "
              << static_cast<double>(warm_iterations) / (simulation_steps - 1)
              << " / " << maximum_warm_iterations << '\n';
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
