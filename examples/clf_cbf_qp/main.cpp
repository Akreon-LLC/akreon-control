/*
 * Copyright © 2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */
#include "acc.hpp"

#include <iostream>

int main() {
  using namespace example::acc;
  try {
    // Edit these values to explore speed tracking and following distance.
    constexpr Config config{.desired_speed = 25.0,
                            .standstill_gap = 5.0,
                            .headway = 1.8,
                            .barrier_reserve = 0.1,
                            .min_acceleration = -3.0,
                            .max_acceleration = 2.0,
                            .clf_rate = 1.0,
                            .cbf_rate = 0.5,
                            .slack_weight = 0.01,
                            .nominal_gain = 0.5};
    State state{18.0, 20.0, 65.0};
    constexpr double dt = 0.02;
    constexpr double duration = 45.0;
    constexpr int steps = static_cast<int>(duration / dt);
    const auto lead_acceleration = [](double t) {
      if (10 <= t && t < 14) return -2.0;
      if (25 <= t && t < 33) return 1.0;
      return 0.0;
    };

    std::optional<Iterate> previous;
    std::size_t cold_iterations = 0;
    std::size_t warm_iterations = 0;
    std::size_t max_warm_iterations = 0;
    double minimum_h = state.gap;
    double maximum_slack = 0.0;
    for (int k = 0; k <= steps; ++k) {
      const auto command = control(state, config, previous);
      if (k == 0) {
        cold_iterations = command.iterations;
      } else {
        warm_iterations += command.iterations;
        max_warm_iterations = std::max(max_warm_iterations, command.iterations);
      }
      if (!std::isfinite(command.barrier) || command.barrier < 0)
        throw std::runtime_error("Sampled headway margin violated");
      minimum_h = std::min(minimum_h, command.barrier);
      maximum_slack = std::max(maximum_slack, command.slack);
      if (k < steps)
        state =
            step(state, command.acceleration, lead_acceleration(k * dt), dt);
    }
    std::cout << "Final speed: " << state.speed
              << " m/s (lead: " << state.lead_speed << " m/s)\n"
              << "Minimum headway margin: " << minimum_h << " m\n"
              << "Maximum CLF slack: " << maximum_slack << '\n'
              << "Cold iterations: " << cold_iterations << '\n'
              << "Warm iterations (mean / max): "
              << static_cast<double>(warm_iterations) / steps << " / "
              << max_warm_iterations << '\n';
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
