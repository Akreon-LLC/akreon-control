/*
 * Copyright © 2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */
#include "mhe.hpp"

#include <iostream>
#include <optional>
#include <random>
#include <string>

int main() {
  using namespace example::mhe;
  try {
    constexpr Config config{.dt = 0.2,
                            .measurement_std = 0.1,
                            .acceleration_std = 0.05,
                            .arrival_position_std = 1.0,
                            .arrival_velocity_std = 0.5,
                            .max_velocity = 2.0};
    constexpr std::size_t intervals = 10;
    constexpr int simulation_steps = 120;
    const auto system = make_system(config);
    const Window<intervals> window(system, config);
    State truth{4.0, 0.6};
    State arrival_prior{3.0, 0.0};
    // A fixed-gain observer consumes only the measurement leaving the window.
    constexpr State arrival_gain{0.5, 0.1};
    const auto input = [](int k) { return (k / 20) % 2 == 0 ? 0.2 : -0.2; };
    const auto disturbance = [](double t) { return 0.04 * std::sin(1.1 * t); };
    std::mt19937 random(42);
    std::uniform_real_distribution<double> noise(
        -std::sqrt(3.0) * config.measurement_std,
        std::sqrt(3.0) * config.measurement_std);
    Window<intervals>::Measurements measurements{};
    Window<intervals>::Inputs inputs{};
    using Problem =
        decltype(window.problem(arrival_prior, measurements, inputs));
    using Solver = convex::PrimalDualIPM<Problem>;
    using Iterate =
        convex::PrimalDualIterate<Solver::Var, Solver::ConeVar, Solver::NuVar>;
    std::optional<Iterate> previous;
    State shifted_initial{};
    config::PrimalDualIPMConfig<double> settings;
    settings.steps = 128;
    settings.Delta = 10.0;
    double position_error = 0, velocity_error = 0;
    double raw_position_error = 0, difference_velocity_error = 0;
    std::size_t solves = 0, cold_iterations = 0, warm_iterations = 0,
                max_warm = 0;
    State estimate{};
    for (int k = 0; k <= simulation_steps; ++k) {
      if (k > 0) {
        const double u = input(k - 1);
        truth = system.state_update(
            truth,
            linalg::Vector<double, 1>{u + disturbance((k - 1) * config.dt)});
        if (k > static_cast<int>(intervals)) {
          for (std::size_t i = 0; i < intervals; ++i)
            measurements[i] = measurements[i + 1];
          for (std::size_t i = 0; i + 1 < intervals; ++i)
            inputs[i] = inputs[i + 1];
        }
        inputs[std::min(static_cast<std::size_t>(k - 1), intervals - 1)] = u;
      }
      measurements[std::min(static_cast<std::size_t>(k), intervals)] =
          truth(0) + noise(random);
      if (k < static_cast<int>(intervals)) continue; // Fill the first window.
      const auto problem = window.problem(arrival_prior, measurements, inputs);
      const auto solver =
          previous
              ? Solver(problem,
                       window.warm_start(
                           *previous, shifted_initial - arrival_prior, problem),
                       settings)
              : Solver(problem, settings);
      const auto& result = solver.res;
      if (result.tag != config::SolverResult::OPTIMAL || result.stalled ||
          linalg::is_fatal_event(result.event))
        throw std::runtime_error(
            "MHE solve failed at sample " + std::to_string(k) +
            ", iterations=" + std::to_string(result.num_iter));
      if (!previous) cold_iterations = result.num_iter;
      else {
        warm_iterations += result.num_iter;
        max_warm = std::max(max_warm, result.num_iter);
      }
      Window<intervals>::Decision decision{};
      for (std::size_t i = 0; i < window.Dim; ++i)
        decision(i) = result.iterate.z(i);
      const auto stacked =
          window.E * decision + window.drift(arrival_prior, inputs);
      State predicted = arrival_prior + State{decision(0), decision(1)};
      for (std::size_t i = 0; i <= intervals; ++i) {
        if (!std::isfinite(predicted(0)) || !std::isfinite(predicted(1)) ||
            std::abs(predicted(1)) > config.max_velocity + 1e-3 ||
            std::abs(predicted(0) - stacked(2 * i)) > 1e-9 ||
            std::abs(predicted(1) - stacked(2 * i + 1)) > 1e-9)
          throw std::runtime_error("MHE trajectory failed physical checks");
        if (i == 1) shifted_initial = predicted;
        if (i < intervals)
          predicted = system.state_update(
              predicted,
              linalg::Vector<double, 1>{inputs[i] + decision(i + 2)});
      }
      estimate = predicted;
      position_error += std::pow(estimate(0) - truth(0), 2);
      velocity_error += std::pow(estimate(1) - truth(1), 2);
      raw_position_error += std::pow(measurements.back() - truth(0), 2);
      const double difference =
          (measurements[intervals] - measurements[intervals - 1]) / config.dt;
      difference_velocity_error += std::pow(difference - truth(1), 2);
      ++solves;
      previous = result.iterate;
      // y_0 is excluded from the next window: incorporate it once, then
      // predict the arrival prior to that window's first state using u_0.
      const State corrected =
          arrival_prior + (measurements[0] - arrival_prior(0)) * arrival_gain;
      arrival_prior =
          system.state_update(corrected, linalg::Vector<double, 1>{inputs[0]});
    }
    const double p_rmse = std::sqrt(position_error / solves);
    const double v_rmse = std::sqrt(velocity_error / solves);
    const double raw_rmse = std::sqrt(raw_position_error / solves);
    const double difference_rmse =
        std::sqrt(difference_velocity_error / solves);
    if (!std::isfinite(p_rmse) || !std::isfinite(v_rmse) ||
        p_rmse >= raw_rmse || v_rmse >= difference_rmse || p_rmse > 0.15 ||
        v_rmse > 0.2)
      throw std::runtime_error("MHE estimation-error check failed");
    std::cout << "Final estimated position / velocity: " << estimate(0) << " / "
              << estimate(1) << '\n'
              << "Final true position / velocity: " << truth(0) << " / "
              << truth(1) << '\n'
              << "Position RMSE (MHE / measurement): " << p_rmse << " / "
              << raw_rmse << '\n'
              << "Velocity RMSE (MHE / finite difference): " << v_rmse << " / "
              << difference_rmse << '\n'
              << "Cold iterations: " << cold_iterations << '\n'
              << "Warm iterations (mean / max): "
              << static_cast<double>(warm_iterations) / (solves - 1) << " / "
              << max_warm << '\n';
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
