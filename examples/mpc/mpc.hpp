/*
 * Copyright © 2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <akreon/control/control.hpp>
#include <akreon/convex/convex.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <tuple>
#include <string>
#include <utility>

namespace example::mpc {
using State = linalg::Vector<double, 2>;
using Mat = linalg::Matrix<double, 2, 2>;
using Input = linalg::Matrix<double, 2, 1>;
using Gain = linalg::Matrix<double, 1, 2>;
using System =
    control::StateSpace<control::TimeDomainType::DISCRETE, Mat, Input, Mat>;

struct Config {
  double dt{0.2};
  double position_weight{1.0};
  double velocity_weight{0.1};
  double input_weight{0.1};
  double max_acceleration{1.0};
  double max_velocity{2.0};
};

constexpr auto make_system(const Config& c) {
  for (double value : {c.dt, c.position_weight, c.velocity_weight,
                       c.input_weight, c.max_acceleration, c.max_velocity}) {
    if (!std::isfinite(value) || value <= 0)
      throw std::invalid_argument(
          "MPC configuration values must be finite and positive");
  }
  // Exact zero-order-hold discretization of p_dot = v, v_dot = u.
  return System(Mat{State{1.0, c.dt}, State{0.0, 1.0}},
                Input{linalg::Vector<double, 1>{0.5 * c.dt * c.dt},
                      linalg::Vector<double, 1>{c.dt}},
                linalg::make_identity<double, 2>());
}

inline void check_solve(const auto& result, const std::string& context) {
  if (result.tag != config::SolverResult::OPTIMAL || result.stalled ||
      linalg::is_fatal_event(result.event))
    throw std::runtime_error(context + " optimization failed: status=" +
                             std::to_string(static_cast<int>(result.tag)) +
                             ", iterations=" + std::to_string(result.num_iter) +
                             ", gap=" + std::to_string(result.mu));
}

// H2 is continuous-time: synthesize K for the continuous double integrator,
// then evaluate that gain on the sampled plant to obtain the terminal cost.
inline auto terminal_cost(const System& system, const Config& c) {
  using H2System = control::StateSpace<control::TimeDomainType::CONTINUOUS, Mat,
                                       linalg::Matrix<double, 2, 3>,
                                       linalg::Matrix<double, 3, 2>>;
  using Inputs =
      control::PortLayout<control::Port<control::PortRole::EXOGENOUS, 0, 2>,
                          control::Port<control::PortRole::CONTROL, 2, 1>>;
  using Outputs =
      control::PortLayout<control::Port<control::PortRole::PERFORMANCE, 0, 3>>;
  const auto plant = control::PlantModel<H2System, Inputs, Outputs>(H2System(
      Mat{State{0.0, 1.0}, State{0.0, 0.0}},
      linalg::Matrix<double, 2, 3>{linalg::Vector<double, 3>{1.0, 0.0, 0.0},
                                   linalg::Vector<double, 3>{0.0, 1.0, 1.0}},
      linalg::Matrix<double, 3, 2>{State{std::sqrt(c.position_weight), 0.0},
                                   State{0.0, std::sqrt(c.velocity_weight)},
                                   State{0.0, 0.0}},
      linalg::Matrix<double, 3, 3>{
          linalg::Vector<double, 3>{0.0, 0.0, 0.0},
          linalg::Vector<double, 3>{0.0, 0.0, 0.0},
          linalg::Vector<double, 3>{0.0, 0.0, std::sqrt(c.input_weight)}}));
  using H2 = control::H2StateFeedback<decltype(plant)>;
  constexpr auto decision = convex::Variable<
      convex::VarMeta<linalg::RowVector<double, H2::DecVarDim>>>();
  const auto problem = control::make_h2_state_feedback_problem(plant, decision);
  config::PrimalDualIPMConfig<double> settings;
  settings.steps = 128;
  const convex::PrimalDualIPM solver(problem, settings);
  check_solve(solver.res, "H2");
  const auto [q, yz] = linalg::splice<H2::QDim>(solver.res.iterate.z);
  const auto gramian = linalg::make_smat<2>(q);
  const Gain Y{State{yz(0), yz(1)}};
  const auto [K, event] = H2{}.K(gramian, Y);
  if (linalg::is_fatal_event(event))
    throw std::runtime_error("H2 gain recovery failed");

  const Mat Q{State{c.position_weight, 0.0}, State{0.0, c.velocity_weight}};
  const auto closed = system.A + system.B * K;
  const auto stage = Q + c.input_weight * linalg::transpose(K) * K;
  Mat P{};
  // Fixed-point evaluation of the discrete Lyapunov equation, with a checked
  // residual. Divergent sampled gains fail instead of producing a tail cost.
  for (int iteration = 0; iteration < 1000; ++iteration) {
    const auto next = linalg::transpose(closed) * P * closed + stage;
    double residual = 0;
    for (std::size_t i = 0; i < 2; ++i)
      for (std::size_t j = 0; j < 2; ++j) {
        if (!std::isfinite(next(i, j)))
          throw std::runtime_error(
              "Sampled H2 gain does not yield a finite terminal cost");
        residual = std::max(residual, std::abs(next(i, j) - P(i, j)));
      }
    P = next;
    if (residual < 1e-10) {
      if (!linalg::Cholesky<Mat>(P).is_valid())
        throw std::runtime_error("Invalid terminal cost");
      return std::pair{P, K};
    }
  }
  throw std::runtime_error(
      "Discrete terminal-cost evaluation did not converge");
}

// X = Sx*x + Su*U, X=[x1,...,xN], U=[u0,...,u(N-1)].
// The stage cost includes x0,...,x(N-1); x0's constant is omitted from the QP.
template <std::size_t N>
struct Horizon {
  static_assert(N > 1);
  using Sequence = linalg::Vector<double, N>;
  using Hessian = linalg::Matrix<double, N, N>;
  using Bounds = linalg::Vector<double, 4 * N>;
  static constexpr double objective_scale = 0.01;
  linalg::Matrix<double, 2 * N, 2> Sx{};
  linalg::Matrix<double, 2 * N, N> Su{};
  linalg::Matrix<double, 2 * N, 2 * N> W{};
  Hessian H{};
  linalg::Matrix<double, N, 2> F{};
  linalg::Matrix<double, 4 * N, N> G{};
  Bounds limits{};

  constexpr Horizon(const System& system, const Config& c, const Mat& P) {
    // Build predictions recursively; every new input affects all later states.
    Mat power = linalg::make_identity<double, 2>();
    for (std::size_t i = 0; i < N; ++i) {
      power = system.A * power;
      for (std::size_t r = 0; r < 2; ++r) {
        for (std::size_t col = 0; col < 2; ++col)
          Sx(2 * i + r, col) = power(r, col);
        Su(2 * i + r, i) = system.B(r, 0);
        for (std::size_t j = 0; j < i; ++j)
          for (std::size_t k = 0; k < 2; ++k)
            Su(2 * i + r, j) += system.A(r, k) * Su(2 * (i - 1) + k, j);
      }
      const Mat weight = i + 1 == N ? P
                                    : Mat{State{c.position_weight, 0.0},
                                          State{0.0, c.velocity_weight}};
      for (std::size_t r = 0; r < 2; ++r)
        for (std::size_t col = 0; col < 2; ++col)
          W(2 * i + r, 2 * i + col) = weight(r, col);
      G(i, i) = 1.0;
      G(N + i, i) = -1.0;
      limits(i) = limits(N + i) = c.max_acceleration;
      limits(2 * N + i) = limits(3 * N + i) = c.max_velocity;
      for (std::size_t j = 0; j < N; ++j) {
        G(2 * N + i, j) = Su(2 * i + 1, j);
        G(3 * N + i, j) = -Su(2 * i + 1, j);
      }
    }
    H = 2.0 * (linalg::transpose(Su) * W * Su +
               c.input_weight * linalg::make_identity<double, N>());
    F = 2.0 * linalg::transpose(Su) * W * Sx;
  }

  // Shift the previous primal plan and append bounded terminal feedback.
  // Rebuild the cost lifts and cone seed for the new horizon; old duals are
  // deliberately reinitialized because their stage alignment has changed.
  template <typename Iterate, typename Problem>
  auto warm_start(const Iterate& previous, double tail_input,
                  const Problem& problem) const {
    // Native QuadForm appends a norm lift followed by its squared epigraph.
    static_assert(linalg::vector_dim_v<decltype(previous.z)> == N + 2);
    Iterate seed{};
    for (std::size_t i = 0; i + 1 < N; ++i) seed.z(i) = previous.z(i + 1);
    seed.z(N - 1) = tail_input;
    Sequence inputs{};
    for (std::size_t i = 0; i < N; ++i) inputs(i) = seed.z(i);
    const auto weighted = H * inputs;
    double quadratic = 0;
    for (std::size_t i = 0; i < N; ++i) quadratic += inputs(i) * weighted(i);
    seed.z(N) =
        std::sqrt(std::max(0.0, 0.5 * objective_scale * quadratic)) + 0.01;
    seed.z(N + 1) = seed.z(N) * seed.z(N) + 0.01;
    seed.x = problem.constraint_map()(seed.z);
    // Move cone variables into the interior and initialize duals from the
    // negative barrier gradient. The solver handles the resulting residuals.
    std::size_t offset = 0;
    std::apply(
        [&](const auto&... cones) {
          auto recenter = [&](const auto& cone) {
            const auto x0 = cone.x0();
            auto point = x0;
            for (std::size_t j = 0; j < linalg::vector_dim_v<decltype(x0)>;
                 ++j) {
              point(j) = seed.x(offset + j) + 0.01 * x0(j);
            }
            if (!cone.in_cone(point)) point = x0;
            const auto [gradient, event] = cone.grad(point);
            if (linalg::is_fatal_event(event))
              throw std::runtime_error("Invalid warm seed");
            for (std::size_t j = 0; j < linalg::vector_dim_v<decltype(x0)>;
                 ++j) {
              seed.x(offset) = point(j);
              seed.y(offset) = -0.01 * gradient(j);
              ++offset;
            }
          };
          (recenter(cones), ...);
        },
        problem.constraint_cones());
    return seed;
  }

  auto problem(const State& state) const {
    if (!std::isfinite(state(0)) || !std::isfinite(state(1)))
      throw std::invalid_argument("MPC state must be finite");
    using Space = linalg::RowVector<double, N>;
    constexpr auto U = convex::Variable<convex::VarMeta<Space>>();
    auto b = limits;
    const auto drift = Sx * state;
    for (std::size_t i = 0; i < N; ++i) {
      b(2 * N + i) -= drift(2 * i + 1);
      b(3 * N + i) += drift(2 * i + 1);
    }
    // Positive objective scaling improves the conic epigraph conditioning.
    return convex::make_problem(U)
        .in_set(convex::Polyhedron(G, b, U))
        .minimize(convex::make_quad_form(objective_scale * H,
                                         objective_scale * (F * state), U));
  }
};
} // namespace example::mpc
