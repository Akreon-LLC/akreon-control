/*
 * Copyright © 2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <akreon/control/control.hpp>
#include <akreon/convex/convex.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <tuple>

namespace example::mhe {
using State = linalg::Vector<double, 2>;
using Mat = linalg::Matrix<double, 2, 2>;
using Input = linalg::Matrix<double, 2, 1>;
using Output = linalg::Matrix<double, 1, 2>;
using System =
    control::StateSpace<control::TimeDomainType::DISCRETE, Mat, Input, Output>;

struct Config {
  double dt{0.2};
  double measurement_std{0.1};
  double acceleration_std{0.05};
  double arrival_position_std{1.0};
  double arrival_velocity_std{0.5};
  double max_velocity{2.0};
};

constexpr auto make_system(const Config& c) {
  for (double value :
       {c.dt, c.measurement_std, c.acceleration_std, c.arrival_position_std,
        c.arrival_velocity_std, c.max_velocity})
    if (!std::isfinite(value) || value <= 0)
      throw std::invalid_argument(
          "MHE configuration must be finite and positive");
  using Continuous = control::StateSpace<control::TimeDomainType::CONTINUOUS,
                                         Mat, Input, Output>;
  return control::c2d_zoh(Continuous(Mat{State{0.0, 1.0}, State{0.0, 0.0}},
                                     Input{linalg::Vector<double, 1>{0.0},
                                           linalg::Vector<double, 1>{1.0}},
                                     Output{State{1.0, 0.0}}),
                          c.dt);
}

// Condense x_(i+1) = A*x_i + B*(u_i + d_i) into X = E*z + f.
// z = [x_0 - arrival_prior, d_0,...,d_(N-1)]; measure position only.
template <std::size_t N>
struct Window {
  static_assert(N > 1);
  static constexpr std::size_t Dim = N + 2;
  using Decision = linalg::Vector<double, Dim>;
  using Measurements = std::array<double, N + 1>;
  using Inputs = std::array<double, N>;
  using Trajectory = linalg::Vector<double, 2 * (N + 1)>;
  System system;
  linalg::Matrix<double, 2 * (N + 1), Dim> E{};
  linalg::Matrix<double, Dim, Dim> H{};
  linalg::Matrix<double, 2 * (N + 1), Dim> G{};
  double max_velocity;

  constexpr Window(const System& model, const Config& c)
      : system(model), max_velocity(c.max_velocity) {
    E(0, 0) = E(1, 1) = 1.0;
    for (std::size_t i = 1; i <= N; ++i) {
      for (std::size_t r = 0; r < 2; ++r) {
        E(2 * i + r, i + 1) = system.B(r, 0);
        for (std::size_t j = 0; j < Dim; ++j)
          for (std::size_t k = 0; k < 2; ++k)
            E(2 * i + r, j) += system.A(r, k) * E(2 * (i - 1) + k, j);
      }
    }
    // Multiply the weighted least-squares objective by measurement_std^2: unit
    // weight on measurement errors, relative weights on the prior and
    // disturbances.
    const double variance = c.measurement_std * c.measurement_std;
    H(0, 0) = 2 * variance / (c.arrival_position_std * c.arrival_position_std);
    H(1, 1) = 2 * variance / (c.arrival_velocity_std * c.arrival_velocity_std);
    for (std::size_t j = 2; j < Dim; ++j)
      H(j, j) = 2 * variance / (c.acceleration_std * c.acceleration_std);
    for (std::size_t i = 0; i <= N; ++i) {
      for (std::size_t j = 0; j < Dim; ++j) {
        G(i, j) = E(2 * i + 1, j);
        G(N + 1 + i, j) = -E(2 * i + 1, j);
        for (std::size_t k = 0; k < Dim; ++k)
          H(j, k) += 2 * E(2 * i, j) * E(2 * i, k);
      }
    }
  }

  constexpr Trajectory drift(const State& prior, const Inputs& inputs) const {
    Trajectory f{};
    State x = prior;
    for (std::size_t i = 0; i <= N; ++i) {
      f(2 * i) = x(0);
      f(2 * i + 1) = x(1);
      if (i < N)
        x = system.state_update(x, linalg::Vector<double, 1>{inputs[i]});
    }
    return f;
  }

  auto problem(const State& prior, const Measurements& measurements,
               const Inputs& inputs) const {
    const auto f = drift(prior, inputs);
    Decision q{};
    linalg::Vector<double, 2 * (N + 1)> b{};
    for (std::size_t i = 0; i <= N; ++i) {
      if (!std::isfinite(measurements[i]) || !std::isfinite(f(2 * i)) ||
          !std::isfinite(f(2 * i + 1)))
        throw std::invalid_argument("MHE window data must be finite");
      for (std::size_t j = 0; j < Dim; ++j)
        q(j) += 2 * E(2 * i, j) * (f(2 * i) - measurements[i]);
      b(i) = max_velocity - f(2 * i + 1);
      b(N + 1 + i) = max_velocity + f(2 * i + 1);
    }
    constexpr auto z =
        convex::Variable<convex::VarMeta<linalg::RowVector<double, Dim>>>();
    return convex::make_problem(z)
        .in_set(convex::Polyhedron(G, b, z))
        .minimize(convex::make_quad_form(H, q, z));
  }

  template <typename Iterate, typename Problem>
  auto warm_start(const Iterate& previous, const State& shifted_correction,
                  const Problem& problem) const {
    static_assert(linalg::vector_dim_v<decltype(previous.z)> == Dim + 2);
    Iterate seed{};
    seed.z(0) = shifted_correction(0);
    seed.z(1) = shifted_correction(1);
    for (std::size_t i = 0; i + 1 < N; ++i) seed.z(i + 2) = previous.z(i + 3);
    // Append zero disturbance. Rebuild native QuadForm's norm/square lifts.
    Decision decision{};
    for (std::size_t i = 0; i < Dim; ++i) decision(i) = seed.z(i);
    const auto weighted = H * decision;
    double quadratic = 0;
    for (std::size_t i = 0; i < Dim; ++i)
      quadratic += decision(i) * weighted(i);
    seed.z(Dim) = std::sqrt(std::max(0.0, 0.5 * quadratic)) + 0.01;
    seed.z(Dim + 1) = seed.z(Dim) * seed.z(Dim) + 0.01;
    seed.x = problem.constraint_map()(seed.z);
    std::size_t offset = 0;
    std::apply(
        [&](const auto&... cones) {
          auto recenter = [&](const auto& cone) {
            const auto x0 = cone.x0();
            auto point = x0;
            for (std::size_t j = 0; j < linalg::vector_dim_v<decltype(x0)>; ++j)
              point(j) = seed.x(offset + j) + 0.01 * x0(j);
            if (!cone.in_cone(point)) point = x0;
            const auto [gradient, event] = cone.grad(point);
            if (linalg::is_fatal_event(event))
              throw std::runtime_error("Invalid MHE warm seed");
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
};
} // namespace example::mhe
