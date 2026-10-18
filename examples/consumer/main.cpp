/*
 * Copyright © 2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#include <akreon/control/control.hpp>
#include <akreon/convex/convex.hpp>

#include <cmath>
#include <iostream>

int main() {
  // Minimize the H2 norm from disturbance w to z = [x0, x1, u]
  // for xdot = diag(-1, -2)*x + w + [0, 1]*u, with u = K*x.
  using System = control::StateSpace<
      control::TimeDomainType::CONTINUOUS, linalg::Matrix<double, 2, 2>,
      linalg::Matrix<double, 2, 3>, linalg::Matrix<double, 3, 2>>;
  using Inputs =
      control::PortLayout<control::Port<control::PortRole::EXOGENOUS, 0, 2>,
                          control::Port<control::PortRole::CONTROL, 2, 1>>;
  using Outputs =
      control::PortLayout<control::Port<control::PortRole::PERFORMANCE, 0, 3>>;
  constexpr auto A =
      linalg::Matrix<double, 2, 2>{linalg::Vector<double, 2>{-1.0, 0.0},
                                   linalg::Vector<double, 2>{0.0, -2.0}};
  constexpr auto B =
      linalg::Matrix<double, 2, 3>{linalg::Vector<double, 3>{1.0, 0.0, 0.0},
                                   linalg::Vector<double, 3>{0.0, 1.0, 1.0}};
  constexpr auto C = linalg::Matrix<double, 3, 2>{
      linalg::Vector<double, 2>{1.0, 0.0}, linalg::Vector<double, 2>{0.0, 1.0},
      linalg::Vector<double, 2>{0.0, 0.0}};
  constexpr auto D =
      linalg::Matrix<double, 3, 3>{linalg::Vector<double, 3>{0.0, 0.0, 0.0},
                                   linalg::Vector<double, 3>{0.0, 0.0, 0.0},
                                   linalg::Vector<double, 3>{0.0, 0.0, 1.0}};
  constexpr auto plant =
      control::PlantModel<System, Inputs, Outputs>(System(A, B, C, D));

  // Assemble the SDP in the lifted variables [svec(Q), vec(Y), svec(Z)].
  using H2 = control::H2StateFeedback<decltype(plant)>;
  using Space = linalg::RowVector<double, H2::DecVarDim>;
  constexpr auto decision = convex::Variable<convex::VarMeta<Space>>();
  const auto problem = control::make_h2_state_feedback_problem(plant, decision);
  auto solver_config = config::PrimalDualIPMConfig<double>{};
  solver_config.steps = 64;
  solver_config.duality_gap_tol = 1e-6;
  const convex::PrimalDualIPM solver(problem, solver_config);
  const auto& result = solver.res;
  if (result.tag != config::SolverResult::OPTIMAL || result.stalled ||
      linalg::is_fatal_event(result.event)) {
    std::cerr << "H2 synthesis failed: status=" << static_cast<int>(result.tag)
              << ", event=" << static_cast<int>(result.event) << '\n';
    return 1;
  }

  // Recover K = Y*Q^{-1} from the solved decision vector.
  const auto [q, yz] = linalg::splice<H2::QDim>(result.iterate.z);
  const auto Q = linalg::make_smat<2>(q);
  const auto Y =
      linalg::RowVector<double, 2>{linalg::Vector<double, 2>{yz(0), yz(1)}};
  const auto [K, event] = H2{}.K(Q, Y);
  // The analytic optimum is K = [0, 2 - sqrt(5)], with poles -1, -sqrt(5).
  const double pole = -2.0 + K(0, 1);
  const double expected_cost = 0.5 + std::sqrt(5.0) - 2.0;
  if (event != linalg::NumericalEvent::NONE || !std::isfinite(K(0, 0)) ||
      !std::isfinite(K(0, 1)) || !std::isfinite(result.convex_obj) ||
      pole >= 0.0 || std::abs(K(0, 0)) > 1e-2 ||
      std::abs(K(0, 1) - (2.0 - std::sqrt(5.0))) > 1e-2 ||
      std::abs(result.convex_obj - expected_cost) > 1e-3) {
    std::cerr << "Unexpected H2 solution\n";
    return 1;
  }
  std::cout << "H2 state feedback: K=[" << K(0, 0) << ", " << K(0, 1)
            << "], closed-loop poles=[-1, " << pole
            << "], squared H2 bound=" << result.convex_obj << '\n';
}
