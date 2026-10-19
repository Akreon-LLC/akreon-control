/*
 * Copyright © 2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */
#include <catch2/catch_test_macros.hpp>

#include <akreon/control/model/discretization.hpp>

#include <limits>

using namespace control;
using namespace linalg;

TEST_CASE("ZOH discretizes the double integrator at compile time",
          "[discretization]") {
  using System = StateSpace<TimeDomainType::CONTINUOUS, Matrixd<2, 2>,
                            Matrixd<2, 1>, Matrixd<2, 2>>;
  constexpr auto continuous =
      System(Matrixd<2, 2>{Vectord<2>{0.0, 1.0}, Vectord<2>{0.0, 0.0}},
             Matrixd<2, 1>{Vectord<1>{0.0}, Vectord<1>{1.0}},
             make_identity<double, 2>());
  constexpr auto discrete = c2d_zoh(continuous, 0.2);
  STATIC_REQUIRE(DiscreteStateSpaceLike<decltype(discrete)>);
  STATIC_REQUIRE(all_equal(
      discrete.A, Matrixd<2, 2>{Vectord<2>{1.0, 0.2}, Vectord<2>{0.0, 1.0}},
      1e-8));
  STATIC_REQUIRE(all_equal(
      discrete.B, Matrixd<2, 1>{Vectord<1>{0.02}, Vectord<1>{0.2}}, 1e-8));
  STATIC_REQUIRE(all_equal(discrete.C, continuous.C));
  STATIC_REQUIRE(!discrete.D.has_value());
  const auto next =
      discrete.state_update(Vectord<2>{1.0, 2.0}, Vectord<1>{3.0});
  REQUIRE(all_equal(next, Vectord<2>{1.46, 2.6}, 1e-8));
}

TEST_CASE("ZOH preserves output and feedthrough for a stable scalar system",
          "[discretization]") {
  using System = StateSpace<TimeDomainType::CONTINUOUS, Matrixd<1, 1>,
                            Matrixd<1, 1>, Matrixd<1, 1>>;
  constexpr auto continuous =
      System(Matrixd<1, 1>{Vectord<1>{-2.0}}, Matrixd<1, 1>{Vectord<1>{3.0}},
             Matrixd<1, 1>{Vectord<1>{4.0}}, Matrixd<1, 1>{Vectord<1>{5.0}});
  constexpr auto discrete = c2d_zoh(continuous, 0.25);
  constexpr double decay = constexpr_exp(-0.5);
  STATIC_REQUIRE(constexpr_abs(discrete.A(0, 0) - decay) < 1e-8);
  STATIC_REQUIRE(constexpr_abs(discrete.B(0, 0) - 1.5 * (1.0 - decay)) < 1e-8);
  STATIC_REQUIRE(all_equal(discrete.C, continuous.C));
  STATIC_REQUIRE(discrete.D.has_value());
  STATIC_REQUIRE(all_equal(*discrete.D, *continuous.D));
}

TEST_CASE("ZOH handles multiple held inputs with singular A",
          "[discretization]") {
  using System = StateSpace<TimeDomainType::CONTINUOUS, Matrixd<1, 1>,
                            Matrixd<1, 2>, Matrixd<1, 1>>;
  const auto continuous =
      System(Matrixd<1, 1>{}, Matrixd<1, 2>{Vectord<2>{2.0, -3.0}},
             make_identity<double, 1>());
  const auto discrete = c2d_zoh(continuous, 0.1);
  REQUIRE(std::abs(discrete.A(0, 0) - 1.0) < 1e-8);
  REQUIRE(all_equal(discrete.B, Matrixd<1, 2>{Vectord<2>{0.2, -0.3}}, 1e-8));
}

TEST_CASE("ZOH rejects invalid periods and nonfinite models",
          "[discretization]") {
  using System = StateSpace<TimeDomainType::CONTINUOUS, Matrixd<1, 1>,
                            Matrixd<1, 1>, Matrixd<1, 1>>;
  auto continuous = System(Matrixd<1, 1>{}, make_identity<double, 1>(),
                           make_identity<double, 1>());
  for (double dt : {0.0, -0.1, std::numeric_limits<double>::infinity(),
                    std::numeric_limits<double>::quiet_NaN()})
    REQUIRE_THROWS_AS(c2d_zoh(continuous, dt), std::invalid_argument);
  continuous.B(0, 0) = std::numeric_limits<double>::quiet_NaN();
  REQUIRE_THROWS_AS(c2d_zoh(continuous, 0.1), std::invalid_argument);
  continuous.B(0, 0) = std::numeric_limits<double>::max();
  REQUIRE_THROWS_AS(c2d_zoh(continuous, 2.0), std::invalid_argument);
}

TEST_CASE("ZOH returns real matrices for oscillatory dynamics",
          "[discretization]") {
  using System = StateSpace<TimeDomainType::CONTINUOUS, Matrixd<2, 2>,
                            Matrixd<2, 1>, Matrixd<2, 2>>;
  const auto continuous =
      System(Matrixd<2, 2>{Vectord<2>{0.0, -1.0}, Vectord<2>{1.0, 0.0}},
             Matrixd<2, 1>{Vectord<1>{1.0}, Vectord<1>{0.0}},
             make_identity<double, 2>());
  const auto discrete = c2d_zoh(continuous, 0.3);
  const double c = std::cos(0.3), s = std::sin(0.3);
  REQUIRE(all_equal(discrete.A,
                    Matrixd<2, 2>{Vectord<2>{c, -s}, Vectord<2>{s, c}}, 1e-8));
  REQUIRE(all_equal(discrete.B,
                    Matrixd<2, 1>{Vectord<1>{s}, Vectord<1>{1.0 - c}}, 1e-8));
}

TEST_CASE("ZOH supports single precision", "[discretization]") {
  using Mat = Matrix<float, 1, 1>;
  const auto continuous = StateSpace<TimeDomainType::CONTINUOUS, Mat, Mat, Mat>(
      Mat{Vector<float, 1>{-1.0f}}, make_identity<float, 1>(),
      make_identity<float, 1>());
  const auto discrete = c2d_zoh(continuous, 0.1f);
  REQUIRE(std::abs(discrete.A(0, 0) - std::exp(-0.1f)) < 1e-4f);
  REQUIRE(std::abs(discrete.B(0, 0) - (1.0f - std::exp(-0.1f))) < 1e-4f);
}
