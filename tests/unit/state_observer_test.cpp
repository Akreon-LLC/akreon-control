/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#include <catch2/catch_test_macros.hpp>

#include "akreon/control/control.hpp"
#include "akreon/linalg/linalg.hpp"
#include "akreon/convex/convex.hpp"
#include <akreon/convex/test_helpers.hpp>

using namespace linalg;
using namespace convex;
using namespace control;
using namespace unit_test;
using namespace config;

static constexpr auto rel_tol{1e-3};

template <typename T>
static constexpr auto state_observer_solver_config() {
  auto cfg = config::PrimalDualIPMConfig<T>{};
  cfg.steps = 96;
  cfg.duality_gap_tol = T(1e-3);
  return cfg;
}

TEST_CASE("Luenberger Observer 2x2", "[state_observer_test]") {
  using Space = Matrixd<2, 2>;
  using PlantSS = StateSpace<TimeDomainType::CONTINUOUS, Space, Space, Space>;
  using Inputs = PortLayout<Port<PortRole::CONTROL, 0, 2>>;
  using Outputs = PortLayout<Port<PortRole::MEASUREMENT, 0, 2>>;
  using Plant = PlantModel<PlantSS, Inputs, Outputs>;
  using Observer = LuenbergerObserver<Plant>;
  using DecVarSpace = RowVectord<Observer::ZDim>;
  constexpr auto dec = Variable<VarMeta<DecVarSpace>>();
  constexpr auto I = make_identity<double, 2>();
  constexpr auto Z = Matrixd<2, 2>{};
  constexpr auto MatBuilder = mat_traits<Space>();

  SECTION("Hand-checkable canonical LMI") {
    constexpr auto A = Z;
    constexpr auto B = I;
    constexpr auto C = I;
    constexpr auto plant = Plant(PlantSS(A, B, C));

    constexpr auto obs = LuenbergerObserver<decltype(plant)>();
    const auto lmi = obs(plant, dec);

    constexpr auto builder = conic::create_empty_cblk_builder<DecVarSpace>();
    const auto cone_block =
        lmi.template canonicalize<AtomType::SET>(builder).cb;

    constexpr auto s2 = constexpr_sqrt_safe(2.0);
    constexpr auto F_exp =
        Matrixd<6, 7>{Vectord<7>{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                      Vectord<7>{0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                      Vectord<7>{0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
                      Vectord<7>{0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0},
                      Vectord<7>{0.0, 0.0, 0.0, 0.0, s2, s2, 0.0},
                      Vectord<7>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0}};
    REQUIRE(all_equal(cone_block.map.linear_part(), F_exp));
    REQUIRE(cone_block.map.translation() ==
            Vectord<6>{-1e-6, 0.0, -1e-6, 0.0, 0.0, 0.0});

    const auto cones = cone_block.cones;
    constexpr auto sz = std::tuple_size<decltype(cones)>::value;
    STATIC_REQUIRE(sz == 2);
    REQUIRE(cone_block.validate());
  }

  SECTION("Solver finds feasible Q and stabilizing observer gain") {
    constexpr auto A = MatBuilder(0.2, 0.0, 0.0, 0.1);
    constexpr auto B = I;
    constexpr auto C = I;
    constexpr auto plant = Plant(PlantSS(A, B, C));

    constexpr auto obs = LuenbergerObserver<decltype(plant)>();
    constexpr auto P = make_identity<double, Observer::ZDim>();
    constexpr auto q = Vectord<Observer::ZDim>{};
    constexpr auto obj = make_quad_form(P, q, dec);
    constexpr auto prob = make_observer_problem(plant, dec, obj);

    STATIC_REQUIRE(is_problem_set_v<decltype(prob)>);
    STATIC_REQUIRE(prob.validate());
    STATIC_REQUIRE(std::tuple_size<decltype(prob.constraint_cones())>::value ==
                   4);

    const PrimalDualIPM solver(prob);
    const auto& res = solver.res;
#ifdef AKREON_REPORT_OUTPUT
    if (res.tag != SolverResult::OPTIMAL || res.stalled) log(res);
#endif

    REQUIRE(res.tag == SolverResult::OPTIMAL);
    REQUIRE(!res.stalled);

    const auto z_split = linalg::splice<Observer::QDim>(res.iterate.z);
    const auto Q_res = make_smat<2>(std::get<0>(z_split));
    Matrixd<2, 2> W_res{};
    for (std::size_t i = 0; i < 2; ++i)
      for (std::size_t j = 0; j < 2; ++j)
        W_res(i, j) = std::get<1>(z_split)(i * 2 + j);

    REQUIRE(all_equal(Q_res, transpose(Q_res), rel_tol));
    REQUIRE(Q_res(0, 0) > 1e-6);
    REQUIRE(Q_res(1, 1) > 1e-6);

    const auto [L_res, L_event] = obs.L(Q_res, W_res);
    REQUIRE(L_event == linalg::NumericalEvent::NONE);
    REQUIRE(all_equal(W_res, Q_res * L_res, 1e-4));

    const auto error_dynamics = A - L_res * C;
    REQUIRE(error_dynamics(0, 0) < 0.0);
    REQUIRE(error_dynamics(1, 1) < 0.0);

    const auto cert = (-1.0) * (transpose(A) * Q_res + Q_res * A - W_res * C -
                                transpose(C) * transpose(W_res));
    const auto cert_sym = linalg::SymmetricMatrix(cert);
    REQUIRE(cert_sym.is_valid());
    REQUIRE(cert_sym.is_psd());

    const auto lyap =
        (-1.0) * (transpose(error_dynamics) * Q_res + Q_res * error_dynamics);
    const auto lyap_sym = linalg::SymmetricMatrix(lyap);
    REQUIRE(lyap_sym.is_valid());
    REQUIRE(lyap_sym.is_psd());
  }

  SECTION("Explicit feasible point satisfies canonical constraints") {
    constexpr auto A = MatBuilder(0.2, 0.0, 0.0, 0.1);
    constexpr auto B = I;
    constexpr auto C = I;
    constexpr auto plant = Plant(PlantSS(A, B, C));

    constexpr auto prob = make_observer_problem(plant, dec);

    STATIC_REQUIRE(is_problem_set_v<decltype(prob)>);
    STATIC_REQUIRE(prob.validate());

    const auto Q_feas = I;
    const auto W_feas = I;
    Vectord<Observer::ZDim> z_feas{};
    const auto q_feas = get_svec(Q_feas);
    for (std::size_t i = 0; i < Observer::QDim; ++i) z_feas(i) = q_feas(i);
    for (std::size_t i = 0; i < 2; ++i)
      for (std::size_t j = 0; j < 2; ++j)
        z_feas(Observer::QDim + i * 2 + j) = W_feas(i, j);
    const auto x_feas = prob.constraint_map().linear_part() * z_feas +
                        prob.constraint_map().translation();

    const auto Q_slack =
        make_smat<2>(Vectord<3>{x_feas(0), x_feas(1), x_feas(2)});
    REQUIRE(Q_slack(0, 0) > 0.0);
    REQUIRE(Q_slack(1, 1) > 0.0);

    const auto cert_slack =
        make_smat<2>(Vectord<3>{x_feas(3), x_feas(4), x_feas(5)});
    const auto cert_sym = SymmetricMatrix(cert_slack, 1e-6);
    REQUIRE(cert_sym.is_valid());
    REQUIRE(cert_sym.is_psd());
    REQUIRE(all_equal(W_feas, Q_feas, rel_tol));
  }
}

TEST_CASE("Luenberger Observer 3x3", "[state_observer_test]") {
  using Space = Matrixd<3, 3>;
  using PlantSS = StateSpace<TimeDomainType::CONTINUOUS, Space, Space, Space>;
  using Inputs = PortLayout<Port<PortRole::CONTROL, 0, 3>>;
  using Outputs = PortLayout<Port<PortRole::MEASUREMENT, 0, 3>>;
  using Plant = PlantModel<PlantSS, Inputs, Outputs>;
  using Observer = LuenbergerObserver<Plant>;
  using DecVarSpace = RowVectord<Observer::ZDim>;
  constexpr auto dec = Variable<VarMeta<DecVarSpace>>();
  constexpr auto I = make_identity<double, 3>();
  constexpr auto MatBuilder = mat_traits<Space>();

  SECTION("Solver finds feasible Q on coupled observable plant") {
    constexpr auto A =
        MatBuilder(0.2, 0.05, 0.0, 0.0, 0.1, 0.04, 0.0, 0.0, 0.15);
    constexpr auto B = I;
    constexpr auto C = I;
    constexpr auto plant = Plant(PlantSS(A, B, C));

    constexpr auto obs = LuenbergerObserver<decltype(plant)>();
    constexpr auto P = make_identity<double, Observer::ZDim>();
    constexpr auto q = Vectord<Observer::ZDim>{};
    constexpr auto obj = make_quad_form(P, q, dec);
    constexpr auto prob = make_observer_problem(plant, dec, obj);

    STATIC_REQUIRE(is_problem_set_v<decltype(prob)>);
    STATIC_REQUIRE(prob.validate());
    STATIC_REQUIRE(std::tuple_size<decltype(prob.constraint_cones())>::value ==
                   4);

    const PrimalDualIPM solver(prob);
    const auto& res = solver.res;
#ifdef AKREON_REPORT_OUTPUT
    if (res.tag != SolverResult::OPTIMAL || res.stalled) log(res);
#endif

    REQUIRE(res.tag == SolverResult::OPTIMAL);
    REQUIRE(!res.stalled);

    const auto z_split = linalg::splice<Observer::QDim>(res.iterate.z);
    const auto Q_res = make_smat<3>(std::get<0>(z_split));
    Matrixd<3, 3> W_res{};
    for (std::size_t i = 0; i < 3; ++i)
      for (std::size_t j = 0; j < 3; ++j)
        W_res(i, j) = std::get<1>(z_split)(i * 3 + j);

    REQUIRE(all_equal(Q_res, transpose(Q_res), rel_tol));
    REQUIRE(Q_res(0, 0) > 1e-6);
    REQUIRE(Q_res(1, 1) > 1e-6);
    REQUIRE(Q_res(2, 2) > 1e-6);

    const auto cert = (-1.0) * (transpose(A) * Q_res + Q_res * A - W_res * C -
                                transpose(C) * transpose(W_res));
    const auto cert_sym = linalg::SymmetricMatrix(cert);
    REQUIRE(cert_sym.is_valid());
    REQUIRE(cert_sym.is_psd());

    const auto [L_res, L_event] = obs.L(Q_res, W_res);
    REQUIRE(L_event == linalg::NumericalEvent::NONE);
    REQUIRE(all_equal(W_res, Q_res * L_res, 1e-4));

    const auto error_dynamics = A - L_res * C;
    const auto lyap =
        (-1.0) * (transpose(error_dynamics) * Q_res + Q_res * error_dynamics);
    const auto lyap_sym = linalg::SymmetricMatrix(lyap, 1e-6);
    REQUIRE(lyap_sym.is_valid());
    REQUIRE(lyap_sym.is_psd());
  }

  SECTION("Explicit feasible 3x3 point satisfies canonical constraints") {
    constexpr auto A =
        MatBuilder(0.2, 0.05, 0.0, 0.0, 0.1, 0.04, 0.0, 0.0, 0.15);
    constexpr auto B = I;
    constexpr auto C = I;
    constexpr auto plant = Plant(PlantSS(A, B, C));

    constexpr auto prob = make_observer_problem(plant, dec);

    STATIC_REQUIRE(is_problem_set_v<decltype(prob)>);
    STATIC_REQUIRE(prob.validate());

    const auto Q_feas = I;
    const auto W_feas = I;
    Vectord<Observer::ZDim> z_feas{};
    const auto q_feas = get_svec(Q_feas);
    for (std::size_t i = 0; i < Observer::QDim; ++i) z_feas(i) = q_feas(i);
    for (std::size_t i = 0; i < 3; ++i)
      for (std::size_t j = 0; j < 3; ++j)
        z_feas(Observer::QDim + i * 3 + j) = W_feas(i, j);
    const auto x_feas = prob.constraint_map().linear_part() * z_feas +
                        prob.constraint_map().translation();

    const auto Q_slack = make_smat<3>(Vectord<6>{
        x_feas(0), x_feas(1), x_feas(2), x_feas(3), x_feas(4), x_feas(5)});
    REQUIRE(Q_slack(0, 0) > 0.0);
    REQUIRE(Q_slack(1, 1) > 0.0);
    REQUIRE(Q_slack(2, 2) > 0.0);

    const auto cert_slack = make_smat<3>(Vectord<6>{
        x_feas(6), x_feas(7), x_feas(8), x_feas(9), x_feas(10), x_feas(11)});
    const auto cert_sym = SymmetricMatrix(cert_slack, 1e-6);
    REQUIRE(cert_sym.is_valid());
    REQUIRE(cert_sym.is_psd());
  }
}

TEST_CASE("H2 State Observer synthesis 2x2", "[state_observer_test]") {
  using StateSpaceMat = Matrixd<2, 2>;
  using InputMat = Matrixd<2, 2>;
  using OutputMat = Matrixd<2, 2>;
  using SS = StateSpace<TimeDomainType::CONTINUOUS, StateSpaceMat, InputMat,
                        OutputMat>;
  using Inputs = PortLayout<Port<PortRole::EXOGENOUS, 0, 1>,
                            Port<PortRole::CONTROL, 1, 1>>;
  using Outputs = PortLayout<Port<PortRole::MEASUREMENT, 0, 1>,
                             Port<PortRole::PERFORMANCE, 1, 1>>;
  using Plant = PlantModel<SS, Inputs, Outputs>;

  constexpr auto MatBuilder = mat_traits<StateSpaceMat>();
  constexpr auto A = MatBuilder(-1.0, 0.0, 0.0, -2.0);
  constexpr auto B = InputMat{Vectord<2>{1.0, 0.0}, Vectord<2>{0.0, 1.0}};
  constexpr auto C = OutputMat{Vectord<2>{1.0, 0.0}, Vectord<2>{1.0, 0.0}};
  constexpr auto D = Matrixd<2, 2>{Vectord<2>{0.0, 0.0}, Vectord<2>{0.5, 0.0}};
  constexpr auto plant = Plant(SS(A, B, C, D));

  using H2 = H2StateObserver<decltype(plant)>;
  using DecVarSpace = RowVectord<H2::DecVarDim>;
  constexpr auto Z_hat = Variable<VarMeta<DecVarSpace>>();
  const auto prob = make_h2_state_observer_problem(plant, Z_hat);

  REQUIRE(is_problem_set_v<decltype(prob)>);
  REQUIRE(prob.validate());
  REQUIRE(std::tuple_size<decltype(prob.constraint_cones())>::value == 2);

  const auto c = prob.linear_objective();
  const auto c_exp = Vectord<H2::DecVarDim>{0.0, 0.0, 0.0, 0.0, 0.0, 1.0};
  REQUIRE(c == c_exp);

  using Solver = PrimalDualIPM<decltype(prob)>;
  const auto cfg = state_observer_solver_config<double>();
  const Solver solver(prob, cfg);
  const auto& res = solver.res;
#ifdef AKREON_REPORT_OUTPUT
  if (res.tag != SolverResult::OPTIMAL || res.stalled) log(res);
#endif

  REQUIRE(res.tag == SolverResult::OPTIMAL);
  REQUIRE(!res.stalled);

  const auto q_wz_split = linalg::splice<H2::QDim>(res.iterate.z);
  const auto Q_res = make_smat<2>(std::get<0>(q_wz_split));
  const auto w_z_split = linalg::splice<H2::WDim>(std::get<1>(q_wz_split));
  ColVectord<2> W_res{};
  for (std::size_t i = 0; i < H2::WDim; ++i)
    W_res(i, 0) = std::get<0>(w_z_split)(i);
  const auto Z_res = make_smat<1>(std::get<1>(w_z_split));

  REQUIRE(all_equal(Q_res, transpose(Q_res), rel_tol));
  REQUIRE(Q_res(0, 0) > 1e-6);
  REQUIRE(Q_res(1, 1) > 1e-6);
  REQUIRE(Z_res(0, 0) > 1e-6);

  const auto Q_sym = SymmetricMatrix(Q_res, 1e-6);
  REQUIRE(Q_sym.is_valid());
  REQUIRE(Q_sym.is_psd());

  constexpr auto h2 = H2();
  const auto [L_res, L_event] = h2.L(Q_res, W_res);
  REQUIRE(L_event == linalg::NumericalEvent::NONE);
  REQUIRE(all_equal(W_res, Q_res * L_res, 1e-4));

  constexpr auto B_w = ColVectord<2>{Vectord<1>{1.0}, Vectord<1>{0.0}};
  constexpr auto C_y = RowVectord<2>{Vectord<2>{1.0, 0.0}};
  constexpr auto C_e = RowVectord<2>{Vectord<2>{1.0, 0.0}};
  constexpr auto D_yw = RowVectord<1>{Vectord<1>{0.0}};
  const auto stability_cert = (-1.0) * (transpose(A) * Q_res + Q_res * A -
                                        transpose(C_y) * transpose(W_res) -
                                        W_res * C_y + transpose(C_e) * C_e);
  const auto stability_sym = SymmetricMatrix(stability_cert, 1e-6);
  REQUIRE(stability_sym.is_valid());
  REQUIRE(stability_sym.is_psd());

  const auto bottom_left = Q_res * B_w - W_res * D_yw;
  const auto epigraph = Matrixd<3, 3>{
      Vectord<3>{Z_res(0, 0), bottom_left(0, 0), bottom_left(1, 0)},
      Vectord<3>{bottom_left(0, 0), Q_res(0, 0), Q_res(0, 1)},
      Vectord<3>{bottom_left(1, 0), Q_res(1, 0), Q_res(1, 1)}};
  const auto epigraph_sym = SymmetricMatrix(epigraph, 1e-6);
  REQUIRE(epigraph_sym.is_valid());
  REQUIRE(epigraph_sym.is_psd());

  const auto error_dynamics = A - L_res * C_y;
  const auto lyap = (-1.0) * (transpose(error_dynamics) * Q_res +
                              Q_res * error_dynamics + transpose(C_e) * C_e);
  const auto lyap_sym = SymmetricMatrix(lyap, 1e-6);
  REQUIRE(lyap_sym.is_valid());
  REQUIRE(lyap_sym.is_psd());
}

TEST_CASE("Hinf State Observer synthesis 2x2", "[state_observer_test]") {
  using StateSpaceMat = Matrixd<2, 2>;
  using InputMat = Matrixd<2, 2>;
  using OutputMat = Matrixd<2, 2>;
  using SS = StateSpace<TimeDomainType::CONTINUOUS, StateSpaceMat, InputMat,
                        OutputMat>;
  using Inputs = PortLayout<Port<PortRole::EXOGENOUS, 0, 1>,
                            Port<PortRole::CONTROL, 1, 1>>;
  using Outputs = PortLayout<Port<PortRole::MEASUREMENT, 0, 1>,
                             Port<PortRole::PERFORMANCE, 1, 1>>;
  using Plant = PlantModel<SS, Inputs, Outputs>;

  constexpr auto MatBuilder = mat_traits<StateSpaceMat>();
  constexpr auto A = MatBuilder(-1.0, 0.0, 1.0, -2.0);
  constexpr auto B = InputMat{Vectord<2>{1.0, 0.0}, Vectord<2>{0.0, 1.0}};
  constexpr auto C = OutputMat{Vectord<2>{1.0, 0.0}, Vectord<2>{0.0, 1.0}};
  constexpr auto D = Matrixd<2, 2>{Vectord<2>{0.0, 0.0}, Vectord<2>{0.5, 0.0}};
  constexpr auto plant = Plant(SS(A, B, C, D));

  using Hinf = HinfStateObserver<decltype(plant)>;
  using DecVarSpace = RowVectord<Hinf::DecVarDim>;
  constexpr auto Z_hat = Variable<VarMeta<DecVarSpace>>();
  const auto prob = make_hinf_state_observer_problem(plant, Z_hat);

  REQUIRE(is_problem_set_v<decltype(prob)>);
  REQUIRE(prob.validate());
  REQUIRE(std::tuple_size<decltype(prob.constraint_cones())>::value == 2);

  const auto c = prob.linear_objective();
  const auto c_exp = Vectord<Hinf::DecVarDim>{0.0, 0.0, 0.0, 0.0, 0.0, 1.0};
  REQUIRE(c == c_exp);

  const auto Q_res =
      Matrixd<2, 2>{Vectord<2>{10.0, 0.0}, Vectord<2>{0.0, 10.0}};
  const auto W_res = ColVectord<2>{Vectord<1>{0.0}, Vectord<1>{0.0}};
  const auto gamma_res = 20.0;
  const auto z_feas =
      Vectord<Hinf::DecVarDim>{10.0, 0.0, 10.0, 0.0, 0.0, gamma_res};
  const auto x_feas = prob.constraint_map().linear_part() * z_feas +
                      prob.constraint_map().translation();
  const auto slack_split = linalg::splice<3>(x_feas);
  const auto Q_slack = make_smat<2>(std::get<0>(slack_split));
  const auto BRL_slack = make_smat<4>(std::get<1>(slack_split));

  REQUIRE(all_equal(Q_res, transpose(Q_res), rel_tol));
  REQUIRE(Q_res(0, 0) > 1e-6);
  REQUIRE(Q_res(1, 1) > 1e-6);
  REQUIRE(gamma_res > 1e-6);

  const auto Q_sym = SymmetricMatrix(Q_res, 1e-6);
  REQUIRE(Q_sym.is_valid());
  REQUIRE(Q_sym.is_psd());

  const auto Q_slack_sym = SymmetricMatrix(Q_slack, 1e-6);
  REQUIRE(Q_slack_sym.is_valid());
  REQUIRE(Q_slack_sym.is_psd());

  constexpr auto hinf = Hinf();
  const auto [L_res, L_event] = hinf.L(Q_res, W_res);
  REQUIRE(L_event == linalg::NumericalEvent::NONE);
  REQUIRE(all_equal(W_res, Q_res * L_res, 1e-4));

  constexpr auto B_w = ColVectord<2>{Vectord<1>{1.0}, Vectord<1>{0.0}};
  constexpr auto C_y = RowVectord<2>{Vectord<2>{1.0, 0.0}};
  constexpr auto C_e = RowVectord<2>{Vectord<2>{0.0, 1.0}};
  constexpr auto D_yw = RowVectord<1>{Vectord<1>{0.0}};
  constexpr auto D_ew = RowVectord<1>{Vectord<1>{0.5}};

  const auto top_left = transpose(A) * Q_res + Q_res * A -
                        transpose(C_y) * transpose(W_res) - W_res * C_y;
  const auto top_right = Q_res * B_w - W_res * D_yw;

  Matrixd<4, 4> brl{};
  brl = set_block<0, 2, 0, 2>(brl, top_left);
  brl = set_block<0, 2, 2, 3>(brl, top_right);
  brl = set_block<2, 3, 0, 2>(brl, transpose(top_right));
  brl = set_block<0, 2, 3, 4>(brl, transpose(C_e));
  brl = set_block<3, 4, 0, 2>(brl, C_e);
  brl = set_block<2, 3, 2, 3>(brl, Matrixd<1, 1>{Vectord<1>{-gamma_res}});
  brl = set_block<2, 3, 3, 4>(brl, transpose(D_ew));
  brl = set_block<3, 4, 2, 3>(brl, D_ew);
  brl = set_block<3, 4, 3, 4>(brl, Matrixd<1, 1>{Vectord<1>{-gamma_res}});

  const auto brl_cert = (-1.0) * brl;
  const auto brl_sym = SymmetricMatrix(brl_cert, 1e-6);
  REQUIRE(brl_sym.is_valid());
  REQUIRE(brl_sym.is_psd());

  const auto brl_slack_sym = SymmetricMatrix(BRL_slack, 1e-6);
  REQUIRE(brl_slack_sym.is_valid());
  REQUIRE(brl_slack_sym.is_psd());

  const auto error_dynamics = A - L_res * C_y;
  const auto disturbance_dynamics = B_w - L_res * D_yw;
  REQUIRE(all_equal(top_left,
                    transpose(error_dynamics) * Q_res + Q_res * error_dynamics,
                    1e-4));
  REQUIRE(all_equal(top_right, Q_res * disturbance_dynamics, 1e-4));
}

TEST_CASE("Hinf State Observer synthesis with zero D_ew is numerically stable",
          "[state_observer_test]") {
  using StateSpaceMat = Matrixd<2, 2>;
  using InputMat = Matrixd<2, 2>;
  using OutputMat = Matrixd<2, 2>;
  using SS = StateSpace<TimeDomainType::CONTINUOUS, StateSpaceMat, InputMat,
                        OutputMat>;
  using Inputs = PortLayout<Port<PortRole::EXOGENOUS, 0, 1>,
                            Port<PortRole::CONTROL, 1, 1>>;
  using Outputs = PortLayout<Port<PortRole::MEASUREMENT, 0, 1>,
                             Port<PortRole::PERFORMANCE, 1, 1>>;
  using Plant = PlantModel<SS, Inputs, Outputs>;

  constexpr auto MatBuilder = mat_traits<StateSpaceMat>();
  constexpr auto A = MatBuilder(-1.0, 0.0, 1.0, -2.0);
  constexpr auto B = InputMat{Vectord<2>{1.0, 0.0}, Vectord<2>{0.0, 1.0}};
  constexpr auto C = OutputMat{Vectord<2>{1.0, 0.0}, Vectord<2>{0.0, 1.0}};
  constexpr auto D = Matrixd<2, 2>{Vectord<2>{0.25, 0.0}, Vectord<2>{0.0, 0.0}};
  constexpr auto plant = Plant(SS(A, B, C, D));

  using Hinf = HinfStateObserver<decltype(plant)>;
  using DecVarSpace = RowVectord<Hinf::DecVarDim>;
  constexpr auto Z_hat = Variable<VarMeta<DecVarSpace>>();
  const auto prob = make_hinf_state_observer_problem(plant, Z_hat);

  REQUIRE(is_problem_set_v<decltype(prob)>);
  REQUIRE(prob.validate());
  REQUIRE(std::tuple_size<decltype(prob.constraint_cones())>::value == 2);

  using Solver = PrimalDualIPM<decltype(prob)>;
  auto cfg = state_observer_solver_config<double>();
  cfg.duality_gap_tol = 1e-3;
  cfg.dual_feas_tol = 1e-4;
  const Solver solver(prob, cfg);
  const auto& res = solver.res;
#ifdef AKREON_REPORT_OUTPUT
  if (res.tag != SolverResult::OPTIMAL || res.stalled) log(res);
#endif

  REQUIRE(res.tag == SolverResult::OPTIMAL);
  REQUIRE(!res.stalled);

  const auto q_w_gamma_split = linalg::splice<Hinf::QDim>(res.iterate.z);
  const auto Q_res = make_smat<2>(std::get<0>(q_w_gamma_split));
  const auto w_gamma_split =
      linalg::splice<Hinf::WDim>(std::get<1>(q_w_gamma_split));
  ColVectord<2> W_res{};
  for (std::size_t i = 0; i < Hinf::WDim; ++i)
    W_res(i, 0) = std::get<0>(w_gamma_split)(i);
  const auto gamma_res = std::get<1>(w_gamma_split)(0);

  REQUIRE(all_equal(Q_res, transpose(Q_res), rel_tol));
  REQUIRE(Q_res(0, 0) > 1e-6);
  REQUIRE(Q_res(1, 1) > 1e-6);
  REQUIRE(gamma_res > 1e-6);

  const auto Q_sym = SymmetricMatrix(Q_res, 1e-6);
  REQUIRE(Q_sym.is_valid());
  REQUIRE(Q_sym.is_psd());

  constexpr auto hinf = Hinf();
  const auto [L_res, L_event] = hinf.L(Q_res, W_res);
  REQUIRE(L_event == linalg::NumericalEvent::NONE);
  REQUIRE(all_equal(W_res, Q_res * L_res, 1e-4));

  constexpr auto B_w = ColVectord<2>{Vectord<1>{1.0}, Vectord<1>{0.0}};
  constexpr auto C_y = RowVectord<2>{Vectord<2>{1.0, 0.0}};
  constexpr auto C_e = RowVectord<2>{Vectord<2>{0.0, 1.0}};
  constexpr auto D_yw = RowVectord<1>{Vectord<1>{0.25}};
  constexpr auto D_ew = RowVectord<1>{Vectord<1>{0.0}};

  const auto top_left = transpose(A) * Q_res + Q_res * A -
                        transpose(C_y) * transpose(W_res) - W_res * C_y;
  const auto top_right = Q_res * B_w - W_res * D_yw;

  Matrixd<4, 4> brl{};
  brl = set_block<0, 2, 0, 2>(brl, top_left);
  brl = set_block<0, 2, 2, 3>(brl, top_right);
  brl = set_block<2, 3, 0, 2>(brl, transpose(top_right));
  brl = set_block<0, 2, 3, 4>(brl, transpose(C_e));
  brl = set_block<3, 4, 0, 2>(brl, C_e);
  brl = set_block<2, 3, 2, 3>(brl, Matrixd<1, 1>{Vectord<1>{-gamma_res}});
  brl = set_block<2, 3, 3, 4>(brl, transpose(D_ew));
  brl = set_block<3, 4, 2, 3>(brl, D_ew);
  brl = set_block<3, 4, 3, 4>(brl, Matrixd<1, 1>{Vectord<1>{-gamma_res}});

  const auto brl_cert = (-1.0) * brl;
  const auto brl_sym = SymmetricMatrix(brl_cert, 1e-6);
  REQUIRE(brl_sym.is_valid());
  REQUIRE(brl_sym.is_psd());

  const auto error_dynamics = A - L_res * C_y;
  const auto disturbance_dynamics = B_w - L_res * D_yw;
  REQUIRE(all_equal(top_left,
                    transpose(error_dynamics) * Q_res + Q_res * error_dynamics,
                    1e-4));
  REQUIRE(all_equal(top_right, Q_res * disturbance_dynamics, 1e-4));
}
