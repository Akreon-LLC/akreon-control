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
static constexpr auto pldi_solver_config() {
  auto cfg = config::PrimalDualIPMConfig<T>{};
  cfg.steps = 64;
  cfg.duality_gap_tol = T(1e-4);
  return cfg;
}

TEST_CASE("Polytopic State Feedback 2x2", "[polytopic_state_feedback_test]") {
  using Space = Matrixd<2, 2>;
  using FullPlantSS =
      StateSpace<TimeDomainType::CONTINUOUS, Space, Space, Space>;
  using FullInputs = PortLayout<Port<PortRole::CONTROL, 0, 2>>;
  using FullOutputs = PortLayout<Port<PortRole::MEASUREMENT, 0, 2>>;
  using FullPlant = PlantModel<FullPlantSS, FullInputs, FullOutputs>;
  constexpr auto I = make_identity<double, 2>();
  constexpr auto Z = Matrixd<2, 2>{};
  constexpr auto MatBuilder = mat_traits<Space>();

  static constexpr std::size_t QDim = 3;
  static constexpr std::size_t YDim = 4;
  static constexpr std::size_t ZDim = QDim + YDim;
  using ZSpace = RowVectord<ZDim>;
  constexpr auto dec = Variable<VarMeta<ZSpace>>();

  SECTION("Hand-checkable merged canonical LMI") {
    constexpr auto A0 = Z;
    constexpr auto A1 = MatBuilder(-1.0, 0.0, 0.0, -2.0);
    constexpr auto B0 = I;
    constexpr auto B1 = I;
    constexpr auto C = I;

    constexpr auto plant0 = FullPlant(FullPlantSS(A0, B0, C));
    constexpr auto plant1 = FullPlant(FullPlantSS(A1, B1, C));
    constexpr auto poly = Vector<decltype(plant0), 2>{plant0, plant1};

    constexpr auto psf = PolytopicStateFeedback<decltype(plant0), 2>();
    const auto lmis = psf(poly, dec);

    constexpr auto builder = conic::create_empty_cblk_builder<ZSpace>();
    const auto canon = lmis.template canonicalize<AtomType::SET>(builder);
    const auto cone_block = canon.cb;

    constexpr auto s2 = constexpr_sqrt_safe(2.0);
    constexpr auto F_exp = Matrixd<9, 7>{
        Vectord<7>{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        Vectord<7>{0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        Vectord<7>{0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
        Vectord<7>{0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0},
        Vectord<7>{0.0, 0.0, 0.0, 0.0, -s2, -s2, 0.0},
        Vectord<7>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0},
        Vectord<7>{2.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0},
        Vectord<7>{0.0, 3.0, 0.0, 0.0, -s2, -s2, 0.0},
        Vectord<7>{0.0, 0.0, 4.0, 0.0, 0.0, 0.0, -2.0},
    };

    REQUIRE(all_equal(cone_block.map.linear_part(), F_exp));
    REQUIRE(cone_block.map.translation() ==
            Vectord<9>{-1e-6, 0.0, -1e-6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    REQUIRE(std::tuple_size<decltype(cone_block.cones)>::value == 3);
    REQUIRE(cone_block.validate());
  }

  SECTION("Solver finds feasible merged z with PSD Q block") {
    using InputSpace = Matrixd<2, 1>;
    using OutputSpace = RowVectord<2>;
    using SolverPlantSS =
        StateSpace<TimeDomainType::CONTINUOUS, Space, InputSpace, OutputSpace>;
    using SolverInputs = PortLayout<Port<PortRole::CONTROL, 0, 1>>;
    using SolverOutputs = PortLayout<Port<PortRole::MEASUREMENT, 0, 1>>;
    using SolverPlant = PlantModel<SolverPlantSS, SolverInputs, SolverOutputs>;
    constexpr auto BCol = ColVectord<2>{Vectord<1>{1.0}, Vectord<1>{0.5}};
    constexpr auto CRow = RowVectord<2>{Vectord<2>{1.0, 0.0}};
    static constexpr std::size_t SolverQDim = 3;
    static constexpr std::size_t SolverYDim = 2;
    static constexpr std::size_t SolverZDim = SolverQDim + SolverYDim;
    using SolverZSpace = RowVectord<SolverZDim>;
    constexpr auto solver_dec = Variable<VarMeta<SolverZSpace>>();

    constexpr auto A0 = MatBuilder(-1.0, 0.0, 0.0, -2.0);
    constexpr auto A1 = MatBuilder(-0.5, 0.0, 0.0, -1.0);
    constexpr auto B0 = BCol;
    constexpr auto B1 = BCol;
    constexpr auto C = CRow;

    constexpr auto plant0 = SolverPlant(SolverPlantSS(A0, B0, C));
    constexpr auto plant1 = SolverPlant(SolverPlantSS(A1, B1, C));
    constexpr auto poly = Vector<decltype(plant0), 2>{plant0, plant1};

    constexpr auto psf = PolytopicStateFeedback<decltype(plant0), 2>();
    constexpr auto p_w = Vectord<SolverQDim>{1.0, 0.0, 1.0};
    constexpr auto y_w = Vectord<SolverYDim>{0.0, 0.0};
    constexpr auto obj =
        make_polytopic_state_feedback_linear_objective<decltype(plant0), 2>(
            p_w, y_w, solver_dec);
    const auto prob =
        make_polytopic_state_feedback_problem<decltype(plant0), 2>(
            poly, solver_dec, obj);

    REQUIRE(is_problem_set_v<decltype(prob)>);
    REQUIRE(prob.validate());
    REQUIRE(std::tuple_size<decltype(prob.constraint_cones())>::value == 3);

    using Solver = PrimalDualIPM<decltype(prob)>;
    const auto cfg = pldi_solver_config<double>();
    const Solver solver(prob, cfg);
    const auto& res = solver.res;
#ifdef AKREON_REPORT_OUTPUT
    if (res.tag != SolverResult::OPTIMAL || res.stalled) log(res);
#endif

    REQUIRE(res.tag == SolverResult::OPTIMAL);
    REQUIRE(!res.stalled);

    Vectord<SolverQDim> q_vec{};
    for (std::size_t i = 0; i < SolverQDim; ++i) q_vec(i) = res.iterate.z(i);
    const auto Q_res = make_smat<2>(q_vec);

    RowVectord<2> Y_res{};
    for (std::size_t j = 0; j < 2; ++j)
      Y_res(0, j) = res.iterate.z(SolverQDim + j);

    REQUIRE(all_equal(Q_res, transpose(Q_res), rel_tol));
    REQUIRE(Q_res(0, 0) > 1e-6);
    REQUIRE(Q_res(1, 1) > 1e-6);

    const auto Q_sym = SymmetricMatrix(Q_res, 1e-6);
    REQUIRE(Q_sym.is_valid());
    REQUIRE(Q_sym.is_psd());

    const auto [K_res, K_event] = psf.K(Q_res, Y_res);
    REQUIRE(K_event == linalg::NumericalEvent::NONE);
    REQUIRE(all_equal(Y_res, K_res * Q_res, 1e-4));

    const auto cert0 = (-1.0) * (A0 * Q_res + Q_res * transpose(A0) +
                                 B0 * Y_res + transpose(Y_res) * transpose(B0));
    const auto cert1 = (-1.0) * (A1 * Q_res + Q_res * transpose(A1) +
                                 B1 * Y_res + transpose(Y_res) * transpose(B1));

    const auto cert0_sym = SymmetricMatrix(cert0, 1e-6);
    const auto cert1_sym = SymmetricMatrix(cert1, 1e-6);
    REQUIRE(cert0_sym.is_valid());
    REQUIRE(cert1_sym.is_valid());
    REQUIRE(cert0_sym.is_psd());
    REQUIRE(cert1_sym.is_psd());

    const auto cl0 = A0 + B0 * K_res;
    const auto cl1 = A1 + B1 * K_res;
    const auto lyap0 = (-1.0) * (cl0 * Q_res + Q_res * transpose(cl0));
    const auto lyap1 = (-1.0) * (cl1 * Q_res + Q_res * transpose(cl1));
    const auto lyap0_sym = SymmetricMatrix(lyap0, 1e-6);
    const auto lyap1_sym = SymmetricMatrix(lyap1, 1e-6);
    REQUIRE(lyap0_sym.is_valid());
    REQUIRE(lyap1_sym.is_valid());
    REQUIRE(lyap0_sym.is_psd());
    REQUIRE(lyap1_sym.is_psd());
  }

  SECTION("Quadratic objective helper solves lifted PLDI problem") {
    using InputSpace = Matrixd<2, 1>;
    using OutputSpace = RowVectord<2>;
    using SolverPlantSS =
        StateSpace<TimeDomainType::CONTINUOUS, Space, InputSpace, OutputSpace>;
    using SolverInputs = PortLayout<Port<PortRole::CONTROL, 0, 1>>;
    using SolverOutputs = PortLayout<Port<PortRole::MEASUREMENT, 0, 1>>;
    using SolverPlant = PlantModel<SolverPlantSS, SolverInputs, SolverOutputs>;
    constexpr auto BCol = ColVectord<2>{Vectord<1>{1.0}, Vectord<1>{0.5}};
    constexpr auto CRow = RowVectord<2>{Vectord<2>{1.0, 0.0}};
    static constexpr std::size_t SolverQDim = 3;
    static constexpr std::size_t SolverYDim = 2;
    static constexpr std::size_t SolverZDim = SolverQDim + SolverYDim;
    using SolverZSpace = RowVectord<SolverZDim>;
    constexpr auto solver_dec = Variable<VarMeta<SolverZSpace>>();

    constexpr auto A0 = MatBuilder(-1.0, 0.0, 0.0, -2.0);
    constexpr auto A1 = MatBuilder(-0.5, 0.0, 0.0, -1.0);
    constexpr auto B0 = BCol;
    constexpr auto B1 = BCol;
    constexpr auto C = CRow;

    constexpr auto plant0 = SolverPlant(SolverPlantSS(A0, B0, C));
    constexpr auto plant1 = SolverPlant(SolverPlantSS(A1, B1, C));
    constexpr auto poly = Vector<decltype(plant0), 2>{plant0, plant1};

    constexpr auto psf = PolytopicStateFeedback<decltype(plant0), 2>();
    constexpr auto Q = Matrixd<SolverZDim, SolverZDim>{
        Vectord<SolverZDim>{1.0, 0.0, 0.0, 0.0, 0.0},
        Vectord<SolverZDim>{0.0, 1.0, 0.0, 0.0, 0.0},
        Vectord<SolverZDim>{0.0, 0.0, 1.0, 0.0, 0.0},
        Vectord<SolverZDim>{0.0, 0.0, 0.0, 1.0, 0.0},
        Vectord<SolverZDim>{0.0, 0.0, 0.0, 0.0, 1.0},
    };
    constexpr auto q = Vectord<SolverZDim>{};
    constexpr auto obj =
        make_polytopic_state_feedback_quadratic_objective<decltype(plant0), 2>(
            Q, q, solver_dec);
    const auto prob =
        make_polytopic_state_feedback_problem<decltype(plant0), 2>(
            poly, solver_dec, obj);

    REQUIRE(prob.validate());
    REQUIRE(std::tuple_size<decltype(prob.constraint_cones())>::value > 3);

    const auto cfg = pldi_solver_config<double>();
    const PrimalDualIPM solver(prob, cfg);
    const auto& res = solver.res;
#ifdef AKREON_REPORT_OUTPUT
    if (res.tag != SolverResult::OPTIMAL || res.stalled) log(res);
#endif

    REQUIRE(res.tag == SolverResult::OPTIMAL);
    REQUIRE(!res.stalled);

    Vectord<SolverQDim> q_vec{};
    for (std::size_t i = 0; i < SolverQDim; ++i) q_vec(i) = res.iterate.z(i);
    const auto Q_res = make_smat<2>(q_vec);

    RowVectord<2> Y_res{};
    for (std::size_t j = 0; j < 2; ++j)
      Y_res(0, j) = res.iterate.z(SolverQDim + j);

    const auto [K_res, K_event] = psf.K(Q_res, Y_res);
    REQUIRE(K_event == linalg::NumericalEvent::NONE);
    REQUIRE(all_equal(Y_res, K_res * Q_res, 1e-4));

    const auto cert0 = (-1.0) * (A0 * Q_res + Q_res * transpose(A0) +
                                 B0 * Y_res + transpose(Y_res) * transpose(B0));
    const auto cert1 = (-1.0) * (A1 * Q_res + Q_res * transpose(A1) +
                                 B1 * Y_res + transpose(Y_res) * transpose(B1));
    const auto cert0_sym = SymmetricMatrix(cert0, 1e-6);
    const auto cert1_sym = SymmetricMatrix(cert1, 1e-6);
    REQUIRE(cert0_sym.is_valid());
    REQUIRE(cert1_sym.is_valid());
    REQUIRE(cert0_sym.is_psd());
    REQUIRE(cert1_sym.is_psd());
  }
}

TEST_CASE("H2 Polytopic State Feedback synthesis 2x2",
          "[polytopic_state_feedback_test]") {
  using StateSpaceMat = Matrixd<2, 2>;
  using InputMat = Matrixd<2, 2>;
  using OutputMat = RowVectord<2>;
  using SS = StateSpace<TimeDomainType::CONTINUOUS, StateSpaceMat, InputMat,
                        OutputMat>;
  using Inputs = PortLayout<Port<PortRole::EXOGENOUS, 0, 1>,
                            Port<PortRole::CONTROL, 1, 1>>;
  using Outputs = PortLayout<Port<PortRole::PERFORMANCE, 0, 1>>;
  using Plant = PlantModel<SS, Inputs, Outputs>;

  constexpr auto MatBuilder = mat_traits<StateSpaceMat>();
  constexpr auto A0 = MatBuilder(-1.0, 0.0, 0.0, -2.0);
  constexpr auto A1 = MatBuilder(-0.75, 0.0, 0.0, -1.5);
  constexpr auto B = InputMat{Vectord<2>{1.0, 0.0}, Vectord<2>{0.0, 1.0}};
  constexpr auto C = OutputMat{Vectord<2>{1.0, 0.0}};
  constexpr auto D = RowVectord<2>{Vectord<2>{0.0, 0.0}};
  constexpr auto plant0 = Plant(SS(A0, B, C, D));
  constexpr auto plant1 = Plant(SS(A1, B, C, D));
  constexpr auto poly = Vector<decltype(plant0), 2>{plant0, plant1};

  using H2 = H2PolytopicStateFeedback<decltype(plant0), 2>;
  using DecVarSpace = RowVectord<H2::DecVarDim>;
  constexpr auto Z_hat = Variable<VarMeta<DecVarSpace>>();
  const auto prob =
      make_h2_polytopic_state_feedback_problem<decltype(plant0), 2>(poly,
                                                                    Z_hat);

  REQUIRE(is_problem_set_v<decltype(prob)>);
  REQUIRE(prob.validate());
  REQUIRE(std::tuple_size<decltype(prob.constraint_cones())>::value == 4);

  const auto c = prob.linear_objective();
  const auto c_exp = Vectord<H2::DecVarDim>{0.0, 0.0, 0.0, 0.0, 0.0, 1.0};
  REQUIRE(c == c_exp);

  using Solver = PrimalDualIPM<decltype(prob)>;
  auto cfg = pldi_solver_config<double>();
  cfg.steps = 96;
  cfg.duality_gap_tol = 1e-3;
  const Solver solver(prob, cfg);
  const auto& res = solver.res;
#ifdef AKREON_REPORT_OUTPUT
  if (res.tag != SolverResult::OPTIMAL || res.stalled) log(res);
#endif

  REQUIRE(res.tag == SolverResult::OPTIMAL);
  REQUIRE(!res.stalled);

  const auto q_yz_split = linalg::splice<H2::QDim>(res.iterate.z);
  const auto Q_res = make_smat<2>(std::get<0>(q_yz_split));
  const auto y_z_split = linalg::splice<H2::YDim>(std::get<1>(q_yz_split));
  RowVectord<2> Y_res{};
  for (std::size_t j = 0; j < H2::YDim; ++j)
    Y_res(0, j) = std::get<0>(y_z_split)(j);
  const auto Z_res = make_smat<1>(std::get<1>(y_z_split));

  REQUIRE(all_equal(Q_res, transpose(Q_res), rel_tol));
  REQUIRE(Q_res(0, 0) > 1e-6);
  REQUIRE(Q_res(1, 1) > 1e-6);
  REQUIRE(Z_res(0, 0) > 1e-6);

  const auto Q_sym = SymmetricMatrix(Q_res, 1e-6);
  REQUIRE(Q_sym.is_valid());
  REQUIRE(Q_sym.is_psd());

  constexpr auto h2 = H2();
  const auto [K_res, K_event] = h2.K(Q_res, Y_res);
  REQUIRE(K_event == linalg::NumericalEvent::NONE);
  REQUIRE(all_equal(Y_res, K_res * Q_res, 1e-4));

  constexpr auto B_w = ColVectord<2>{Vectord<1>{1.0}, Vectord<1>{0.0}};
  constexpr auto B_u = ColVectord<2>{Vectord<1>{0.0}, Vectord<1>{1.0}};
  constexpr auto C_z = RowVectord<2>{Vectord<2>{1.0, 0.0}};
  constexpr auto D_u = RowVectord<1>{Vectord<1>{0.0}};

  const auto cert0 =
      (-1.0) * (A0 * Q_res + Q_res * transpose(A0) + B_u * Y_res +
                transpose(Y_res) * transpose(B_u) + B_w * transpose(B_w));
  const auto cert1 =
      (-1.0) * (A1 * Q_res + Q_res * transpose(A1) + B_u * Y_res +
                transpose(Y_res) * transpose(B_u) + B_w * transpose(B_w));
  const auto cert0_sym = SymmetricMatrix(cert0, 1e-6);
  const auto cert1_sym = SymmetricMatrix(cert1, 1e-6);
  REQUIRE(cert0_sym.is_valid());
  REQUIRE(cert1_sym.is_valid());
  REQUIRE(cert0_sym.is_psd());
  REQUIRE(cert1_sym.is_psd());

  const auto top_right = C_z * Q_res + D_u * Y_res;
  const auto epigraph =
      Matrixd<3, 3>{Vectord<3>{Z_res(0, 0), top_right(0, 0), top_right(0, 1)},
                    Vectord<3>{top_right(0, 0), Q_res(0, 0), Q_res(0, 1)},
                    Vectord<3>{top_right(0, 1), Q_res(1, 0), Q_res(1, 1)}};
  const auto epigraph_sym = SymmetricMatrix(epigraph, 1e-6);
  REQUIRE(epigraph_sym.is_valid());
  REQUIRE(epigraph_sym.is_psd());

  const auto cl0 = A0 + B_u * K_res;
  const auto cl1 = A1 + B_u * K_res;
  const auto lyap0 =
      (-1.0) * (cl0 * Q_res + Q_res * transpose(cl0) + B_w * transpose(B_w));
  const auto lyap1 =
      (-1.0) * (cl1 * Q_res + Q_res * transpose(cl1) + B_w * transpose(B_w));
  const auto lyap0_sym = SymmetricMatrix(lyap0, 1e-6);
  const auto lyap1_sym = SymmetricMatrix(lyap1, 1e-6);
  REQUIRE(lyap0_sym.is_valid());
  REQUIRE(lyap1_sym.is_valid());
  REQUIRE(lyap0_sym.is_psd());
  REQUIRE(lyap1_sym.is_psd());
}

TEST_CASE("Hinf Polytopic State Feedback synthesis 2x2",
          "[polytopic_state_feedback_test]") {
  using StateSpaceMat = Matrixd<2, 2>;
  using InputMat = Matrixd<2, 2>;
  using OutputMat = RowVectord<2>;
  using SS = StateSpace<TimeDomainType::CONTINUOUS, StateSpaceMat, InputMat,
                        OutputMat>;
  using Inputs = PortLayout<Port<PortRole::EXOGENOUS, 0, 1>,
                            Port<PortRole::CONTROL, 1, 1>>;
  using Outputs = PortLayout<Port<PortRole::PERFORMANCE, 0, 1>>;
  using Plant = PlantModel<SS, Inputs, Outputs>;

  constexpr auto MatBuilder = mat_traits<StateSpaceMat>();
  constexpr auto A0 = MatBuilder(-1.0, 0.0, 0.0, -2.0);
  constexpr auto A1 = MatBuilder(-0.75, 0.0, 0.0, -1.5);
  constexpr auto B = InputMat{Vectord<2>{1.0, 0.0}, Vectord<2>{0.0, 1.0}};
  constexpr auto C = OutputMat{Vectord<2>{1.0, 0.0}};
  constexpr auto D = RowVectord<2>{Vectord<2>{0.0, 0.0}};
  constexpr auto plant0 = Plant(SS(A0, B, C, D));
  constexpr auto plant1 = Plant(SS(A1, B, C, D));
  constexpr auto poly = Vector<decltype(plant0), 2>{plant0, plant1};

  using Hinf = HinfPolytopicStateFeedback<decltype(plant0), 2>;
  using DecVarSpace = RowVectord<Hinf::DecVarDim>;
  constexpr auto Z_hat = Variable<VarMeta<DecVarSpace>>();
  const auto prob =
      make_hinf_polytopic_state_feedback_problem<decltype(plant0), 2>(poly,
                                                                      Z_hat);

  REQUIRE(is_problem_set_v<decltype(prob)>);
  REQUIRE(prob.validate());
  REQUIRE(std::tuple_size<decltype(prob.constraint_cones())>::value == 4);

  const auto c = prob.linear_objective();
  const auto c_exp = Vectord<Hinf::DecVarDim>{0.0, 0.0, 0.0, 0.0, 0.0, 1.0};
  REQUIRE(c == c_exp);

  using Solver = PrimalDualIPM<decltype(prob)>;
  auto cfg = pldi_solver_config<double>();
  cfg.steps = 96;
  cfg.duality_gap_tol = 1e-3;
  const Solver solver(prob, cfg);
  const auto& res = solver.res;
#ifdef AKREON_REPORT_OUTPUT
  if (res.tag != SolverResult::OPTIMAL || res.stalled) log(res);
#endif

  REQUIRE(res.tag == SolverResult::OPTIMAL);
  REQUIRE(!res.stalled);

  const auto q_y_gamma_split = linalg::splice<Hinf::QDim>(res.iterate.z);
  const auto Q_res = make_smat<2>(std::get<0>(q_y_gamma_split));
  const auto y_gamma_split =
      linalg::splice<Hinf::YDim>(std::get<1>(q_y_gamma_split));
  RowVectord<2> Y_res{};
  for (std::size_t j = 0; j < Hinf::YDim; ++j)
    Y_res(0, j) = std::get<0>(y_gamma_split)(j);
  const auto gamma_res = std::get<1>(y_gamma_split)(0);

  REQUIRE(all_equal(Q_res, transpose(Q_res), rel_tol));
  REQUIRE(Q_res(0, 0) > 1e-6);
  REQUIRE(Q_res(1, 1) > 1e-6);
  REQUIRE(gamma_res > 1e-6);

  const auto Q_sym = SymmetricMatrix(Q_res, 1e-6);
  REQUIRE(Q_sym.is_valid());
  REQUIRE(Q_sym.is_psd());

  constexpr auto hinf = Hinf();
  const auto [K_res, K_event] = hinf.K(Q_res, Y_res);
  REQUIRE(K_event == linalg::NumericalEvent::NONE);
  REQUIRE(all_equal(Y_res, K_res * Q_res, 1e-4));

  constexpr auto B_w = ColVectord<2>{Vectord<1>{1.0}, Vectord<1>{0.0}};
  constexpr auto B_u = ColVectord<2>{Vectord<1>{0.0}, Vectord<1>{1.0}};
  constexpr auto C_z = RowVectord<2>{Vectord<2>{1.0, 0.0}};
  constexpr auto D_w = RowVectord<1>{Vectord<1>{0.0}};
  constexpr auto D_u = RowVectord<1>{Vectord<1>{0.0}};

  auto check_vertex = [&](const StateSpaceMat& A) {
    const auto top_left = A * Q_res + Q_res * transpose(A) + B_u * Y_res +
                          transpose(Y_res) * transpose(B_u);
    const auto top_right = C_z * Q_res + D_u * Y_res;

    Matrixd<4, 4> brl{};
    brl = set_block<0, 2, 0, 2>(brl, top_left);
    brl = set_block<0, 2, 2, 3>(brl, B_w);
    brl = set_block<2, 3, 0, 2>(brl, transpose(B_w));
    brl = set_block<0, 2, 3, 4>(brl, transpose(top_right));
    brl = set_block<3, 4, 0, 2>(brl, top_right);
    brl = set_block<2, 3, 2, 3>(brl, Matrixd<1, 1>{Vectord<1>{-gamma_res}});
    brl = set_block<2, 3, 3, 4>(brl, D_w);
    brl = set_block<3, 4, 2, 3>(brl, transpose(D_w));
    brl = set_block<3, 4, 3, 4>(brl, Matrixd<1, 1>{Vectord<1>{-gamma_res}});

    const auto brl_sym = SymmetricMatrix((-1.0) * brl, 1e-6);
    REQUIRE(brl_sym.is_valid());
    REQUIRE(brl_sym.is_psd());

    const auto cl = A + B_u * K_res;
    const auto cl_perf = C_z + D_u * K_res;
    REQUIRE(all_equal(top_left, cl * Q_res + Q_res * transpose(cl), 1e-4));
    REQUIRE(all_equal(top_right, cl_perf * Q_res, 1e-4));
  };

  check_vertex(A0);
  check_vertex(A1);
}
