/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#include <catch2/catch_test_macros.hpp>

#include "akreon/control/control.hpp"
#include "akreon/linalg/linalg.hpp"
#include "akreon/convex/convex.hpp"
#include <akreon/convex/test_helpers.hpp>

using namespace control;
using namespace convex;
using namespace linalg;
using namespace unit_test;

TEST_CASE("PlantModel port selection", "[plant_model_test]") {
  using AMap = Matrixd<2, 2>;
  using BMap = Matrixd<2, 4>;
  using CMap = Matrixd<4, 2>;
  using SS = StateSpace<TimeDomainType::CONTINUOUS, AMap, BMap, CMap>;
  using Inputs =
      PortLayout<Port<PortRole::CONTROL, 2, 1>, Port<PortRole::EXOGENOUS, 0, 1>,
                 Port<PortRole::CONTROL, 1, 1>,
                 Port<PortRole::EXOGENOUS, 3, 1>>;
  using Outputs = PortLayout<
      Port<PortRole::PERFORMANCE, 0, 1>, Port<PortRole::MEASUREMENT, 2, 1>,
      Port<PortRole::MEASUREMENT, 3, 1>, Port<PortRole::PERFORMANCE, 1, 1>>;
  using Plant = PlantModel<SS, Inputs, Outputs>;

  STATIC_REQUIRE(StateSpaceLike<SS>);
  STATIC_REQUIRE(!PlantModelLike<SS>);
  STATIC_REQUIRE(PlantModelLike<Plant>);
  STATIC_REQUIRE(ContinuousPlantModelLike<Plant>);
  STATIC_REQUIRE(plant_traits<Plant>::ControlDim == 2);
  STATIC_REQUIRE(plant_traits<Plant>::ExogenousDim == 2);
  STATIC_REQUIRE(plant_traits<Plant>::MeasurementDim == 2);
  STATIC_REQUIRE(plant_traits<Plant>::PerformanceDim == 2);

  constexpr auto A = Matrixd<2, 2>{
      Vectord<2>{1.0, 2.0},
      Vectord<2>{3.0, 4.0},
  };
  constexpr auto B = Matrixd<2, 4>{
      Vectord<4>{10.0, 11.0, 12.0, 13.0},
      Vectord<4>{20.0, 21.0, 22.0, 23.0},
  };
  constexpr auto C = Matrixd<4, 2>{
      Vectord<2>{30.0, 31.0},
      Vectord<2>{40.0, 41.0},
      Vectord<2>{50.0, 51.0},
      Vectord<2>{60.0, 61.0},
  };
  constexpr auto D = Matrixd<4, 4>{
      Vectord<4>{100.0, 101.0, 102.0, 103.0},
      Vectord<4>{110.0, 111.0, 112.0, 113.0},
      Vectord<4>{120.0, 121.0, 122.0, 123.0},
      Vectord<4>{130.0, 131.0, 132.0, 133.0},
  };
  constexpr auto plant = Plant(SS(A, B, C, D));

  REQUIRE(
      all_equal(plant.control_matrix(),
                Matrixd<2, 2>{Vectord<2>{12.0, 11.0}, Vectord<2>{22.0, 21.0}}));
  REQUIRE(
      all_equal(plant.exogenous_matrix(),
                Matrixd<2, 2>{Vectord<2>{10.0, 13.0}, Vectord<2>{20.0, 23.0}}));
  REQUIRE(
      all_equal(plant.performance_matrix(),
                Matrixd<2, 2>{Vectord<2>{30.0, 31.0}, Vectord<2>{40.0, 41.0}}));
  REQUIRE(
      all_equal(plant.measurement_matrix(),
                Matrixd<2, 2>{Vectord<2>{50.0, 51.0}, Vectord<2>{60.0, 61.0}}));
  REQUIRE(all_equal(
      plant.feedthrough<PortRole::MEASUREMENT, PortRole::CONTROL>(),
      Matrixd<2, 2>{Vectord<2>{122.0, 121.0}, Vectord<2>{132.0, 131.0}}));
}

TEST_CASE("PlantModel selected ports match vanilla synthesis",
          "[plant_model_test]") {
  using AMap = Matrixd<2, 2>;
  using BVanilla = Matrixd<2, 1>;
  using BAug = Matrixd<2, 2>;
  using CVanilla = RowVectord<2>;
  using CAug = Matrixd<2, 2>;

  constexpr auto A = Matrixd<2, 2>{
      Vectord<2>{0.0, 1.0},
      Vectord<2>{-2.0, -3.0},
  };
  constexpr auto B_u = ColVectord<2>{Vectord<1>{1.0}, Vectord<1>{0.5}};
  constexpr auto B_aug = Matrixd<2, 2>{
      Vectord<2>{9.0, 1.0},
      Vectord<2>{8.0, 0.5},
  };
  constexpr auto C_y = RowVectord<2>{Vectord<2>{1.0, 0.0}};
  constexpr auto C_aug = Matrixd<2, 2>{
      Vectord<2>{7.0, 7.0},
      Vectord<2>{1.0, 0.0},
  };

  using VanillaSS =
      StateSpace<TimeDomainType::CONTINUOUS, AMap, BVanilla, CVanilla>;
  using PlantSS = StateSpace<TimeDomainType::CONTINUOUS, AMap, BAug, CAug>;
  using VanillaInputs = PortLayout<Port<PortRole::CONTROL, 0, 1>>;
  using VanillaOutputs = PortLayout<Port<PortRole::MEASUREMENT, 0, 1>>;
  using VanillaPlant = PlantModel<VanillaSS, VanillaInputs, VanillaOutputs>;
  using Inputs = PortLayout<Port<PortRole::EXOGENOUS, 0, 1>,
                            Port<PortRole::CONTROL, 1, 1>>;
  using Outputs = PortLayout<Port<PortRole::PERFORMANCE, 0, 1>,
                             Port<PortRole::MEASUREMENT, 1, 1>>;
  using Plant = PlantModel<PlantSS, Inputs, Outputs>;

  constexpr auto vanilla = VanillaPlant(VanillaSS(A, B_u, C_y));
  constexpr auto plant = Plant(PlantSS(A, B_aug, C_aug));

  using FeedbackZ = RowVectord<StateFeedback<VanillaPlant>::ZDim>;
  constexpr auto feedback_dec = Variable<VarMeta<FeedbackZ>>();

  const auto vanilla_feedback_lmi =
      StateFeedback<VanillaPlant>()(vanilla, feedback_dec);
  const auto plant_feedback_lmi = StateFeedback<Plant>()(plant, feedback_dec);
  constexpr auto feedback_builder =
      conic::create_empty_cblk_builder<FeedbackZ>();
  const auto vanilla_feedback =
      vanilla_feedback_lmi
          .template canonicalize<AtomType::SET>(feedback_builder)
          .cb;
  const auto plant_feedback =
      plant_feedback_lmi.template canonicalize<AtomType::SET>(feedback_builder)
          .cb;

  REQUIRE(all_equal(vanilla_feedback.map.linear_part(),
                    plant_feedback.map.linear_part()));
  REQUIRE(vanilla_feedback.map.translation() ==
          plant_feedback.map.translation());

  using ObserverZ = RowVectord<LuenbergerObserver<VanillaPlant>::ZDim>;
  constexpr auto observer_dec = Variable<VarMeta<ObserverZ>>();
  const auto vanilla_observer_lmi =
      LuenbergerObserver<VanillaPlant>()(vanilla, observer_dec);
  const auto plant_observer_lmi =
      LuenbergerObserver<Plant>()(plant, observer_dec);
  constexpr auto observer_builder =
      conic::create_empty_cblk_builder<ObserverZ>();
  const auto vanilla_observer =
      vanilla_observer_lmi
          .template canonicalize<AtomType::SET>(observer_builder)
          .cb;
  const auto plant_observer =
      plant_observer_lmi.template canonicalize<AtomType::SET>(observer_builder)
          .cb;

  REQUIRE(all_equal(vanilla_observer.map.linear_part(),
                    plant_observer.map.linear_part()));
  REQUIRE(vanilla_observer.map.translation() ==
          plant_observer.map.translation());
}
