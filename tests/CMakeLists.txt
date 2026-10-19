function(akreon_test_options target)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(${target} PRIVATE -fconstexpr-depth=65536 -fconstexpr-steps=2000000000)
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 16
        AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 17)
      # These synthesis tests solve at runtime. Bound GCC 16's speculative
      # constant evaluation, as in Convex's SDP tests; retain constexpr model checks.
      # Larger control LMI canonicalizations require a five-million-op budget.
      target_compile_options(${target} PRIVATE -fconstexpr-depth=65536 -fconstexpr-ops-limit=5000000)
    else()
      target_compile_options(${target} PRIVATE -fconstexpr-depth=65536 -fconstexpr-ops-limit=2000000000)
    endif()
  endif()
  target_compile_definitions(${target} PRIVATE AKREON_REPORT_OUTPUT)
endfunction()

if(BUILD_UNIT_TESTS)
  include(FetchContent)
  find_package(Catch2 3 CONFIG QUIET)
  if(NOT TARGET Catch2::Catch2WithMain)
    FetchContent_Declare(Catch2 GIT_REPOSITORY https://github.com/catchorg/Catch2.git GIT_TAG v3.8.1)
    FetchContent_MakeAvailable(Catch2)
  endif()

  add_executable(discretization_test unit/discretization_test.cpp)
  target_link_libraries(discretization_test PRIVATE Akreon::Control Catch2::Catch2WithMain)
  akreon_test_options(discretization_test)
  add_test(NAME discretization_test COMMAND discretization_test)

  add_executable(plant_model_test unit/plant_model_test.cpp)
  target_link_libraries(plant_model_test PRIVATE Akreon::Control Akreon::ConvexTestSupport Catch2::Catch2WithMain)
  akreon_test_options(plant_model_test)
  add_test(NAME plant_model_test COMMAND plant_model_test)

  add_executable(polytopic_observer_test unit/polytopic_observer_test.cpp)
  target_link_libraries(polytopic_observer_test PRIVATE Akreon::Control Akreon::ConvexTestSupport Catch2::Catch2WithMain)
  akreon_test_options(polytopic_observer_test)
  add_test(NAME polytopic_observer_test COMMAND polytopic_observer_test)

  add_executable(polytopic_state_feedback_test unit/polytopic_state_feedback_test.cpp)
  target_link_libraries(polytopic_state_feedback_test PRIVATE Akreon::Control Akreon::ConvexTestSupport Catch2::Catch2WithMain)
  akreon_test_options(polytopic_state_feedback_test)
  add_test(NAME polytopic_state_feedback_test COMMAND polytopic_state_feedback_test)

  add_executable(state_feedback_test unit/state_feedback_test.cpp)
  target_link_libraries(state_feedback_test PRIVATE Akreon::Control Akreon::ConvexTestSupport Catch2::Catch2WithMain)
  akreon_test_options(state_feedback_test)
  add_test(NAME state_feedback_test COMMAND state_feedback_test)

  add_executable(state_observer_test unit/state_observer_test.cpp)
  target_link_libraries(state_observer_test PRIVATE Akreon::Control Akreon::ConvexTestSupport Catch2::Catch2WithMain)
  akreon_test_options(state_observer_test)
  add_test(NAME state_observer_test COMMAND state_observer_test)
endif()
