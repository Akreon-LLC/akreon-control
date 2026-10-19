# Copyright © 2026 Akreon LLC
# SPDX-License-Identifier: Apache-2.0

# Optional native C ABI and Python extension; the C++ interface target stays lean.
set(AKREON_CONTROL_BINDING_PD_IPM_STEPS 64 CACHE STRING "Iteration budget compiled into control bindings")
file(GLOB_RECURSE binding_sources CONFIGURE_DEPENDS c/src/*.cpp)
add_library(akreon_control_c STATIC ${binding_sources})
add_library(Akreon::ControlC ALIAS akreon_control_c)
set_target_properties(akreon_control_c PROPERTIES EXPORT_NAME ControlC POSITION_INDEPENDENT_CODE ON)
target_link_libraries(akreon_control_c PUBLIC Akreon::Control)
target_include_directories(akreon_control_c PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/c/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_definitions(akreon_control_c PUBLIC PD_IPM_STEPS=${AKREON_CONTROL_BINDING_PD_IPM_STEPS})
install(TARGETS akreon_control_c EXPORT AkreonControlTargets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY c/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

if(AKREON_CONTROL_BUILD_PYTHON)
  find_package(Python 3.12 COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(nanobind CONFIG QUIET)
  if(NOT nanobind_FOUND)
    execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
      OUTPUT_VARIABLE nanobind_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
      COMMAND_ERROR_IS_FATAL ANY)
    find_package(nanobind CONFIG REQUIRED)
  endif()
  nanobind_add_module(_akreon_native python/src/akreon_native_bindings.cpp)
  target_link_libraries(_akreon_native PRIVATE Akreon::ControlC)
  install(TARGETS _akreon_native LIBRARY DESTINATION akreon_native COMPONENT python)
endif()

if(AKREON_CONTROL_BUILD_BINDING_TESTS)
  add_executable(akreon_native_c_test c/tests/test_state_feedback.cpp)
  target_link_libraries(akreon_native_c_test PRIVATE Akreon::ControlC)
  add_test(NAME akreon_native_c_test COMMAND akreon_native_c_test)
endif()
