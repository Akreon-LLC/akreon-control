cmake_minimum_required(VERSION 3.30)
project(AkreonControl VERSION 0.1.0 LANGUAGES CXX)
option(BUILD_UNIT_TESTS "Build Catch2 unit tests" ${PROJECT_IS_TOP_LEVEL})
option(BUILD_FUNCTIONAL_TESTS "Build functional tests" ${PROJECT_IS_TOP_LEVEL})
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(cmake/AkreonDependencies.cmake)
akreon_resolve_dependency(linalg AkreonLinalg Akreon::Linalg)
akreon_resolve_dependency(convex AkreonConvex Akreon::Convex)

add_library(akreon_control INTERFACE)
add_library(Akreon::Control ALIAS akreon_control)
set_target_properties(akreon_control PROPERTIES EXPORT_NAME Control)
target_compile_features(akreon_control INTERFACE cxx_std_26)
target_include_directories(akreon_control INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(akreon_control INTERFACE Akreon::Linalg Akreon::Convex)

install(TARGETS akreon_control EXPORT AkreonControlTargets)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES LICENSE NOTICE DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/${PROJECT_NAME})
install(EXPORT AkreonControlTargets NAMESPACE Akreon:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/AkreonControl)
configure_package_config_file(cmake/AkreonControlConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/AkreonControlConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/AkreonControl)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/AkreonControlConfigVersion.cmake
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/AkreonControlConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/AkreonControlConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/AkreonControl)

if(PROJECT_IS_TOP_LEVEL AND (BUILD_UNIT_TESTS OR BUILD_FUNCTIONAL_TESTS))
  enable_testing()
  add_subdirectory(tests)
endif()
option(AKREON_CONTROL_BUILD_C_BINDINGS "Build the optional native C ABI" OFF)
option(AKREON_CONTROL_BUILD_PYTHON "Build the optional Python extension" OFF)
option(AKREON_CONTROL_BUILD_BINDING_TESTS "Build C ABI tests" OFF)
if(AKREON_CONTROL_BUILD_C_BINDINGS OR AKREON_CONTROL_BUILD_PYTHON)
  if(PROJECT_IS_TOP_LEVEL AND AKREON_CONTROL_BUILD_BINDING_TESTS)
    enable_testing()
  endif()
  add_subdirectory(bindings)
endif()

find_program(CLANG_FORMAT_BIN NAMES clang-format-22 clang-format)
if(PROJECT_IS_TOP_LEVEL AND CLANG_FORMAT_BIN)
  file(GLOB_RECURSE format_files CONFIGURE_DEPENDS include/*.hpp tests/*.cpp tests/*.hpp test-support/*.hpp examples/*.cpp examples/*.hpp bindings/*.hpp bindings/*.h bindings/*.cpp)
  add_custom_target(clang_format_check COMMAND ${CLANG_FORMAT_BIN} --dry-run --Werror ${format_files} VERBATIM)
  add_custom_target(clang_format_fix COMMAND ${CLANG_FORMAT_BIN} -i ${format_files} VERBATIM)
endif()

option(AKREON_CONTROL_BUILD_EXAMPLES "Build native control examples and their checks" OFF)
if(AKREON_CONTROL_BUILD_EXAMPLES)
  enable_testing()
  add_subdirectory(examples/clf_cbf_qp)
endif()
