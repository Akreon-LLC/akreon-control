/*
 * Copyright © 2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/** @brief Runtime-to-template dispatch helpers for generalized controllers. */

#include <array>
#include <tuple>

#include "util.hpp"

namespace akreon_native::controls_detail {

template <size_t Index, size_t... StaticDims, typename Tuple, size_t Count,
          typename Fn>
ak_native_status dispatch_dimensions_impl(
    const Tuple& dim_lists, const std::array<size_t, Count>& values, Fn& fn) {
  if constexpr (Index == Count) {
    return fn.template operator()<StaticDims...>();
  } else {
    return dispatch_dimension(
        std::get<Index>(dim_lists), values[Index], [&]<size_t Dim>() {
          return dispatch_dimensions_impl<Index + 1, StaticDims..., Dim>(
              dim_lists, values, fn);
        });
  }
}

/**
 * @brief Dispatch several runtime dimensions to one static template branch.
 *
 * Callers supply the supported dimension lists for each operation.
 */
template <typename... DimLists, typename Fn>
ak_native_status dispatch_dimensions(
    const std::tuple<DimLists...>& dim_lists,
    const std::array<size_t, sizeof...(DimLists)>& values, Fn&& fn) {
  return dispatch_dimensions_impl<0>(dim_lists, values, fn);
}

/** @brief Dispatch exact supported shapes without expanding dimension products.
 */
template <typename... Shapes, size_t Count, typename Fn>
ak_native_status dispatch_shapes(ShapeList<Shapes...>,
                                 const std::array<size_t, Count>& values,
                                 Fn&& fn) {
  ak_native_status status = AK_NATIVE_UNSUPPORTED_DIMENSION;
  ((Shapes::values == values ? (status = Shapes::invoke(fn), 0) : 0), ...);
  return status;
}

/**
 * @brief Dispatch a generalized feedback solve from runtime dimensions.
 *
 * @tparam Problem generalized feedback policy type
 *
 * @param A borrowed dynamics matrix view
 * @param B borrowed combined input matrix view
 * @param C borrowed performance output matrix view
 * @param D borrowed feedthrough matrix view
 * @param config optional native solver config
 * @param K mutable state-feedback gain output view
 * @param result solver diagnostics output
 * @return native status code
 */
template <typename Problem>
ak_native_status dispatch_generalized_feedback(
    const ak_native_matrix_view_f64& A, const ak_native_matrix_view_f64& B,
    const ak_native_matrix_view_f64& C, const ak_native_matrix_view_f64& D,
    size_t w, const ak_native_solver_config_f64* config,
    const ak_native_matrix_mut_f64& K, ak_native_solver_result_f64* result) {
  const size_t m = B.cols - w;
  if constexpr (requires { typename Problem::Shapes; }) {
    return dispatch_shapes(typename Problem::Shapes{},
                           std::array<size_t, 4>{A.rows, w, m, C.rows},
                           [&]<size_t N, size_t W, size_t M, size_t Z>() {
                             return Problem::template solve<N, W, M, Z>(
                                 A, B, C, D, config, K, result);
                           });
  } else {
    return dispatch_dimensions(std::tuple{typename Problem::StateDims{},
                                          typename Problem::ExogenousDims{},
                                          typename Problem::ControlDims{},
                                          typename Problem::PerformanceDims{}},
                               std::array<size_t, 4>{A.rows, w, m, C.rows},
                               [&]<size_t N, size_t W, size_t M, size_t Z>() {
                                 return Problem::template solve<N, W, M, Z>(
                                     A, B, C, D, config, K, result);
                               });
  }
}

/**
 * @brief Dispatch a generalized observer solve from runtime dimensions.
 *
 * @tparam Problem generalized observer policy type
 *
 * @param A borrowed dynamics matrix view
 * @param B borrowed exogenous input matrix view
 * @param C borrowed combined performance/measurement output matrix view
 * @param D borrowed feedthrough matrix view
 * @param config optional native solver config
 * @param L mutable observer gain output view
 * @param result solver diagnostics output
 * @return native status code
 */
template <typename Problem>
ak_native_status dispatch_generalized_observer(
    const ak_native_matrix_view_f64& A, const ak_native_matrix_view_f64& B,
    const ak_native_matrix_view_f64& C, const ak_native_matrix_view_f64& D,
    size_t z, const ak_native_solver_config_f64* config,
    const ak_native_matrix_mut_f64& L, ak_native_solver_result_f64* result) {
  const size_t w = B.cols;
  const size_t y = C.rows - z;
  return dispatch_dimensions(std::tuple{typename Problem::StateDims{},
                                        typename Problem::ExogenousDims{},
                                        typename Problem::PerformanceDims{},
                                        typename Problem::MeasurementDims{}},
                             std::array<size_t, 4>{A.rows, w, z, y},
                             [&]<size_t N, size_t W, size_t Z, size_t Y>() {
                               return Problem::template solve<N, W, Z, Y>(
                                   A, B, C, D, config, L, result);
                             });
}

/**
 * @brief Dispatch a polytopic feedback solve from runtime dimensions.
 *
 * @tparam Problem polytopic feedback policy type
 *
 * @param A borrowed array of dynamics matrix views
 * @param B borrowed array of combined input matrix views
 * @param C borrowed array of performance output matrix views
 * @param D borrowed array of feedthrough matrix views
 * @param config optional native solver config
 * @param K mutable state-feedback gain output view
 * @param result solver diagnostics output
 * @return native status code
 */
template <typename Problem>
ak_native_status dispatch_polytopic_feedback(
    const ak_native_matrix_view_f64* A, const ak_native_matrix_view_f64* B,
    const ak_native_matrix_view_f64* C, const ak_native_matrix_view_f64* D,
    size_t vertex_count, size_t w, const ak_native_solver_config_f64* config,
    const ak_native_matrix_mut_f64& K, ak_native_solver_result_f64* result) {
  const size_t m = B[0].cols - w;
  return dispatch_dimensions(
      std::tuple{typename Problem::VertexDims{}, typename Problem::StateDims{},
                 typename Problem::ExogenousDims{},
                 typename Problem::ControlDims{},
                 typename Problem::PerformanceDims{}},
      std::array<size_t, 5>{vertex_count, A[0].rows, w, m, C[0].rows},
      [&]<size_t V, size_t N, size_t W, size_t M, size_t Z>() {
        return Problem::template solve<V, N, W, M, Z>(A, B, C, D, config, K,
                                                      result);
      });
}

/**
 * @brief Dispatch a polytopic observer solve from runtime dimensions.
 *
 * @tparam Problem polytopic observer policy type
 *
 * @param A borrowed array of dynamics matrix views
 * @param B borrowed array of exogenous input matrix views
 * @param C borrowed array of combined performance/measurement output views
 * @param D borrowed array of feedthrough matrix views
 * @param config optional native solver config
 * @param L mutable observer gain output view
 * @param result solver diagnostics output
 * @return native status code
 */
template <typename Problem>
ak_native_status dispatch_polytopic_observer(
    const ak_native_matrix_view_f64* A, const ak_native_matrix_view_f64* B,
    const ak_native_matrix_view_f64* C, const ak_native_matrix_view_f64* D,
    size_t vertex_count, size_t z, const ak_native_solver_config_f64* config,
    const ak_native_matrix_mut_f64& L, ak_native_solver_result_f64* result) {
  const size_t w = B[0].cols;
  const size_t y = C[0].rows - z;
  return dispatch_dimensions(
      std::tuple{typename Problem::VertexDims{}, typename Problem::StateDims{},
                 typename Problem::ExogenousDims{},
                 typename Problem::PerformanceDims{},
                 typename Problem::MeasurementDims{}},
      std::array<size_t, 5>{vertex_count, A[0].rows, w, z, y},
      [&]<size_t V, size_t N, size_t W, size_t Z, size_t Y>() {
        return Problem::template solve<V, N, W, Z, Y>(A, B, C, D, config, L,
                                                      result);
      });
}

} // namespace akreon_native::controls_detail
