/*
 * Copyright © 2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "akreon_native/akreon_native.h"

namespace akreon_native::python {

namespace nb = nanobind;
using namespace nb::literals;

using Array1D =
    nb::ndarray<double, nb::numpy, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Array2D =
    nb::ndarray<double, nb::numpy, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using Array3D =
    nb::ndarray<double, nb::numpy, nb::ndim<3>, nb::c_contig, nb::device::cpu>;
using MutableArray1D =
    nb::ndarray<double, nb::numpy, nb::ndim<1>, nb::device::cpu>;
using MutableArray2D =
    nb::ndarray<double, nb::numpy, nb::ndim<2>, nb::device::cpu>;
using MutableTraceArray =
    nb::ndarray<int64_t, nb::numpy, nb::ndim<2>, nb::device::cpu>;

inline void require_f64(const auto& array, const char* name) {
  if (array.dtype().code !=
          static_cast<uint8_t>(nb::dlpack::dtype_code::Float) ||
      array.dtype().bits != 64 || array.dtype().lanes != 1) {
    throw nb::value_error(
        (std::string(name) + " must have dtype float64").c_str());
  }
}

inline void require_c_contiguous(const Array1D& array, const char* name) {
  require_f64(array, name);
  if (array.stride(0) != 1) {
    throw nb::value_error(
        (std::string(name) + " must be C-contiguous float64").c_str());
  }
}

inline void require_c_contiguous(const Array2D& array, const char* name) {
  require_f64(array, name);
  if (array.stride(1) != 1 ||
      array.stride(0) != static_cast<int64_t>(array.shape(1))) {
    throw nb::value_error(
        (std::string(name) + " must be C-contiguous float64").c_str());
  }
}

inline void require_c_contiguous(const Array3D& array, const char* name) {
  require_f64(array, name);
  if (array.stride(2) != 1 ||
      array.stride(1) != static_cast<int64_t>(array.shape(2)) ||
      array.stride(0) !=
          static_cast<int64_t>(array.shape(1) * array.shape(2))) {
    throw nb::value_error(
        (std::string(name) + " must be C-contiguous float64").c_str());
  }
}

inline ak_native_matrix_view_f64 view_from_array(const Array2D& array) {
  return ak_native_matrix_view_f64{
      .data = static_cast<const double*>(array.data()),
      .rows = array.shape(0),
      .cols = array.shape(1),
      .stride = static_cast<size_t>(array.stride(0)),
  };
}

inline ak_native_matrix_mut_f64 mut_view_from_array(
    const MutableArray2D& array) {
  return ak_native_matrix_mut_f64{
      .data = array.data(),
      .rows = array.shape(0),
      .cols = array.shape(1),
      .stride = static_cast<size_t>(array.stride(0)),
  };
}

constexpr ak_native_matrix_view_f64 empty_view() {
  return ak_native_matrix_view_f64{
      .data = nullptr,
      .rows = 0,
      .cols = 0,
      .stride = 0,
  };
}

inline ak_native_matrix_view_f64 optional_view(const nb::object& object,
                                               const char* name) {
  if (object.is_none()) return empty_view();
  const auto ndim = nb::cast<size_t>(object.attr("ndim"));
  if (ndim == 1) {
    const auto array = nb::cast<Array1D>(object);
    require_c_contiguous(array, name);
    return ak_native_matrix_view_f64{
        .data = static_cast<const double*>(array.data()),
        .rows = array.shape(0),
        .cols = 1,
        .stride = static_cast<size_t>(array.stride(0)),
    };
  }
  if (ndim == 2) {
    const auto array = nb::cast<Array2D>(object);
    require_c_contiguous(array, name);
    return view_from_array(array);
  }
  throw nb::value_error(
      (std::string(name) + " must be a 1D or 2D array").c_str());
}

inline std::vector<ak_native_matrix_view_f64> vertex_views_from_array(
    const Array3D& array) {
  std::vector<ak_native_matrix_view_f64> views(array.shape(0));
  const auto* data = static_cast<const double*>(array.data());
  const auto vertex_stride = static_cast<size_t>(array.stride(0));
  const auto row_stride = static_cast<size_t>(array.stride(1));
  for (size_t v = 0; v < array.shape(0); ++v) {
    views[v] = ak_native_matrix_view_f64{
        .data = data + v * vertex_stride,
        .rows = array.shape(1),
        .cols = array.shape(2),
        .stride = row_stride,
    };
  }
  return views;
}

inline nb::object make_numpy_empty(size_t rows, size_t cols) {
  const auto numpy = nb::module_::import_("numpy");
  return numpy.attr("empty")(nb::make_tuple(rows, cols),
                             "dtype"_a = numpy.attr("float64"));
}

inline nb::object make_numpy_empty_1d(size_t size) {
  const auto numpy = nb::module_::import_("numpy");
  return numpy.attr("empty")(nb::make_tuple(size),
                             "dtype"_a = numpy.attr("float64"));
}

inline nb::object make_numpy_empty_trace(size_t rows) {
  const auto numpy = nb::module_::import_("numpy");
  return numpy.attr("empty")(nb::make_tuple(rows, size_t{9}),
                             "dtype"_a = numpy.attr("int64"));
}

} // namespace akreon_native::python
