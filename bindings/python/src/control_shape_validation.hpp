/*
 * Copyright © 2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <sstream>
#include <string>

#include <nanobind/nanobind.h>

#include "ndarray_bridge.hpp"

namespace akreon_native::python {

/**
 * @brief Python-facing shape validation for public solve functions.
 *
 * These helpers provide clear `ValueError` messages before crossing the native
 * boundary. They mirror the stable facade contracts and compiled
 * dimension surface, while the C ABI still performs authoritative validation
 * against the exact compiled dimension configuration.
 */
template <size_t Rank>
inline std::string shapeN(const auto& array) {
  std::ostringstream os;
  os << "(";
  for (size_t idx = 0; idx < Rank; ++idx) {
    if (idx != 0) os << ", ";
    os << array.shape(idx);
  }
  os << ")";
  return os.str();
}

inline void fail_shape(const char* op, const char* message) {
  throw nb::value_error((std::string(op) + ": " + message).c_str());
}

inline void expect(bool ok, const char* op, const std::string& message) {
  if (!ok) throw nb::value_error((std::string(op) + ": " + message).c_str());
}

inline void validate_state_feedback_shapes(const Array2D& A, const Array2D& B,
                                           const char* op) {
  const auto n = A.shape(0);
  expect(n == A.shape(1), op,
         "expected A to be square, got A.shape == " + shapeN<2>(A));
  expect(B.shape(0) == n, op,
         "expected B.shape[0] == A.shape[0], got B.shape == " + shapeN<2>(B));
  expect(n != 0 && B.shape(1) != 0, op, "expected A and B to be non-empty");
}

inline void validate_state_observer_shapes(const Array2D& A, const Array2D& C,
                                           const char* op) {
  const auto n = A.shape(0);
  const auto y = C.shape(0);
  expect(n == A.shape(1), op,
         "expected A to be square, got A.shape == " + shapeN<2>(A));
  expect(C.shape(1) == n, op,
         "expected C.shape[1] == A.shape[0], got C.shape == " + shapeN<2>(C));
  expect(n != 0 && y != 0, op, "expected A and C to be non-empty");
}

inline void validate_generalized_state_feedback_shapes(
    const Array2D& A, const Array2D& B, const Array2D& C, const Array2D& D,
    size_t split, const char* op) {
  const auto n = A.shape(0);
  const auto total_inputs = B.shape(1);
  const auto z = C.shape(0);
  expect(n == A.shape(1), op,
         "expected A to be square, got A.shape == " + shapeN<2>(A));
  expect(B.shape(0) == n, op,
         "expected B.shape[0] == A.shape[0], got B.shape == " + shapeN<2>(B));
  expect(C.shape(1) == n, op,
         "expected C.shape[1] == A.shape[0], got C.shape == " + shapeN<2>(C));
  expect(D.shape(0) == z && D.shape(1) == total_inputs, op,
         "expected D.shape == (C.shape[0], B.shape[1]), got D.shape == " +
             shapeN<2>(D));
  expect(split != 0 && split < total_inputs, op,
         "compiled [w,u] split does not match B");
}

inline void validate_generalized_state_observer_shapes(
    const Array2D& A, const Array2D& B, const Array2D& C, const Array2D& D,
    size_t split, const char* op) {
  const auto n = A.shape(0);
  const auto total_inputs = B.shape(1);
  const auto total_outputs = C.shape(0);
  expect(n == A.shape(1), op,
         "expected A to be square, got A.shape == " + shapeN<2>(A));
  expect(B.shape(0) == n, op,
         "expected B.shape[0] == A.shape[0], got B.shape == " + shapeN<2>(B));
  expect(C.shape(1) == n, op,
         "expected C.shape[1] == A.shape[0], got C.shape == " + shapeN<2>(C));
  expect(D.shape(0) == total_outputs && D.shape(1) == total_inputs, op,
         "expected D.shape == (C.shape[0], B.shape[1]), got D.shape == " +
             shapeN<2>(D));
  expect(split != 0 && split < total_outputs, op,
         "compiled [z,y] split does not match C");
}

inline void validate_same_vertex_count(const Array3D& A, const Array3D& B,
                                       const char* op) {
  expect(A.shape(0) == B.shape(0), op,
         "expected matching vertex count, got A.shape == " + shapeN<3>(A) +
             " and B.shape == " + shapeN<3>(B));
}

inline void validate_vertex_count(const Array3D& A, const char* op) {
  expect(A.shape(0) != 0, op, "expected at least one polytope vertex");
}

inline void validate_polytopic_state_feedback_shapes(const Array3D& A,
                                                     const Array3D& B,
                                                     const char* op) {
  validate_same_vertex_count(A, B, op);
  validate_vertex_count(A, op);
  const auto n = A.shape(1);
  expect(n == A.shape(2), op,
         "expected A vertices to be square, got A.shape == " + shapeN<3>(A));
  expect(B.shape(1) == n, op,
         "expected B.shape[1] == A.shape[1], got B.shape == " + shapeN<3>(B));
  expect(n != 0 && B.shape(2) != 0, op,
         "expected A and B vertices to be non-empty");
}

inline void validate_polytopic_state_observer_shapes(const Array3D& A,
                                                     const Array3D& C,
                                                     const char* op) {
  validate_same_vertex_count(A, C, op);
  validate_vertex_count(A, op);
  const auto n = A.shape(1);
  expect(n == A.shape(2), op,
         "expected A vertices to be square, got A.shape == " + shapeN<3>(A));
  expect(C.shape(2) == n, op,
         "expected C.shape[2] == A.shape[1], got C.shape == " + shapeN<3>(C));
  expect(n != 0 && C.shape(1) != 0, op,
         "expected A and C vertices to be non-empty");
}

inline void validate_hx_polytopic_state_feedback_shapes(
    const Array3D& A, const Array3D& B, const Array3D& C, const Array3D& D,
    size_t split, const char* op) {
  expect(A.shape(0) == B.shape(0) && A.shape(0) == C.shape(0) &&
             A.shape(0) == D.shape(0),
         op, "A, B, C, and D must have the same vertex count");
  validate_vertex_count(A, op);
  const auto n = A.shape(1);
  const auto total_inputs = B.shape(2);
  const auto z = C.shape(1);
  expect(n == A.shape(2), op,
         "expected A vertices to be square, got A.shape == " + shapeN<3>(A));
  expect(B.shape(1) == n, op,
         "expected B.shape[1] == A.shape[1], got B.shape == " + shapeN<3>(B));
  expect(C.shape(2) == n, op,
         "expected C.shape[2] == A.shape[1], got C.shape == " + shapeN<3>(C));
  expect(D.shape(1) == z && D.shape(2) == total_inputs, op,
         "expected D.shape == (vertices, C.shape[1], B.shape[2]), got D.shape "
         "== " +
             shapeN<3>(D));
  expect(split != 0 && split < total_inputs, op,
         "compiled [w,u] split does not match B");
}

inline void validate_hx_polytopic_state_observer_shapes(
    const Array3D& A, const Array3D& B, const Array3D& C, const Array3D& D,
    size_t split, const char* op) {
  expect(A.shape(0) == B.shape(0) && A.shape(0) == C.shape(0) &&
             A.shape(0) == D.shape(0),
         op, "A, B, C, and D must have the same vertex count");
  validate_vertex_count(A, op);
  const auto n = A.shape(1);
  const auto total_inputs = B.shape(2);
  const auto total_outputs = C.shape(1);
  expect(n == A.shape(2), op,
         "expected A vertices to be square, got A.shape == " + shapeN<3>(A));
  expect(B.shape(1) == n, op,
         "expected B.shape[1] == A.shape[1], got B.shape == " + shapeN<3>(B));
  expect(C.shape(2) == n, op,
         "expected C.shape[2] == A.shape[1], got C.shape == " + shapeN<3>(C));
  expect(D.shape(1) == total_outputs && D.shape(2) == total_inputs, op,
         "expected D.shape == (vertices, C.shape[1], B.shape[2]), got D.shape "
         "== " +
             shapeN<3>(D));
  expect(split != 0 && split < total_outputs, op,
         "compiled [z,y] split does not match C");
}

inline void validate_polytopic_state_space_shapes(const Array3D& A,
                                                  const Array3D& B,
                                                  const Array3D& C,
                                                  const Array3D& D,
                                                  const char* op) {
  expect(A.shape(0) == B.shape(0) && A.shape(0) == C.shape(0) &&
             A.shape(0) == D.shape(0),
         op, "plant A, B, C, and D must have the same vertex count");
  validate_vertex_count(A, op);
  const auto n = A.shape(1);
  expect(n == A.shape(2), op,
         "expected plant A vertices to be square, got A.shape == " +
             shapeN<3>(A));
  expect(B.shape(1) == n, op,
         "expected plant B.shape[1] == A.shape[1], got B.shape == " +
             shapeN<3>(B));
  expect(C.shape(2) == n, op,
         "expected plant C.shape[2] == A.shape[1], got C.shape == " +
             shapeN<3>(C));
  expect(D.shape(1) == C.shape(1) && D.shape(2) == B.shape(2), op,
         "expected plant D.shape == (vertices, C.shape[1], B.shape[2]), got "
         "D.shape == " +
             shapeN<3>(D));
  expect(n != 0 && B.shape(2) != 0 && C.shape(1) != 0, op,
         "expected non-empty plant vertices");
}

} // namespace akreon_native::python
