/*
 * Copyright © 2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <nanobind/nanobind.h>

namespace akreon_native::python {

/**
 * @brief Field descriptor for native Python facade structs.
 *
 * This is the dominant metadata path for now: keep binding code consuming
 * fields through descriptors instead of handwritten `.def_ro/.def_rw` chains.
 * Once compiler reflection is available, this file is the replacement point;
 * C++ internals still must not be reflected into Python.
 */
template <auto Member>
struct Field {
  const char* name;
};

template <typename Binding, auto... Members>
Binding& bind_readonly_fields(Binding& cls, Field<Members>... fields) {
  ((cls.def_ro(fields.name, Members)), ...);
  return cls;
}

template <typename Binding, auto... Members>
Binding& bind_readwrite_fields(Binding& cls, Field<Members>... fields) {
  ((cls.def_rw(fields.name, Members)), ...);
  return cls;
}

} // namespace akreon_native::python
