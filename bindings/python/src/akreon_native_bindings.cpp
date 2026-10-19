/*
 * Copyright © 2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#include <nanobind/nanobind.h>

#include "akreon_native/akreon_native.h"
#include "control_operations.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace akreon_native::python;

NB_MODULE(_akreon_native, m) {
  m.doc() = "Akreon Control Python bindings";
  m.attr("canonical_trace_columns") = nb::make_tuple(
      "record_kind", "atom_type", "cone_kind", "node_id", "parent_id",
      "local_index", "global_index", "offset", "dim");

  bind_solver_config_class(m);
  bind_result_classes(m);
  bind_direct_operations(m);
  bind_generalized_operations(m);

  m.def("version", [] {
    const auto version = ak_native_version();
    return nb::make_tuple(version.major, version.minor, version.patch);
  });
}
