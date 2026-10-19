# Copyright © 2026 Akreon LLC
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from akreon_native.plot import (  # noqa: E402
    _compute_cone_block_diagnostics,
    plot_shadow_price_summary,
)


def _result(primal, dual, residual, records):
    return SimpleNamespace(
        canonical_trace=np.asarray(records, dtype=np.int64),
        cone_primal_values=np.asarray(primal, dtype=float),
        cone_dual_values=np.asarray(dual, dtype=float),
        **(
            {"cone_residual_values": np.asarray(residual, dtype=float)}
            if residual is not None
            else {}
        ),
    )


def _cone_record(cone_kind, global_index, offset, dim):
    return [1, 0, cone_kind, 0, 0, global_index, global_index, offset, dim]


def test_cone_block_diagnostics_compute_expected_metrics():
    result = _result(
        primal=[3.0, 4.0, 2.0],
        dual=[0.0, 5.0, 8.0],
        residual=[0.3, 0.4, 0.01],
        records=[
            _cone_record(1, 0, 0, 2),
            _cone_record(2, 1, 2, 1),
        ],
    )

    rows = _compute_cone_block_diagnostics([("controller", result)])

    assert len(rows) == 2
    assert rows[0]["label"] == "controller | SOC c0 [2]"
    assert rows[0]["primal_norm"] == pytest.approx(5.0)
    assert rows[0]["dual_norm"] == pytest.approx(5.0)
    assert rows[0]["pressure_log"] == pytest.approx(np.log10(6.0))
    assert rows[0]["residual_log"] == pytest.approx(np.log10(0.5 + 1e-12))
    assert rows[0]["complementarity_log"] == pytest.approx(np.log10(20.0 + 1e-12))
    assert all(
        np.isfinite(row[field])
        for row in rows
        for field in (
            "pressure_log",
            "primal_scale_log",
            "dual_scale_log",
            "complementarity_log",
            "residual_log",
            "scale_balance",
        )
    )


def test_shadow_price_summary_handles_zero_values_and_missing_residuals():
    zero_result = _result(
        primal=[0.0, 0.0],
        dual=[0.0, 0.0],
        residual=None,
        records=[_cone_record(0, 0, 0, 2)],
    )

    rows = _compute_cone_block_diagnostics([("zero", zero_result)])
    figure = plot_shadow_price_summary([("zero", zero_result)])

    assert len(rows) == 1
    assert rows[0]["pressure_log"] == 0.0
    assert np.isnan(rows[0]["residual_log"])
    assert figure is not None
    panel_titles = {axis.get_title() for axis in figure.axes}
    assert "Dual pressure by cone block" in panel_titles
    assert "Pressure vs residual burden" in panel_titles
    assert "Cone health diagnostics" in panel_titles
    assert "Primal-dual scale balance" in panel_titles
