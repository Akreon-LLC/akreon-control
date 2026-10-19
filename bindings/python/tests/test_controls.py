# Copyright © 2026 Akreon LLC
# SPDX-License-Identifier: Apache-2.0

import re
import tomllib
from pathlib import Path

import numpy as np
import pytest
from akreon_native import version
from akreon_native.controls import (
    SolverConfig,
    solve_h2_polytopic_state_feedback,
    solve_h2_polytopic_state_observer,
    solve_h2_state_feedback,
    solve_h2_state_observer,
    solve_hinf_polytopic_state_feedback,
    solve_hinf_polytopic_state_observer,
    solve_hinf_state_feedback,
    solve_hinf_state_observer,
    solve_polytopic_state_feedback,
    solve_polytopic_state_observer,
    solve_state_feedback,
    solve_state_observer,
)


def test_release_versions_match():
    root = Path(__file__).resolve().parents[3]
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    expected = tuple(int(part) for part in project["version"].split("."))
    cmake = (root / "CMakeLists.txt").read_text()
    assert f"project(AkreonControl VERSION {project['version']} " in cmake
    header = (root / "bindings/c/include/akreon_native/core.h").read_text()
    header_version = tuple(
        int(re.search(rf"#define AK_NATIVE_VERSION_{part} (\d+)", header)[1])
        for part in ("MAJOR", "MINOR", "PATCH")
    )
    assert header_version == expected
    assert version() == expected


def direct_feedback_case():
    A = -np.diag([1.0, 1.4, 1.8, 2.2]).astype(np.float64)
    B = np.array([[1.0], [0.5], [0.25], [0.1]], dtype=np.float64)
    return A, B


def generalized_feedback_case(feedthrough=0.0):
    A, B_control = direct_feedback_case()
    B_exogenous = np.array([[0.2], [0.0], [0.1], [0.0]], dtype=np.float64)
    B = np.hstack([B_exogenous, B_control])
    C = np.vstack([np.eye(4, dtype=np.float64), np.zeros((1, 4))])
    D = np.zeros((5, 2), dtype=np.float64)
    D[0, 0] = feedthrough
    D[4, 1] = 1.0
    return A, B, C, D


def observer_case():
    A = -np.diag([0.2, 0.4, 0.6, 0.8]).astype(np.float64)
    C = np.eye(4, dtype=np.float64)[:2]
    return A, C


def generalized_observer_case(feedthrough=0.0):
    A, _, _, _ = generalized_feedback_case()
    B = np.array(
        [
            [0.2, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    C = np.vstack([np.eye(4, dtype=np.float64), np.eye(4, dtype=np.float64)[:2]])
    D = np.zeros((6, 3), dtype=np.float64)
    D[0, 0] = feedthrough
    D[4, 1] = 1.0
    D[5, 2] = 1.0
    return A, B, C, D


def polytopic_pair(A, B):
    A_vertices = np.stack([A, A - 0.05 * np.eye(A.shape[0], dtype=np.float64)])
    B_vertices = np.stack([B, B])
    return A_vertices, B_vertices


def assert_canonical_trace_has_cones(result):
    trace = np.asarray(result.canonical_trace)
    assert trace.dtype == np.int64
    assert trace.ndim == 2
    assert trace.shape[1] == 9
    assert trace.shape[0] > 0
    assert trace[trace[:, 0] == 0].shape[0] > 0
    cone_records = trace[trace[:, 0] == 1]
    assert cone_records.shape[0] > 0
    assert len(result.cone_primal_values) == len(result.cone_dual_values)
    assert len(result.cone_primal_values) == len(result.cone_residual_values)


def test_solve_state_feedback_returns_typed_result():
    A, B = direct_feedback_case()

    result = solve_state_feedback(A, B, config=SolverConfig())

    assert result.K.shape == (1, 4)
    assert result.K.dtype == np.float64
    assert result.status == "OPTIMAL"
    assert result.event == "NONE"
    assert result.stalled is False
    assert result.num_iter > 0
    assert np.isfinite(result.normalized_duality_gap)
    assert np.isfinite(result.primal_objective_value)
    assert np.isfinite(result.dual_objective_value)
    assert np.isfinite(result.dual_residual_norm)
    assert np.isfinite(result.primal_residual_norm)
    assert np.isfinite(result.equality_residual_norm)
    assert_canonical_trace_has_cones(result)
    closed_loop = A + B @ result.K
    assert np.max(np.real(np.linalg.eigvals(closed_loop))) < 0.0


def test_non_contiguous_arrays_are_normalized():
    A_full = np.asfortranarray(-np.eye(4, dtype=np.float32))
    A = A_full[:, :]
    B = np.ones((4, 1), dtype=np.float32)

    result = solve_state_feedback(A, B)

    assert result.status == "OPTIMAL"
    assert result.K.dtype == np.float64


def test_h2_and_hinf_state_feedback_return_gain_and_metric():
    A, B, C, D_h2 = generalized_feedback_case()
    _, _, _, D_hinf = generalized_feedback_case(feedthrough=0.1)

    h2 = solve_h2_state_feedback(A, B, C, D_h2, exogenous_dim=1)
    assert h2.K.shape == (1, 4)
    assert h2.status == "OPTIMAL"
    assert h2.objective_value > 0.0

    hinf = solve_hinf_state_feedback(A, B, C, D_hinf, exogenous_dim=1)
    assert hinf.K.shape == (1, 4)
    assert hinf.status == "OPTIMAL"
    assert hinf.gamma > 0.0
    assert_canonical_trace_has_cones(hinf)


def test_state_observer_and_generalized_observers_return_gain():
    A, C = observer_case()
    direct = solve_state_observer(A, C)
    assert direct.L.shape == (4, 2)
    assert direct.status == "OPTIMAL"
    assert np.max(np.real(np.linalg.eigvals(A - direct.L @ C))) < 0.0

    A_gen, B_gen, C_gen, D_h2 = generalized_observer_case()
    h2 = solve_h2_state_observer(A_gen, B_gen, C_gen, D_h2, performance_dim=4)
    assert h2.L.shape == (4, 2)
    assert h2.status == "OPTIMAL"
    assert h2.objective_value > 0.0

    _, _, _, D_hinf = generalized_observer_case()
    hinf = solve_hinf_state_observer(A_gen, B_gen, C_gen, D_hinf, performance_dim=4)
    assert hinf.L.shape == (4, 2)
    assert hinf.status == "OPTIMAL"
    assert hinf.gamma > 0.0


def test_polytopic_controls_return_gains():
    A, B, C, D_h2 = generalized_feedback_case()
    A_vertices, B_vertices = polytopic_pair(A, B)
    C_vertices = np.stack([C, C])
    D_h2_vertices = np.stack([D_h2, D_h2])

    h2_feedback = solve_h2_polytopic_state_feedback(
        A_vertices, B_vertices, C_vertices, D_h2_vertices, exogenous_dim=1
    )
    assert h2_feedback.K.shape == (1, 4)
    assert h2_feedback.status == "OPTIMAL"
    assert h2_feedback.objective_value > 0.0

    A_hinf, B_hinf, C_hinf, D_hinf = generalized_feedback_case(feedthrough=0.1)
    hinf_feedback = solve_hinf_polytopic_state_feedback(
        np.stack([A_hinf, A_hinf - 0.05 * np.eye(4, dtype=np.float64)]),
        np.stack([B_hinf, B_hinf]),
        np.stack([C_hinf, C_hinf]),
        np.stack([D_hinf, D_hinf]),
        exogenous_dim=1,
    )
    assert hinf_feedback.K.shape == (1, 4)
    assert hinf_feedback.status == "OPTIMAL"
    assert hinf_feedback.gamma > 0.0

    A_go, B_go, C_go, D_go = generalized_observer_case()
    A_go_vertices, B_go_vertices = polytopic_pair(A_go, B_go)
    C_go_vertices = np.stack([C_go, C_go])
    D_go_vertices = np.stack([D_go, D_go])

    h2_observer = solve_h2_polytopic_state_observer(
        A_go_vertices,
        B_go_vertices,
        C_go_vertices,
        D_go_vertices,
        performance_dim=4,
    )
    assert h2_observer.L.shape == (4, 2)
    assert h2_observer.status == "OPTIMAL"
    assert h2_observer.objective_value > 0.0

    _, B_hinf_go, C_hinf_go, D_hinf_go = generalized_observer_case()
    hinf_observer_config = SolverConfig()
    hinf_observer_config.global_reg_eps = 1e-9
    hinf_observer_config.dual_feas_tol = 2e-3
    hinf_observer = solve_hinf_polytopic_state_observer(
        np.stack([A_hinf, A_hinf]),
        np.stack([B_hinf_go, B_hinf_go]),
        np.stack([C_hinf_go, C_hinf_go]),
        np.stack([D_hinf_go, D_hinf_go]),
        performance_dim=4,
        config=hinf_observer_config,
    )
    assert hinf_observer.L.shape == (4, 2)
    assert hinf_observer.status == "OPTIMAL"
    assert hinf_observer.gamma > 0.0


def test_direct_polytopic_feedback_and_objectives():
    A, B = direct_feedback_case()
    A_vertices, B_vertices = polytopic_pair(A, B)

    feasible = solve_polytopic_state_feedback(A_vertices, B_vertices)
    assert feasible.K.shape == (1, 4)
    assert feasible.status == "OPTIMAL"
    assert np.isnan(feasible.objective_value)

    linear = solve_polytopic_state_feedback(
        A_vertices,
        B_vertices,
        p_weights=np.r_[1.0, np.zeros(9)].astype(np.float64),
        y_weights=np.zeros(4, dtype=np.float64),
    )
    assert linear.K.shape == (1, 4)
    assert linear.status == "OPTIMAL"
    assert np.isfinite(linear.objective_value)

    quadratic = solve_polytopic_state_feedback(
        A_vertices,
        B_vertices,
        quadratic_P=np.eye(14, dtype=np.float64),
        quadratic_q=np.zeros(14, dtype=np.float64),
    )
    assert quadratic.K.shape == (1, 4)
    assert quadratic.status == "OPTIMAL"
    assert quadratic.objective_value >= 0.0


def test_direct_polytopic_observer_and_quadratic_objective():
    A, C = observer_case()
    A_vertices, C_vertices = polytopic_pair(A, C)

    feasible = solve_polytopic_state_observer(A_vertices, C_vertices)
    assert feasible.L.shape == (4, 2)
    assert feasible.status == "OPTIMAL"
    assert np.isnan(feasible.objective_value)

    quadratic = solve_polytopic_state_observer(
        A_vertices,
        C_vertices,
        quadratic_P=np.eye(18, dtype=np.float64),
        quadratic_q=np.zeros(18, dtype=np.float64),
    )
    assert quadratic.L.shape == (4, 2)
    assert quadratic.status == "OPTIMAL"
    assert quadratic.objective_value >= 0.0


def test_direct_polytopic_objective_validation():
    A, B = direct_feedback_case()
    A_vertices, B_vertices = polytopic_pair(A, B)

    with pytest.raises(ValueError, match="cannot be mixed"):
        solve_polytopic_state_feedback(
            A_vertices,
            B_vertices,
            p_weights=np.ones(10, dtype=np.float64),
            y_weights=np.zeros(4, dtype=np.float64),
            quadratic_P=np.eye(14, dtype=np.float64),
            quadratic_q=np.zeros(14, dtype=np.float64),
        )
    with pytest.raises(ValueError, match="p_weights and y_weights"):
        solve_polytopic_state_feedback(
            A_vertices, B_vertices, p_weights=np.ones(10, dtype=np.float64)
        )
    with pytest.raises(ValueError, match="INVALID_ARGUMENT"):
        solve_polytopic_state_feedback(
            A_vertices,
            B_vertices,
            quadratic_P=np.eye(13, dtype=np.float64),
            quadratic_q=np.zeros(13, dtype=np.float64),
        )


@pytest.mark.parametrize(
    ("A", "B", "message"),
    [
        (
            np.array([[np.nan]], dtype=np.float64),
            np.array([[1.0]], dtype=np.float64),
            "state feedback solve failed",
        ),
        (
            np.ones((2, 3), dtype=np.float64),
            np.ones((2, 1), dtype=np.float64),
            "expected A to be square",
        ),
        (
            np.eye(4, dtype=np.float64),
            np.ones((3, 1), dtype=np.float64),
            "expected B.shape\\[0\\] == A.shape\\[0\\]",
        ),
        (
            np.eye(4, dtype=np.float64),
            np.ones((4, 2), dtype=np.float64),
            ("DIMENSION_MISMATCH.*A.shape=\\(4, 4\\).*B.shape=\\(4, 2\\)"),
        ),
    ],
)
def test_invalid_inputs_raise_value_error(A, B, message):
    with pytest.raises(ValueError, match=message):
        solve_state_feedback(A, B)


def test_solver_config_steps_must_match_compiled_budget():
    A, B = direct_feedback_case()
    config = SolverConfig()
    config.steps += 1

    with pytest.raises(ValueError, match="INVALID_ARGUMENT"):
        solve_state_feedback(A, B, config=config)


def test_generalized_controls_validate_compiled_dimensions():
    A = np.eye(4, dtype=np.float64)
    B = np.ones((4, 4), dtype=np.float64)
    C = np.ones((1, 4), dtype=np.float64)
    D = np.ones((1, 4), dtype=np.float64)

    with pytest.raises(ValueError, match="DIMENSION_MISMATCH"):
        solve_h2_state_feedback(A, B, C, D, exogenous_dim=1)


def test_unsupported_polytopic_vertex_count_is_rejected_safely():
    A, B, C, D = generalized_feedback_case()
    A_vertices = np.stack([A] * 4)
    B_vertices = np.stack([B] * 4)

    with pytest.raises(ValueError, match="DIMENSION_MISMATCH"):
        solve_polytopic_state_feedback(A_vertices, B_vertices[:, :, 1:])

    with pytest.raises(ValueError, match="DIMENSION_MISMATCH"):
        solve_h2_polytopic_state_feedback(
            A_vertices,
            B_vertices,
            np.stack([C] * 4),
            np.stack([D] * 4),
            exogenous_dim=1,
        )


@pytest.mark.parametrize(
    ("states", "disturbances", "performance"), [(4, 2, 4), (7, 1, 5), (9, 1, 4)]
)
def test_hinf_feedback_rejects_uncompiled_shape_combinations(
    states, disturbances, performance
):
    A = -np.eye(states, dtype=np.float64)
    B = np.ones((states, disturbances + 1), dtype=np.float64)
    C = np.ones((performance, states), dtype=np.float64)
    D = np.zeros((performance, disturbances + 1), dtype=np.float64)
    with pytest.raises(ValueError, match="UNSUPPORTED_DIMENSION"):
        solve_hinf_state_feedback(A, B, C, D, exogenous_dim=disturbances)
