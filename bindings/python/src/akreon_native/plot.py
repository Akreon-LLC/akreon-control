# Copyright © 2026 Akreon LLC
# SPDX-License-Identifier: Apache-2.0

import matplotlib.pyplot as plt
import numpy as np

GRID = {"linestyle": "--", "alpha": 0.5}
TRACE_CONE = 1
TRACE_KIND, TRACE_CONE_KIND, TRACE_GLOBAL, TRACE_OFFSET, TRACE_DIM = (
    0,
    2,
    6,
    7,
    8,
)
CONE_LABELS = {0: "R+", 1: "SOC", 2: "PSD", 3: "EXP", 4: "POW", 255: "UNK"}
DIAGNOSTIC_EPS = 1e-12
CONTROLLER_LABELS = {
    "state feedback": "direct state feedback",
    "state observer": "direct state observer",
    "polytopic feedback": "direct polytopic feedback",
    "observer": "direct state observer",
    "polytopic observer": "direct polytopic observer",
    "h2 feedback": "h2 state feedback",
    "h2 observer": "h2 state observer",
    "polytopic h2 feedback": "h2 polytopic state feedback",
    "polytopic h2 observer": "h2 polytopic state observer",
}
LABEL_ABBREVIATIONS = {
    "direct state feedback": "Dir SF",
    "direct polytopic feedback": "Dir PSF",
    "direct state observer": "Dir SO",
    "direct polytopic observer": "Dir PSO",
    "h2 state feedback": "H2 SF",
    "hinf state feedback": "Hinf SF",
    "h2 state observer": "H2 SO",
    "hinf state observer": "Hinf SO",
    "h2 polytopic state feedback": "H2 PSF",
    "hinf polytopic state feedback": "Hinf PSF",
    "h2 polytopic state observer": "H2 PSO",
    "hinf polytopic state observer": "Hinf PSO",
    "state feedback": "Dir SF",
    "polytopic feedback": "Dir PSF",
    "observer": "Dir SO",
    "polytopic observer": "Dir PSO",
}


def _finish(fig, title=None):
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    plt.close(fig)
    return fig


def _grid(*axes, **kwargs):
    for ax in np.ravel(axes):
        ax.grid(True, **(kwargs or GRID))


def _state_axes(count, *, sharex=True):
    rows = int(np.ceil(count / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(11, 3.5 * rows), sharex=sharex)
    axes = np.ravel(np.atleast_1d(axes))
    for ax in axes[count:]:
        ax.axis("off")
    return fig, axes[:count]


def _format_state_axes(axes, labels, xlabel):
    for ax, label in zip(axes, labels):
        ax.axhline(0.0, color="black", lw=0.8)
        ax.set(title=label, xlabel=xlabel)
        _grid(ax)


def _positive(values):
    return np.maximum(np.asarray(values, dtype=float), np.finfo(float).tiny)


def _symlog_linthresh(values):
    values = np.abs(np.asarray(values, dtype=float))
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return 1e-8
    return max(float(np.percentile(values, 15)) * 0.1, 1e-10)


def _format_symlog_tick(value, _position):
    if value == 0.0:
        return "0"
    magnitude = abs(value)
    if 1e-3 <= magnitude < 1e4:
        return f"{value:g}"
    return f"{value:.0e}"


def _configure_symlog_xaxis(ax, linthresh, *, max_major_ticks=7):
    import matplotlib.ticker as mticker

    ax.set_xscale("symlog", linthresh=linthresh)
    major_locator = mticker.SymmetricalLogLocator(
        base=10.0,
        linthresh=linthresh,
        subs=(1.0,),
    )
    major_locator.set_params(numticks=max_major_ticks)
    ax.xaxis.set_major_locator(major_locator)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_format_symlog_tick))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax.tick_params(axis="x", which="major", labelsize=8, rotation=30)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")


def _configure_log_xaxis(ax, *, max_major_ticks=7):
    import matplotlib.ticker as mticker

    ax.set_xscale("log")
    major_locator = mticker.LogLocator(base=10.0, numticks=max_major_ticks)
    ax.xaxis.set_major_locator(major_locator)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_format_symlog_tick))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax.tick_params(axis="x", which="major", labelsize=8, rotation=30)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")


def _slice(values, offset, dim):
    values = np.asarray(values, dtype=float)
    if offset < 0 or dim <= 0 or offset + dim > values.size:
        return np.empty(0, dtype=float)
    return values[offset : offset + dim]


def _controller_label(label):
    return CONTROLLER_LABELS.get(str(label).strip().lower(), str(label))


def _short_controller(label):
    label = _controller_label(label)
    return LABEL_ABBREVIATIONS.get(label.lower(), label)


def plot_poles(poles, title="Pole Plot"):
    poles = np.asarray(poles, dtype=complex).reshape(-1)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(poles.real, poles.imag, marker="x", color="red", s=100)
    ax.axhline(0, color="black", lw=1)
    ax.axvline(0, color="black", lw=1)
    ax.set(xlabel="Real Axis", ylabel="Imaginary Axis", title=title)
    ax.axis("equal")
    _grid(ax, alpha=0.7)
    return _finish(fig)


def plot_state_step_responses(systems, labels, state_labels, T=None):
    import control as ct

    if T is None:
        T = np.linspace(0.0, 5.0, 500)
    elif np.isscalar(T):
        T = np.linspace(0.0, float(T), 1000)

    fig, axes = _state_axes(len(state_labels))
    for sys, label in zip(systems, labels):
        tout, yout = ct.step_response(sys, T=T)
        for idx, ax in enumerate(axes):
            ax.plot(tout, np.squeeze(yout)[idx], label=label)

    _format_state_axes(axes, state_labels, "time [s]")
    axes[0].legend()
    return _finish(fig, "Closed-loop state step responses")


def plot_bode_responses(
    systems,
    labels,
    output_labels,
    *,
    omega=None,
    input_index=0,
    title="Closed-loop Bode response",
):
    import control as ct

    omega = np.logspace(-2, 2, 500) if omega is None else omega
    fig, axes = plt.subplots(
        len(output_labels), 2, figsize=(11, 2.8 * len(output_labels)), sharex=True
    )
    axes = np.atleast_2d(axes)

    for sys, label in zip(systems, labels):
        mag, phase, freq = ct.frequency_response(sys, omega)
        for i, output_label in enumerate(output_labels):
            magnitude = np.maximum(mag[i, input_index, :], np.finfo(float).tiny)
            axes[i, 0].semilogx(freq, 20.0 * np.log10(magnitude), label=label)
            axes[i, 1].semilogx(
                freq, np.unwrap(phase[i, input_index, :]) * 180.0 / np.pi, label=label
            )
            axes[i, 0].set_ylabel(f"{output_label}\nmag [dB]")
            axes[i, 1].set_ylabel("phase [deg]")

    for ax in axes[-1]:
        ax.set_xlabel("frequency [rad/s]")
    axes[0, 0].legend()
    _grid(axes, which="both")
    return _finish(fig, title)


def plot_state_trajectories(trajectories, labels, state_labels, title):
    fig, axes = _state_axes(len(state_labels))
    for trajectory, label in zip(trajectories, labels):
        time = np.arange(trajectory.shape[0])
        for idx, ax in enumerate(axes):
            ax.plot(time, trajectory[:, idx], label=label)

    _format_state_axes(axes, state_labels, "step")
    axes[0].legend()
    return _finish(fig, title)


def plot_rollout_states(rollouts, labels, state_labels, title):
    return plot_state_trajectories(
        [rollout["states"] for rollout in rollouts], labels, state_labels, title
    )


def plot_rollout_channel_psd(
    rollouts,
    labels,
    channels,
    *,
    dt=0.02,
    title="Rollout channel spectrum",
):
    from scipy.signal import welch

    fig, axes = plt.subplots(1, len(channels), figsize=(5.5 * len(channels), 4))
    axes = np.atleast_1d(axes)
    for rollout, label in zip(rollouts, labels):
        states = rollout["states"]
        for ax, (idx, channel_label) in zip(axes, channels):
            freq, psd = welch(
                states[:, idx], fs=1.0 / dt, nperseg=min(256, len(states))
            )
            ax.semilogy(freq, psd, label=label)
            ax.set(title=channel_label, xlabel="frequency [Hz]")
            _grid(ax)

    axes[0].set_ylabel("PSD")
    axes[0].legend()
    return _finish(fig, title)


def _canonical_trace(result):
    trace = np.asarray(result.canonical_trace, dtype=np.int64)
    return trace if trace.ndim == 2 and trace.shape[1] == 9 else np.empty((0, 9), int)


def _compute_cone_block_diagnostics(results, eps=DIAGNOSTIC_EPS):
    """Compute aggregate diagnostics for each serialized cone-vector slice."""
    diagnostics = []
    for controller_label, result in results:
        controller = _short_controller(controller_label)
        primal_values = getattr(result, "cone_primal_values", ())
        dual_values = getattr(result, "cone_dual_values", ())
        residual_values = getattr(result, "cone_residual_values", None)

        for record in _canonical_trace(result):
            if record[TRACE_KIND] != TRACE_CONE:
                continue

            cone_index = int(record[TRACE_GLOBAL])
            cone_kind = int(record[TRACE_CONE_KIND])
            cone_type = CONE_LABELS.get(cone_kind, "UNK")
            offset, dim = int(record[TRACE_OFFSET]), int(record[TRACE_DIM])
            block_label = f"{cone_type} c{cone_index} [{dim}]"
            label = f"{controller} | {block_label}" if controller else block_label

            primal = _slice(primal_values, offset, dim)
            dual = _slice(dual_values, offset, dim)
            residual = (
                _slice(residual_values, offset, dim)
                if residual_values is not None
                else np.empty(0, dtype=float)
            )

            primal_norm = (
                float(np.linalg.norm(primal)) if primal.size == dim else np.nan
            )
            dual_norm = float(np.linalg.norm(dual)) if dual.size == dim else np.nan
            # No global residual is substituted here: it cannot be attributed to a
            # cone block. Missing per-block residual data is represented as NaN.
            residual_norm = (
                float(np.linalg.norm(residual)) if residual.size == dim else np.nan
            )
            complementarity = (
                abs(float(np.dot(primal, dual)))
                if primal.size == dim and dual.size == dim
                else np.nan
            )

            diagnostics.append(
                {
                    "label": label,
                    "cone_type": cone_type,
                    "offset": offset,
                    "dim": dim,
                    "primal_norm": primal_norm,
                    "dual_norm": dual_norm,
                    "pressure_log": np.log10(1.0 + dual_norm),
                    "primal_scale_log": np.log10(eps + primal_norm),
                    "dual_scale_log": np.log10(eps + dual_norm),
                    "complementarity_log": np.log10(eps + complementarity),
                    "residual_log": np.log10(eps + abs(residual_norm)),
                    "scale_balance": np.log10((eps + dual_norm) / (eps + primal_norm)),
                }
            )
    return diagnostics


def _annotate_extreme_cones(ax, diagnostics, *, limit=3):
    finite = [
        (idx, row)
        for idx, row in enumerate(diagnostics)
        if np.isfinite(row["residual_log"]) and np.isfinite(row["pressure_log"])
    ]
    if not finite:
        return

    points = np.asarray(
        [[row["residual_log"], row["pressure_log"]] for _, row in finite]
    )
    center = np.median(points, axis=0)
    spread = np.ptp(points, axis=0)
    spread[spread == 0.0] = 1.0
    scores = np.linalg.norm((points - center) / spread, axis=1)
    ranked = np.argsort(scores)[-min(limit, len(finite)) :]
    for annotation_idx, ranked_idx in enumerate(ranked):
        _, row = finite[int(ranked_idx)]
        annotate_left = row["residual_log"] >= center[0]
        ax.annotate(
            row["label"],
            (row["residual_log"], row["pressure_log"]),
            xytext=(-6 if annotate_left else 6, 6 + 11 * annotation_idx),
            textcoords="offset points",
            fontsize=7,
            horizontalalignment="right" if annotate_left else "left",
        )


def plot_shadow_price_summary(results, title="Shadow price diagnostics"):
    """Plot block-level shadow-price and solver-health proxy diagnostics.

    Dual pressure is ``log10(1 + ||y_i||_2)`` for each serialized cone block.
    Solver-health panels use block primal/dual norms, residual burden, raw
    complementarity, and scale balance. True cone tightness or margin is
    intentionally not computed because cone-internal layout is not serialized.
    """
    diagnostics = _compute_cone_block_diagnostics(results)
    if not diagnostics:
        print("No cone diagnostics are available for these results.")
        return None

    diagnostics.sort(
        key=lambda row: (
            not np.isfinite(row["pressure_log"]),
            -row["pressure_log"] if np.isfinite(row["pressure_log"]) else 0.0,
        )
    )
    labels = [row["label"] for row in diagnostics]
    y = np.arange(len(diagnostics))
    height = max(9.0, 0.38 * len(diagnostics) + 5.0)
    fig, axs = plt.subplots(2, 2, figsize=(15, height))
    pressure_ax, scatter_ax, health_ax, balance_ax = axs.ravel()

    cone_types = list(dict.fromkeys(row["cone_type"] for row in diagnostics))
    color_map = plt.get_cmap("tab10")
    cone_colors = {
        cone_type: color_map(type_idx % 10)
        for type_idx, cone_type in enumerate(cone_types)
    }

    pressure = np.asarray([row["pressure_log"] for row in diagnostics])
    pressure_plot = np.where(np.isfinite(pressure), pressure, 0.0)
    pressure_ax.barh(
        y,
        pressure_plot,
        height=0.65,
        color=[cone_colors[row["cone_type"]] for row in diagnostics],
    )
    pressure_ax.set(
        title="Dual pressure by cone block",
        xlabel="constraint pressure proxy: log10(1 + ‖yᵢ‖₂)",
    )
    pressure_ax.set_yticks(y, labels=labels, fontsize=8)
    pressure_ax.invert_yaxis()
    _grid(pressure_ax, axis="x", alpha=0.25)

    markers = ("o", "s", "^", "D", "P", "X")
    for type_idx, cone_type in enumerate(cone_types):
        typed = [row for row in diagnostics if row["cone_type"] == cone_type]
        residual_log = np.asarray([row["residual_log"] for row in typed])
        pressure_log = np.asarray([row["pressure_log"] for row in typed])
        finite = np.isfinite(residual_log) & np.isfinite(pressure_log)
        scatter_ax.scatter(
            residual_log[finite],
            pressure_log[finite],
            label=cone_type,
            color=cone_colors[cone_type],
            marker=markers[type_idx % len(markers)],
            s=42,
            alpha=0.8,
        )
    scatter_ax.set(
        title="Pressure vs residual burden",
        xlabel="log10(eps + ‖rₚ,ᵢ‖₂)",
        ylabel="dual pressure proxy",
    )
    if 1 < len(cone_types) <= 6:
        scatter_ax.legend(title="cone", fontsize=8)
    _annotate_extreme_cones(scatter_ax, diagnostics)
    _grid(scatter_ax, alpha=0.25)

    heatmap = np.asarray(
        [
            [
                row["primal_scale_log"],
                row["dual_scale_log"],
                row["complementarity_log"],
                row["residual_log"],
            ]
            for row in diagnostics
        ]
    )
    cmap = plt.get_cmap("viridis").with_extremes(bad="#eeeeee")
    image = health_ax.imshow(np.ma.masked_invalid(heatmap), aspect="auto", cmap=cmap)
    health_ax.set_title("Cone health diagnostics")
    health_ax.set_xticks(
        np.arange(4), labels=("primal scale", "dual scale", "comp.", "residual")
    )
    health_ax.tick_params(axis="x", labelrotation=25)
    health_ax.set_yticks(y, labels=labels, fontsize=8)
    fig.colorbar(image, ax=health_ax, label="log10 diagnostic scale")

    balance = np.asarray([row["scale_balance"] for row in diagnostics])
    balance_plot = np.where(np.isfinite(balance), balance, 0.0)
    balance_colors = np.where(balance >= 0.0, "#e45756", "#72b7b2")
    balance_ax.barh(y, balance_plot, height=0.65, color=balance_colors)
    balance_ax.axvline(0.0, color="black", lw=0.9)
    balance_ax.set(
        title="Primal-dual scale balance",
        xlabel="log10((eps + ‖yᵢ‖₂) / (eps + ‖xᵢ‖₂))",
    )
    balance_ax.set_yticks(y, labels=labels, fontsize=8)
    balance_ax.invert_yaxis()
    _grid(balance_ax, axis="x", alpha=0.25)
    return _finish(fig, title)


def show_rollout_frames(rollout, frame_indices=None):
    frames = rollout.get("frames", [])
    if not frames:
        print("No frames captured. Re-run the rollout with capture_frames=True.")
        return None

    indices = (
        np.linspace(0, len(frames) - 1, 6, dtype=int)
        if frame_indices is None
        else frame_indices
    )
    fig, axes = plt.subplots(1, len(indices), figsize=(14, 3))
    for ax, idx in zip(np.atleast_1d(axes), indices):
        ax.imshow(frames[idx])
        ax.set_title(f"step {idx}")
        ax.axis("off")
    return _finish(fig)
