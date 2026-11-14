"""Figures: learning curves (Fig. 3 style) and deployment maps (Figs. 4-5 style)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from numpy.typing import ArrayLike  # noqa: E402

from cfisac.analysis import Curve  # noqa: E402
from cfisac.scenario import ScenarioConfig  # noqa: E402

# Fixed colour per algorithm (colour follows the entity, never its rank in a figure).
ALGO_COLORS = {
    "sac": "#2a78d6",
    "ddpg": "#eb6834",
    "td3": "#1baf7a",
    "a2c": "#eda100",
    "ppo": "#e87ba4",
}
ALGO_ORDER = ("sac", "ddpg", "td3", "a2c", "ppo")
_SURFACE = "#fcfcfb"
_TEXT = "#0b0b0b"
_TEXT_SECONDARY = "#52514e"
_GRID = "#e4e3df"
_TX_COLOR, _RX_COLOR, _UE_COLOR = "#2a78d6", "#eb6834", "#1baf7a"


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(_SURFACE)
    ax.grid(True, color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)
    ax.tick_params(colors=_TEXT_SECONDARY, labelsize=9)
    ax.xaxis.label.set_color(_TEXT_SECONDARY)
    ax.yaxis.label.set_color(_TEXT_SECONDARY)
    ax.title.set_color(_TEXT)


def plot_learning_curves(
    curves: dict[str, Curve],
    path: str | Path,
    title: str = "",
    ylabel: str = "Accumulative reward per episode",
    band: str = "std",
) -> Path:
    """Mean learning curve per algorithm with a spread band across seeds (Fig. 3 style).

    Args:
        curves: ``algo -> Curve`` from :func:`cfisac.analysis.aggregate_runs`.
        path: output image file.
        title: figure title.
        ylabel: y-axis label.
        band: ``"std"`` (mean +/- one standard deviation) or ``"range"`` (min-max over seeds).
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    fig.patch.set_facecolor(_SURFACE)
    ordered = [a for a in ALGO_ORDER if a in curves] + sorted(set(curves) - set(ALGO_ORDER))
    for algo in ordered:
        curve = curves[algo]
        color = ALGO_COLORS.get(algo, _TEXT_SECONDARY)
        if curve.n_seeds > 1:
            low, high = (
                (curve.mean - curve.std, curve.mean + curve.std)
                if band == "std"
                else (curve.low, curve.high)
            )
            ax.fill_between(curve.steps, low, high, color=color, alpha=0.14, linewidth=0)
        label = f"{algo.upper()} (n={curve.n_seeds})"
        ax.plot(curve.steps, curve.mean, color=color, linewidth=2.0, label=label,
                solid_capstyle="round", solid_joinstyle="round")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=11, loc="left")
    _style(ax)
    legend = fig.legend(
        frameon=False, fontsize=9, loc="lower center", ncol=len(ordered), handlelength=1.6,
        columnspacing=1.4,
    )
    for text in legend.get_texts():
        text.set_color(_TEXT)
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 1.0))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def _spread_duplicates(points: np.ndarray, radius: float) -> np.ndarray:
    """Offset coincident markers on a small circle so that stacked APs stay visible."""
    out = points.astype(float).copy()
    keys: dict[tuple[float, float], list[int]] = {}
    for i, p in enumerate(points):
        keys.setdefault((round(float(p[0]), 6), round(float(p[1]), 6)), []).append(i)
    for members in keys.values():
        if len(members) > 1:
            for j, i in enumerate(members):
                angle = 2.0 * np.pi * j / len(members)
                out[i] += radius * np.array([np.cos(angle), np.sin(angle)])
    return out


def plot_deployment(
    scenario: ScenarioConfig,
    tx: ArrayLike,
    rx: ArrayLike,
    path: str | Path,
    title: str = "",
    n_ue_samples: int = 40,
    seed: int = 0,
    spread: float = 1.2,
) -> Path:
    """Deployment map: APs, sampled UE positions and the target trajectory (Figs. 4-5 style).

    APs that share a site are drawn slightly apart (``spread`` metres) for visibility only;
    the caption states the number of APs per role.
    """
    tx_arr = np.asarray(tx, dtype=float).reshape(-1, 2)
    rx_arr = np.asarray(rx, dtype=float).reshape(-1, 2)
    both = _spread_duplicates(np.concatenate([tx_arr, rx_arr]), spread)
    tx_plot, rx_plot = both[: len(tx_arr)], both[len(tx_arr) :]
    ues = scenario.sample_ues(np.random.default_rng(seed), n_ue_samples).reshape(-1, 2)
    angles = np.linspace(0.0, 2.0 * np.pi, 361)
    cx, cy = scenario.target_center

    fig, ax = plt.subplots(figsize=(4.8, 5.3), dpi=150)
    fig.patch.set_facecolor(_SURFACE)
    ax.plot(
        cx + scenario.target_radius * np.cos(angles),
        cy + scenario.target_radius * np.sin(angles),
        color=_TEXT_SECONDARY, linewidth=1.2, label="Target trajectory", zorder=2,
    )
    traj = scenario.trajectory()
    ax.scatter(traj[:, 0], traj[:, 1], s=10, color=_TEXT_SECONDARY, zorder=3, linewidths=0)
    ax.scatter(ues[:, 0], ues[:, 1], s=26, color=_UE_COLOR, alpha=0.22, linewidths=0,
               label="UE samples", zorder=1)
    ax.scatter(tx_plot[:, 0], tx_plot[:, 1], s=90, marker="P", color=_TX_COLOR,
               edgecolors=_SURFACE, linewidths=1.2, label=f"Tx AP (M={len(tx_arr)})", zorder=5)
    ax.scatter(rx_plot[:, 0], rx_plot[:, 1], s=80, marker="X", color=_RX_COLOR,
               edgecolors=_SURFACE, linewidths=1.2, label=f"Rx AP (N={len(rx_arr)})", zorder=6)
    margin = scenario.obs_margin
    ax.set_xlim(scenario.x_min - margin, scenario.x_max + margin)
    ax.set_ylim(scenario.y_min - margin, scenario.y_max + margin)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    if title:
        ax.set_title(title, fontsize=10, loc="left")
    _style(ax)
    legend = fig.legend(frameon=False, fontsize=8, loc="lower center", ncol=2)
    for text in legend.get_texts():
        text.set_color(_TEXT)
    fig.tight_layout(rect=(0.0, 0.09, 1.0, 1.0))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out
