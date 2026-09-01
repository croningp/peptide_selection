"""
Family-level phase-space copy-number sensitivity for R4.3.

This is the compact version of the A-vs-n phase-space plot. It groups samples
into the four reaction families used in the manuscript:

    CDIaq, CDIs, protease, wet-dry

For each family it plots the geometric mean species count and mean A, with
95% intervals across samples. It also reports the mean copy-number sensitivity
for relative copy numbers constrained to 0.5-5.0 across MA bins.

Outputs:
    phi_phase_space_family_ci.png
    phi_phase_space_family_ci.csv
"""

from __future__ import annotations

import argparse
import ast
import io
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

SRC_DIR = Path(__file__).resolve().parents[1]
PKL = SRC_DIR / "all_experiments_analysis.pkl"
LABBOOK = SRC_DIR / "labbook.csv"
OUT_DIR = Path(__file__).resolve().parent

BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
DARK_GREY = "#4d4d4d"
LIGHT_GREY = "#d9d9d9"
GRID = "#ebebeb"

FAMILY_COLORS = {
    "protease": "#1f77b4",
    "CDIs": "#ff7f0e",
    "CDIaq": "#2ca02c",
    "wet-dry": "#d62728",
}

FAMILY_ORDER = ["CDIaq", "CDIs", "protease", "wet-dry"]
FAMILY_LABELS = {
    "protease": "protease",
    "CDIs": "CDI(s)",
    "CDIaq": "CDI(l)",
    "wet-dry": "wet-dry",
}

FIGURE5H_SERIES = {
    "CDIaq": {"ARM_07_2", "ARM_07_3", "ARM_07_4", "ARM_07_5", "ARM_07_6"},
    "CDIs": {"MYW_20_R", "MYW_21_R", "MYW_21_L"},
    "protease": {
        "MYW_34_R",
        "MYW_34_L",
        "MYW_35_R",
        "MYW_35_L",
    },
    "wet-dry": {"MKJ_58_1", "MKJ_58_2", "MKJ_58_3"},
}


def observed_sequences(observed_nodes):
    if isinstance(observed_nodes, (list, tuple)) and observed_nodes:
        first = observed_nodes[0]
        if isinstance(first, (list, tuple, set, np.ndarray)):
            return list(first)
    return list(observed_nodes)


def labbook_sequence_map() -> dict[str, dict]:
    labbook = pd.read_csv(LABBOOK)
    seq_to_meta = {}
    for _, row in labbook.iterrows():
        try:
            seqs = ast.literal_eval(row["sequences"])
        except Exception:
            continue
        for _, sample_name in seqs.items():
            seq_to_meta[sample_name] = {
                "series": row["experiment_id"],
                "parent": row["parent"],
                "reaction_family": row["experiment_type"],
            }
    return seq_to_meta


def endpoint_copy_number(
    values: np.ndarray,
    min_a: float,
    max_a: float,
    copy_min: float,
    copy_max: float,
    direction: str,
) -> np.ndarray:
    span = max(float(max_a - min_a), 1.0)
    position = (values - min_a) / span
    if direction == "high_ma_high_copy":
        log_c = np.log(copy_min) + position * (np.log(copy_max) - np.log(copy_min))
    elif direction == "low_ma_high_copy":
        log_c = np.log(copy_max) - position * (np.log(copy_max) - np.log(copy_min))
    elif direction == "equal":
        log_c = np.zeros_like(values, dtype=float)
    else:
        raise ValueError(f"Unknown direction: {direction}")
    return np.exp(log_c)


def A_with_copy_number(a_i: np.ndarray, c_i: np.ndarray) -> float:
    return float(np.sum(c_i * np.exp(a_i)) / np.sum(c_i))


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def load_family_points(copy_min: float, copy_max: float, min_a: float, max_a: float):
    seq_map = labbook_sequence_map()
    with open(PKL, "rb") as handle:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            df_exp = pickle.load(handle)

    rows = []
    for _, row in df_exp[df_exp["A"] > 0].iterrows():
        meta = seq_map.get(row["name"])
        if meta is None:
            continue

        seqs = observed_sequences(row["observed_nodes"])
        if not seqs:
            continue
        a_i = np.array([len(seq) - 1 for seq in seqs], dtype=float)
        if len(a_i) == 0:
            continue

        c_low = endpoint_copy_number(a_i, min_a, max_a, copy_min, copy_max, "low_ma_high_copy")
        c_equal = endpoint_copy_number(a_i, min_a, max_a, copy_min, copy_max, "equal")
        c_high = endpoint_copy_number(a_i, min_a, max_a, copy_min, copy_max, "high_ma_high_copy")

        A_equal = A_with_copy_number(a_i, c_equal)
        A_low = A_with_copy_number(a_i, c_low)
        A_high = A_with_copy_number(a_i, c_high)

        rows.append(
            {
                "reaction_family": meta["reaction_family"],
                "series": meta["series"],
                "experiment": row["name"],
                "n_species": int(len(a_i)),
                "A_paper": round(float(row["A"]), 6),
                "A_equal": round(A_equal, 6),
                "A_low_ma_high_copy": round(A_low, 6),
                "A_high_ma_high_copy": round(A_high, 6),
                "copy_interval_low": round(min(A_low, A_high), 6),
                "copy_interval_high": round(max(A_low, A_high), 6),
                "delta_low_pct": round((A_low - A_equal) / A_equal * 100.0, 6),
                "delta_high_pct": round((A_high - A_equal) / A_equal * 100.0, 6),
            }
        )
    return pd.DataFrame(rows)


def summarise_by_family(points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family in FAMILY_ORDER:
        sub = points[points["reaction_family"] == family].copy()
        if sub.empty:
            continue

        n_values = sub["n_species"].to_numpy(float)
        log_n = np.log(n_values)
        n_geo = float(np.exp(log_n.mean()))
        n_ci_low, n_ci_high = np.exp(percentile_interval(log_n))
        A_values = sub["A_equal"].to_numpy(float)
        A_ci_low, A_ci_high = percentile_interval(A_values)

        rows.append(
            {
                "reaction_family": family,
                "n_samples": len(sub),
                "n_series": sub["series"].nunique(),
                "n_geomean": n_geo,
                "n_ci95_low": n_ci_low,
                "n_ci95_high": n_ci_high,
                "A_equal_mean": float(A_values.mean()),
                "A_equal_ci95_low": A_ci_low,
                "A_equal_ci95_high": A_ci_high,
                "copy_interval_low_mean": float(sub["copy_interval_low"].mean()),
                "copy_interval_high_mean": float(sub["copy_interval_high"].mean()),
                "delta_low_pct_mean": float(sub["delta_low_pct"].mean()),
                "delta_high_pct_mean": float(sub["delta_high_pct"].mean()),
            }
        )
    return pd.DataFrame(rows)


def filter_figure5h_points(points: pd.DataFrame) -> pd.DataFrame:
    keep = np.zeros(len(points), dtype=bool)
    for family, series_names in FIGURE5H_SERIES.items():
        keep |= (
            (points["reaction_family"] == family)
            & (points["series"].isin(series_names))
        ).to_numpy()
    return points.loc[keep].copy()


def make_plot(points: pd.DataFrame, summary: pd.DataFrame, output_prefix: str, copy_min: float, copy_max: float):
    fig, ax = plt.subplots(figsize=(6.5, 6.25))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for family in FAMILY_ORDER:
        sub = points[points["reaction_family"] == family]
        if sub.empty:
            continue
        color = FAMILY_COLORS[family]
        ax.scatter(
            sub["n_species"],
            sub["A_equal"],
            s=48,
            color=color,
            alpha=0.70,
            edgecolor="#1f1f1f",
            linewidth=1.15,
            zorder=1,
            label=FAMILY_LABELS[family],
        )

    for _, row in summary.iterrows():
        family = row["reaction_family"]
        color = FAMILY_COLORS[family]
        x = row["n_geomean"]
        y = row["A_equal_mean"]
        yerr = np.array([[y - row["A_equal_ci95_low"]], [row["A_equal_ci95_high"] - y]])

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="D",
            markersize=9.5,
            color=color,
            ecolor="#1f1f1f",
            elinewidth=1.7,
            capsize=4.5,
            markeredgecolor="#1f1f1f",
            markeredgewidth=1.35,
            zorder=7,
        )

        # Vertical copy-number sensitivity interval at the family mean location.
        ax.vlines(
            x,
            row["copy_interval_low_mean"],
            row["copy_interval_high_mean"],
            color=DARK_GREY,
            linewidth=2.2,
            alpha=0.85,
            zorder=6,
        )
        ax.scatter(
            x,
            row["copy_interval_low_mean"],
            s=34,
            color=BLUE,
            edgecolor="#1f1f1f",
            linewidth=0.7,
            zorder=8,
        )
        ax.scatter(
            x,
            row["copy_interval_high_mean"],
            s=34,
            color=ORANGE,
            edgecolor="#1f1f1f",
            linewidth=0.7,
            zorder=8,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(6, 4200)
    ax.set_ylim(2.8, 230)
    ax.set_xlabel("diversity", fontsize=25)
    ax.set_ylabel("ensemble assembly / A", fontsize=25)
    ax.tick_params(axis="both", which="major", labelsize=24, width=2.4, length=10)
    ax.tick_params(axis="both", which="minor", width=1.7, length=5)
    for spine in ax.spines.values():
        spine.set_linewidth(2.4)

    handles, labels = ax.get_legend_handles_labels()
    # Preserve family order and avoid duplicate labels.
    by_label = dict(zip(labels, handles))
    ordered_labels = [FAMILY_LABELS[fam] for fam in ["protease", "CDIs", "CDIaq", "wet-dry"]]
    leg = ax.legend(
        [by_label[label] for label in ordered_labels if label in by_label],
        [label for label in ordered_labels if label in by_label],
        fontsize=24,
        frameon=True,
        loc="lower right",
        borderpad=0.7,
        handlelength=1.0,
    )
    leg.get_frame().set_linewidth(2.0)

    ax.text(
        0.03,
        0.96,
        "copy-number CI",
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        ha="left",
        color=DARK_GREY,
    )

    fig.tight_layout()
    png_path = OUT_DIR / f"{output_prefix}.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white", transparent=False)
    print(f"Figure saved: {png_path}")


def chi2_radius_2d(confidence: float) -> float:
    """Radius for a 2D Gaussian confidence ellipse."""
    common = {
        0.50: 1.38629436112,
        0.68: 2.27886856638,
        0.80: 3.21887582487,
        0.90: 4.60517018599,
        0.95: 5.99146454711,
    }
    key = round(confidence, 2)
    if key in common:
        return float(np.sqrt(common[key]))
    # Fallback for 2 degrees of freedom: CDF = 1 - exp(-x/2).
    return float(np.sqrt(-2.0 * np.log(max(1.0 - confidence, 1e-9))))


def ellipse_points_logspace(
    x: np.ndarray,
    y: np.ndarray,
    confidence: float,
    n_points: int = 240,
) -> tuple[np.ndarray, np.ndarray]:
    """Return confidence ellipse points calculated in log10(x), log10(y)."""
    log_xy = np.column_stack([np.log10(x), np.log10(y)])
    center = log_xy.mean(axis=0)
    cov = np.cov(log_xy, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-10)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    radius = chi2_radius_2d(confidence)

    theta = np.linspace(0, 2 * np.pi, n_points)
    circle = np.vstack([np.cos(theta), np.sin(theta)])
    ellipse = center[:, None] + vecs @ (np.sqrt(vals)[:, None] * radius * circle)
    return np.power(10.0, ellipse[0]), np.power(10.0, ellipse[1])


def make_mound_plot(
    points: pd.DataFrame,
    summary: pd.DataFrame,
    output_prefix: str,
    confidence: float,
):
    fig, ax = plt.subplots(figsize=(7.4, 6.25))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for family in FAMILY_ORDER:
        sub = points[points["reaction_family"] == family]
        if sub.empty:
            continue
        color = FAMILY_COLORS[family]
        x = sub["n_species"].to_numpy(float)
        y = sub["A_equal"].to_numpy(float)

        if len(sub) >= 3:
            ex, ey = ellipse_points_logspace(x, y, confidence=confidence)
            ax.fill(ex, ey, color=color, alpha=0.055, zorder=0)
            ax.plot(ex, ey, color=color, linewidth=2.6, alpha=0.98, zorder=2)

        ax.scatter(
            x,
            y,
            s=46,
            color=color,
            alpha=0.70,
            edgecolor="#1f1f1f",
            linewidth=1.15,
            zorder=3,
        )

    for _, row in summary.iterrows():
        family = row["reaction_family"]
        color = FAMILY_COLORS[family]
        ax.scatter(
            row["n_geomean"],
            row["A_equal_mean"],
            marker="D",
            s=122,
            color=color,
            edgecolor="#1f1f1f",
            linewidth=1.5,
            zorder=8,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(6, 4200)
    ax.set_ylim(2.8, 230)
    ax.set_xlabel("diversity", fontsize=25)
    ax.set_ylabel("ensemble assembly / A", fontsize=25)
    ax.tick_params(axis="both", which="major", labelsize=24, width=2.4, length=10)
    ax.tick_params(axis="both", which="minor", width=1.7, length=5)
    for spine in ax.spines.values():
        spine.set_linewidth(2.4)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=FAMILY_COLORS[fam],
            markeredgecolor="#1f1f1f",
            markeredgewidth=1.15,
            markersize=13,
            label=FAMILY_LABELS[fam],
        )
        for fam in ["protease", "CDIs", "CDIaq", "wet-dry"]
    ]
    leg = ax.legend(
        handles=handles,
        fontsize=22,
        frameon=True,
        loc="center left",
        bbox_to_anchor=(1.01, 0.50),
        borderpad=0.7,
        handlelength=1.0,
    )
    leg.get_frame().set_linewidth(2.0)

    ax.text(
        0.03,
        0.96,
        f"{int(confidence * 100)}% family regions",
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        ha="left",
        color=DARK_GREY,
    )

    fig.tight_layout()
    png_path = OUT_DIR / f"{output_prefix}_mounds.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white", transparent=False)
    print(f"Mound figure saved: {png_path}")


def trajectory_bound_rows(points: pd.DataFrame, n_bins: int, center: str) -> pd.DataFrame:
    rows = []
    work = points.copy()
    if center == "paper":
        ratio_low = work["copy_interval_low"] / work["A_equal"]
        ratio_high = work["copy_interval_high"] / work["A_equal"]
        work["trajectory_A"] = work["A_paper"]
        work["trajectory_bound_low"] = work["A_paper"] * ratio_low
        work["trajectory_bound_high"] = work["A_paper"] * ratio_high
    elif center == "equal":
        work["trajectory_A"] = work["A_equal"]
        work["trajectory_bound_low"] = work["copy_interval_low"]
        work["trajectory_bound_high"] = work["copy_interval_high"]
    else:
        raise ValueError(f"Unknown trajectory center: {center}")

    for family in FAMILY_ORDER:
        sub = work[work["reaction_family"] == family].copy()
        if sub.empty:
            continue

        sub = sub.sort_values("n_species")
        bins_here = min(n_bins, max(3, len(sub) // 5))
        ranked = sub["n_species"].rank(method="first")
        sub["trajectory_bin"] = pd.qcut(ranked, q=bins_here, labels=False, duplicates="drop")

        for bin_id, bin_df in sub.groupby("trajectory_bin", sort=True):
            n_values = bin_df["n_species"].to_numpy(float)
            rows.append(
                {
                    "reaction_family": family,
                    "family_label": FAMILY_LABELS[family],
                    "trajectory_bin": int(bin_id),
                    "n_points": len(bin_df),
                    "n_geomean": float(np.exp(np.mean(np.log(n_values)))),
                    "n_min": int(bin_df["n_species"].min()),
                    "n_max": int(bin_df["n_species"].max()),
                    "A_center_mean": float(bin_df["trajectory_A"].mean()),
                    "copy_bound_low_mean": float(bin_df["trajectory_bound_low"].mean()),
                    "copy_bound_high_mean": float(bin_df["trajectory_bound_high"].mean()),
                    "delta_low_pct_mean": float(bin_df["delta_low_pct"].mean()),
                    "delta_high_pct_mean": float(bin_df["delta_high_pct"].mean()),
                }
            )
    return pd.DataFrame(rows)


def make_trajectory_bound_plot(
    points: pd.DataFrame,
    output_prefix: str,
    copy_min: float,
    copy_max: float,
    n_bins: int,
    center: str,
):
    trajectory = trajectory_bound_rows(points, n_bins=n_bins, center=center)
    csv_path = OUT_DIR / f"{output_prefix}_trajectory_bounds.csv"
    trajectory.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(7.7, 6.25))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    handles = []
    for family in ["protease", "CDIs", "CDIaq", "wet-dry"]:
        sub = trajectory[trajectory["reaction_family"] == family].sort_values("n_geomean")
        if sub.empty:
            continue

        color = FAMILY_COLORS[family]
        x = sub["n_geomean"].to_numpy(float)
        y = sub["A_center_mean"].to_numpy(float)
        y_low = sub["copy_bound_low_mean"].to_numpy(float)
        y_high = sub["copy_bound_high_mean"].to_numpy(float)

        ax.fill_between(x, y_low, y_high, color=color, alpha=0.16, linewidth=0, zorder=1)
        ax.plot(x, y_low, color=color, linewidth=1.25, alpha=0.58, zorder=2)
        ax.plot(x, y_high, color=color, linewidth=1.25, alpha=0.58, zorder=2)
        line = ax.plot(
            x,
            y,
            "-o",
            color=color,
            markerfacecolor=color,
            markeredgecolor="#1f1f1f",
            markeredgewidth=1.0,
            linewidth=2.5,
            markersize=6.2,
            zorder=4,
            label=FAMILY_LABELS[family],
        )[0]
        handles.append(line)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(6, 4200)
    ax.set_ylim(2.8, 230)
    ax.set_xlabel("diversity", fontsize=25)
    ax.set_ylabel("ensemble assembly / A", fontsize=25)
    ax.tick_params(axis="both", which="major", labelsize=24, width=2.4, length=10)
    ax.tick_params(axis="both", which="minor", width=1.7, length=5)
    for spine in ax.spines.values():
        spine.set_linewidth(2.4)

    leg = ax.legend(
        handles=handles,
        fontsize=17,
        frameon=True,
        loc="lower right",
        borderpad=0.65,
        handlelength=1.35,
    )
    leg.get_frame().set_linewidth(1.7)

    ax.text(
        0.03,
        0.96,
        f"copy-number bounds ({copy_min:g}-{copy_max:g}x)",
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        ha="left",
        color=DARK_GREY,
    )

    fig.tight_layout()
    png_path = OUT_DIR / f"{output_prefix}_trajectory_bounds.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white", transparent=False)
    print(f"Trajectory-bound figure saved: {png_path}")
    print(f"Trajectory-bound CSV saved: {csv_path}")


def make_point_bound_plot(
    points: pd.DataFrame,
    output_prefix: str,
    copy_min: float,
    copy_max: float,
    center: str,
):
    work = points.copy()
    if center == "paper":
        ratio_low = work["copy_interval_low"] / work["A_equal"]
        ratio_high = work["copy_interval_high"] / work["A_equal"]
        work["A_center"] = work["A_paper"]
        work["copy_bound_low"] = work["A_paper"] * ratio_low
        work["copy_bound_high"] = work["A_paper"] * ratio_high
    elif center == "equal":
        work["A_center"] = work["A_equal"]
        work["copy_bound_low"] = work["copy_interval_low"]
        work["copy_bound_high"] = work["copy_interval_high"]
    else:
        raise ValueError(f"Unknown point center: {center}")

    csv_cols = [
        "reaction_family",
        "series",
        "experiment",
        "n_species",
        "A_paper",
        "A_equal",
        "A_center",
        "copy_bound_low",
        "copy_bound_high",
        "A_low_ma_high_copy",
        "A_high_ma_high_copy",
        "delta_low_pct",
        "delta_high_pct",
    ]
    csv_path = OUT_DIR / f"{output_prefix}_point_bounds.csv"
    work[csv_cols].to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(5.4, 5.1))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for family in ["protease", "CDIs", "CDIaq", "wet-dry"]:
        sub = work[work["reaction_family"] == family].sort_values("n_species")
        if sub.empty:
            continue
        color = FAMILY_COLORS[family]

        ax.vlines(
            sub["n_species"],
            sub["copy_bound_low"],
            sub["copy_bound_high"],
            color=color,
            linewidth=1.35,
            alpha=0.30,
            zorder=1,
        )
        ax.scatter(
            sub["n_species"],
            sub["A_center"],
            s=58,
            color=color,
            edgecolor="#1f1f1f",
            linewidth=1.05,
            alpha=0.88,
            zorder=3,
            label=FAMILY_LABELS[family],
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(6, 4000)
    ax.set_ylim(2.8, 150)
    ax.set_xlabel("diversity", fontsize=21)
    ax.set_ylabel("ensemble assembly / A", fontsize=21)
    ax.tick_params(axis="both", which="major", labelsize=19, width=1.7, length=7)
    ax.tick_params(axis="both", which="minor", width=1.25, length=3.8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ordered_labels = [FAMILY_LABELS[fam] for fam in ["protease", "CDIs", "CDIaq", "wet-dry"]]
    leg = ax.legend(
        [by_label[label] for label in ordered_labels if label in by_label],
        [label for label in ordered_labels if label in by_label],
        fontsize=13,
        frameon=True,
        loc="lower right",
        borderpad=0.45,
        handlelength=0.9,
        labelspacing=0.35,
    )
    leg.get_frame().set_linewidth(1.25)

    fig.tight_layout(pad=0.45)
    png_path = OUT_DIR / f"{output_prefix}_point_bounds.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white", transparent=False)
    print(f"Point-bound figure saved: {png_path}")
    print(f"Point-bound CSV saved: {csv_path}")


def point_bound_dataframe(points: pd.DataFrame, center: str) -> pd.DataFrame:
    work = points.copy()
    if center == "paper":
        ratio_low = work["copy_interval_low"] / work["A_equal"]
        ratio_high = work["copy_interval_high"] / work["A_equal"]
        work["A_center"] = work["A_paper"]
        work["copy_bound_low"] = work["A_paper"] * ratio_low
        work["copy_bound_high"] = work["A_paper"] * ratio_high
    elif center == "equal":
        work["A_center"] = work["A_equal"]
        work["copy_bound_low"] = work["copy_interval_low"]
        work["copy_bound_high"] = work["copy_interval_high"]
    else:
        raise ValueError(f"Unknown point center: {center}")
    return work


def make_connected_bound_plot(
    points: pd.DataFrame,
    output_prefix: str,
    copy_min: float,
    copy_max: float,
    center: str,
):
    work = point_bound_dataframe(points, center=center)
    csv_cols = [
        "reaction_family",
        "series",
        "experiment",
        "n_species",
        "A_paper",
        "A_equal",
        "A_center",
        "copy_bound_low",
        "copy_bound_high",
        "A_low_ma_high_copy",
        "A_high_ma_high_copy",
        "delta_low_pct",
        "delta_high_pct",
    ]
    csv_path = OUT_DIR / f"{output_prefix}_connected_bounds.csv"
    work[csv_cols].to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(5.4, 5.1))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for family in ["protease", "CDIs", "CDIaq", "wet-dry"]:
        sub = work[work["reaction_family"] == family].sort_values(["n_species", "A_center"])
        if sub.empty:
            continue

        color = FAMILY_COLORS[family]
        x = sub["n_species"].to_numpy(float)
        y = sub["A_center"].to_numpy(float)
        y_low = sub["copy_bound_low"].to_numpy(float)
        y_high = sub["copy_bound_high"].to_numpy(float)

        ax.fill_between(
            x,
            y_low,
            y_high,
            color=color,
            alpha=0.15,
            linewidth=0,
            zorder=1,
        )
        ax.plot(x, y_low, color=color, linewidth=1.05, alpha=0.45, zorder=2)
        ax.plot(x, y_high, color=color, linewidth=1.05, alpha=0.45, zorder=2)
        ax.scatter(
            x,
            y,
            s=58,
            color=color,
            edgecolor="#1f1f1f",
            linewidth=1.05,
            alpha=0.88,
            zorder=3,
            label=FAMILY_LABELS[family],
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(6, 4000)
    ax.set_ylim(2.8, 150)
    ax.set_xlabel("diversity", fontsize=21)
    ax.set_ylabel("ensemble assembly / A", fontsize=21)
    ax.tick_params(axis="both", which="major", labelsize=19, width=1.7, length=7)
    ax.tick_params(axis="both", which="minor", width=1.25, length=3.8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ordered_labels = [FAMILY_LABELS[fam] for fam in ["protease", "CDIs", "CDIaq", "wet-dry"]]
    leg = ax.legend(
        [by_label[label] for label in ordered_labels if label in by_label],
        [label for label in ordered_labels if label in by_label],
        fontsize=13,
        frameon=True,
        loc="lower right",
        borderpad=0.45,
        handlelength=0.9,
        labelspacing=0.35,
    )
    leg.get_frame().set_linewidth(1.25)

    ax.text(
        0.04,
        0.96,
        f"copy-number boundary ({copy_min:g}-{copy_max:g}x)",
        transform=ax.transAxes,
        fontsize=8.5,
        va="top",
        ha="left",
        color=DARK_GREY,
    )

    fig.tight_layout(pad=0.45)
    png_path = OUT_DIR / f"{output_prefix}_connected_bounds.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white", transparent=False)
    print(f"Connected-bound figure saved: {png_path}")
    print(f"Connected-bound CSV saved: {csv_path}")


def envelope_bound_rows(work: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    rows = []
    for family in ["protease", "CDIs", "CDIaq", "wet-dry"]:
        sub = work[work["reaction_family"] == family].copy()
        if sub.empty:
            continue

        sub = sub.sort_values("n_species")
        bins_here = min(n_bins, max(3, len(sub) // 3))
        ranked = sub["n_species"].rank(method="first")
        sub["envelope_bin"] = pd.qcut(ranked, q=bins_here, labels=False, duplicates="drop")

        for bin_id, bin_df in sub.groupby("envelope_bin", sort=True):
            n_values = bin_df["n_species"].to_numpy(float)
            rows.append(
                {
                    "reaction_family": family,
                    "family_label": FAMILY_LABELS[family],
                    "envelope_bin": int(bin_id),
                    "n_points": len(bin_df),
                    "n_geomean": float(np.exp(np.mean(np.log(n_values)))),
                    "n_min": int(bin_df["n_species"].min()),
                    "n_max": int(bin_df["n_species"].max()),
                    "A_center_mean": float(bin_df["A_center"].mean()),
                    "copy_bound_low": float(bin_df["copy_bound_low"].min()),
                    "copy_bound_high": float(bin_df["copy_bound_high"].max()),
                }
            )
    return pd.DataFrame(rows)


def make_envelope_bound_plot(
    points: pd.DataFrame,
    output_prefix: str,
    copy_min: float,
    copy_max: float,
    center: str,
    n_bins: int,
):
    work = point_bound_dataframe(points, center=center)
    envelope = envelope_bound_rows(work, n_bins=n_bins)

    point_csv = OUT_DIR / f"{output_prefix}_envelope_points.csv"
    envelope_csv = OUT_DIR / f"{output_prefix}_envelope_bounds.csv"
    work.to_csv(point_csv, index=False)
    envelope.to_csv(envelope_csv, index=False)

    fig, ax = plt.subplots(figsize=(5.4, 5.1))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for family in ["protease", "CDIs", "CDIaq", "wet-dry"]:
        env = envelope[envelope["reaction_family"] == family].sort_values("n_geomean")
        pts = work[work["reaction_family"] == family].sort_values("n_species")
        if env.empty or pts.empty:
            continue

        color = FAMILY_COLORS[family]
        x_env = np.ravel(
            np.column_stack(
                [
                    env["n_min"].to_numpy(float),
                    env["n_max"].to_numpy(float),
                ]
            )
        )
        y_low = np.repeat(env["copy_bound_low"].to_numpy(float), 2)
        y_high = np.repeat(env["copy_bound_high"].to_numpy(float), 2)

        ax.fill_between(
            x_env,
            y_low,
            y_high,
            color=color,
            alpha=0.15,
            linewidth=0,
            zorder=1,
        )
        ax.plot(x_env, y_low, color=color, linewidth=1.25, alpha=0.55, zorder=2)
        ax.plot(x_env, y_high, color=color, linewidth=1.25, alpha=0.55, zorder=2)
        ax.scatter(
            pts["n_species"],
            pts["A_center"],
            s=58,
            color=color,
            edgecolor="#1f1f1f",
            linewidth=1.05,
            alpha=0.88,
            zorder=3,
            label=FAMILY_LABELS[family],
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(6, 4000)
    ax.set_ylim(2.8, 150)
    ax.set_xlabel("diversity", fontsize=21)
    ax.set_ylabel("ensemble assembly / A", fontsize=21)
    ax.tick_params(axis="both", which="major", labelsize=19, width=1.7, length=7)
    ax.tick_params(axis="both", which="minor", width=1.25, length=3.8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ordered_labels = [FAMILY_LABELS[fam] for fam in ["protease", "CDIs", "CDIaq", "wet-dry"]]
    leg = ax.legend(
        [by_label[label] for label in ordered_labels if label in by_label],
        [label for label in ordered_labels if label in by_label],
        fontsize=13,
        frameon=True,
        loc="lower right",
        borderpad=0.45,
        handlelength=0.9,
        labelspacing=0.35,
    )
    leg.get_frame().set_linewidth(1.25)

    ax.text(
        0.04,
        0.96,
        f"copy-number envelope ({copy_min:g}-{copy_max:g}x)",
        transform=ax.transAxes,
        fontsize=8.5,
        va="top",
        ha="left",
        color=DARK_GREY,
    )

    fig.tight_layout(pad=0.45)
    png_path = OUT_DIR / f"{output_prefix}_envelope_bounds.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white", transparent=False)
    print(f"Envelope-bound figure saved: {png_path}")
    print(f"Envelope point CSV saved: {point_csv}")
    print(f"Envelope-bound CSV saved: {envelope_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Make reaction-family phase-space CI plot.")
    parser.add_argument("--copy-min", type=float, default=0.5)
    parser.add_argument("--copy-max", type=float, default=5.0)
    parser.add_argument("--copy-min-a", type=float, default=0.0)
    parser.add_argument("--copy-max-a", type=float, default=6.0)
    parser.add_argument("--output-prefix", default="phi_phase_space_family_ci")
    parser.add_argument("--mounds", action="store_true")
    parser.add_argument("--trajectory-bounds", action="store_true")
    parser.add_argument("--point-bounds", action="store_true")
    parser.add_argument("--connected-bounds", action="store_true")
    parser.add_argument("--envelope-bounds", action="store_true")
    parser.add_argument("--envelope-bins", type=int, default=8)
    parser.add_argument("--trajectory-bins", type=int, default=7)
    parser.add_argument("--figure5h-only", action="store_true")
    parser.add_argument(
        "--trajectory-center",
        choices=["paper", "equal"],
        default="paper",
        help="Center trajectory bounds on manuscript A values or equal-copy recalculated A.",
    )
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()

    points = load_family_points(
        copy_min=args.copy_min,
        copy_max=args.copy_max,
        min_a=args.copy_min_a,
        max_a=args.copy_max_a,
    )
    if args.figure5h_only:
        points = filter_figure5h_points(points)
    summary = summarise_by_family(points)

    points_csv = OUT_DIR / f"{args.output_prefix}_points.csv"
    summary_csv = OUT_DIR / f"{args.output_prefix}.csv"
    points.to_csv(points_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    print(f"Point CSV saved: {points_csv}")
    print(f"Family CSV saved: {summary_csv}")
    if args.envelope_bounds:
        make_envelope_bound_plot(
            points,
            args.output_prefix,
            copy_min=args.copy_min,
            copy_max=args.copy_max,
            center=args.trajectory_center,
            n_bins=args.envelope_bins,
        )
    elif args.connected_bounds:
        make_connected_bound_plot(
            points,
            args.output_prefix,
            copy_min=args.copy_min,
            copy_max=args.copy_max,
            center=args.trajectory_center,
        )
    elif args.point_bounds:
        make_point_bound_plot(
            points,
            args.output_prefix,
            copy_min=args.copy_min,
            copy_max=args.copy_max,
            center=args.trajectory_center,
        )
    elif args.trajectory_bounds:
        make_trajectory_bound_plot(
            points,
            args.output_prefix,
            copy_min=args.copy_min,
            copy_max=args.copy_max,
            n_bins=args.trajectory_bins,
            center=args.trajectory_center,
        )
    elif args.mounds:
        make_mound_plot(points, summary, args.output_prefix, confidence=args.confidence)
    else:
        make_plot(points, summary, args.output_prefix, args.copy_min, args.copy_max)


if __name__ == "__main__":
    main()
