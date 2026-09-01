"""
Symmetric two-case explanatory figure for R4.3.

Columns are the two real cases:
    - no-selection/high-exploration example
    - directed-exploration/selection example

Rows are:
    1. probability of species in each MA bin
    2. assumed relative copy number per species in each MA bin

The two columns share axis limits so the cases can be compared directly.
"""

from __future__ import annotations

import argparse
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


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

SRC_DIR = Path(__file__).resolve().parents[1]
PKL = SRC_DIR / "all_experiments_analysis.pkl"
OUT_DIR = Path(__file__).resolve().parent


def observed_sequences(observed_nodes):
    if isinstance(observed_nodes, (list, tuple)) and observed_nodes:
        first = observed_nodes[0]
        if isinstance(first, (list, tuple, set, np.ndarray)):
            return list(first)
    return list(observed_nodes)


def load_case(experiment_name: str, label: str, color: str) -> dict:
    with open(PKL, "rb") as handle:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            df_exp = pickle.load(handle)

    rows = df_exp[df_exp["name"] == experiment_name]
    if rows.empty:
        raise ValueError(f"Experiment not found: {experiment_name}")

    row = rows.iloc[0]
    seqs = observed_sequences(row["observed_nodes"])
    a_i = np.array([len(seq) - 1 for seq in seqs], dtype=int)
    values, counts = np.unique(a_i, return_counts=True)
    p_species = counts / counts.sum()
    return {
        "label": label,
        "name": row["name"],
        "color": color,
        "A_paper": float(row["A"]),
        "n_species": int(len(a_i)),
        "exploration_ratio": float(row["exploration_ratio"]),
        "values": values,
        "counts": counts,
        "p_species": p_species,
    }


def align_distribution(case: dict, all_values: np.ndarray) -> np.ndarray:
    out = np.zeros(len(all_values), dtype=float)
    for value, prob in zip(case["values"], case["p_species"]):
        out[np.where(all_values == value)[0][0]] = prob
    return out


def relative_copy_number(values: np.ndarray, p_species: np.ndarray, theta: float):
    mean_a = float(np.sum(values * p_species))
    raw_c = np.exp(theta * (values - mean_a))
    return raw_c / float(np.sum(p_species * raw_c))


def endpoint_copy_number(
    values: np.ndarray,
    all_values: np.ndarray,
    copy_min: float,
    copy_max: float,
    direction: str,
) -> np.ndarray:
    """Copy number curve with fixed endpoints over the shared MA-bin range."""
    span = max(float(all_values.max() - all_values.min()), 1.0)
    position = (values - all_values.min()) / span
    if direction == "high_ma_high_copy":
        log_c = np.log(copy_min) + position * (np.log(copy_max) - np.log(copy_min))
    elif direction == "low_ma_high_copy":
        log_c = np.log(copy_max) - position * (np.log(copy_max) - np.log(copy_min))
    elif direction == "equal":
        log_c = np.zeros_like(values, dtype=float)
    else:
        raise ValueError(f"Unknown copy-number direction: {direction}")
    return np.exp(log_c)


def ensemble_A(values: np.ndarray, p_distribution: np.ndarray) -> float:
    return float(np.sum(p_distribution * np.exp(values)))


def A_with_copy_number(
    values: np.ndarray,
    p_species: np.ndarray,
    c_rel: np.ndarray,
) -> float:
    p_weighted = p_species * c_rel
    p_weighted = p_weighted / p_weighted.sum()
    return ensemble_A(values, p_weighted)


def make_outputs(
    thetas: list[float],
    output_prefix: str,
    copy_ymin: float,
    copy_ymax: float,
    endpoint_copy_range: bool,
    copy_min: float,
    copy_max: float,
) -> None:
    cases = [
        load_case("ripper_ARM_07_52", "No selection", "#d9d9d9"),
        load_case("ripper_20240704_MYW_34_L_6", "Directed exploration", "#d9d9d9"),
    ]

    all_values = np.arange(
        min(case["values"].min() for case in cases),
        max(case["values"].max() for case in cases) + 1,
    )
    ma_tick_labels = [f"{int(a)}" for a in all_values]

    copy_colors = {
        "low-MA high copy": "#1f77b4",
        "equal copy": "#4d4d4d",
        "high-MA high copy": "#ff7f0e",
        -1.0: "#1f77b4",
        -0.5: "#1f77b4",
        0.0: "#4d4d4d",
        0.5: "#ff7f0e",
        1.0: "#ff7f0e",
    }
    fallback_colors = plt.cm.tab10(np.linspace(0, 1, len(thetas)))

    if endpoint_copy_range:
        copy_curves = [
            ("low-MA high copy", "low_ma_high_copy"),
            ("equal copy", "equal"),
            ("high-MA high copy", "high_ma_high_copy"),
        ]
    else:
        copy_curves = [(f"theta={theta:+.1f}", theta) for theta in thetas]

    rows = []
    max_prob = 0.0
    max_copy = 0.0
    for case in cases:
        p_full = align_distribution(case, all_values)
        max_prob = max(max_prob, float(p_full.max()))
        A_equal = ensemble_A(case["values"], case["p_species"])
        for curve_label, curve_spec in copy_curves:
            if endpoint_copy_range:
                c_rel = endpoint_copy_number(
                    case["values"],
                    all_values,
                    copy_min,
                    copy_max,
                    str(curve_spec),
                )
            else:
                c_rel = relative_copy_number(
                    case["values"], case["p_species"], float(curve_spec)
                )
            max_copy = max(max_copy, float(c_rel.max()))
            A_theta = A_with_copy_number(case["values"], case["p_species"], c_rel)
            for a, p_a in zip(all_values, p_full):
                c_at_a = 0.0
                if a in case["values"]:
                    c_at_a = float(c_rel[np.where(case["values"] == a)[0][0]])
                rows.append(
                    {
                        "case_label": case["label"],
                        "experiment": case["name"],
                        "n_species": case["n_species"],
                        "exploration_ratio": round(case["exploration_ratio"], 6),
                        "a_i": int(a),
                        "MA_i_exp_ai": round(float(np.exp(a)), 6),
                        "p_species": round(float(p_a), 8),
                        "curve": curve_label,
                        "theta": curve_spec if not endpoint_copy_range else "",
                        "relative_copy_number_per_species": round(c_at_a, 8),
                        "A_equal": round(A_equal, 6),
                        "A_weighted": round(A_theta, 6),
                        "delta_A_pct": round((A_theta - A_equal) / A_equal * 100.0, 6),
                    }
                )

    pd.DataFrame(rows).to_csv(OUT_DIR / f"{output_prefix}.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.4), sharex=True)
    fig.patch.set_facecolor("white")

    for col, case in enumerate(cases):
        p_full = align_distribution(case, all_values)
        A_equal = ensemble_A(case["values"], case["p_species"])

        ax = axes[0, col]
        ax.bar(
            all_values,
            p_full,
            width=0.68,
            color=case["color"],
            edgecolor="#4d4d4d",
            linewidth=0.9,
            alpha=1.0,
        )
        ax.set_ylim(0, max_prob * 1.12)
        ax.set_title(
            f"{case['label']}\nA = {A_equal:.1f}, n = {case['n_species']}",
            fontsize=10,
        )
        if col == 0:
            ax.set_ylabel("Species probability")
        ax.text(
            0.98,
            0.88,
            f"ER = {case['exploration_ratio']:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="#2f3437",
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#d0d0d0", alpha=0.9),
        )

        ax = axes[1, col]
        for i, (curve_label, curve_spec) in enumerate(copy_curves):
            color = copy_colors.get(curve_label, fallback_colors[i])
            if endpoint_copy_range:
                c_rel = endpoint_copy_number(
                    case["values"],
                    all_values,
                    copy_min,
                    copy_max,
                    str(curve_spec),
                )
            else:
                color = copy_colors.get(float(curve_spec), fallback_colors[i])
                c_rel = relative_copy_number(
                    case["values"], case["p_species"], float(curve_spec)
                )
            c_full = np.zeros_like(all_values, dtype=float)
            for a, c_a in zip(case["values"], c_rel):
                c_full[np.where(all_values == a)[0][0]] = c_a
            A_theta = A_with_copy_number(case["values"], case["p_species"], c_rel)
            linestyle = ":" if curve_label == "equal copy" else "-"
            ax.plot(
                all_values,
                c_full,
                marker="o",
                linewidth=2.0,
                markersize=4.6,
                color=color,
                linestyle=linestyle,
                label=f"{curve_label}, A={A_theta:.1f}",
            )
        ax.set_ylim(copy_ymin, copy_ymax)
        if col == 0:
            ax.set_ylabel("Relative copy number\nper species")
        ax.set_xlabel("MA bin (a_i)")
        ax.legend(fontsize=7.8, loc="upper left", frameon=False)

    for ax in axes.ravel():
        ax.set_xlim(all_values.min() - 0.6, all_values.max() + 0.6)
        ax.set_xticks(all_values)
        ax.set_facecolor("white")
        ax.grid(axis="y", color="#ebebeb", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8.5)

    for ax in axes[0, :]:
        ax.tick_params(labelbottom=False)
    for ax in axes[1, :]:
        ax.set_xticklabels(ma_tick_labels)

    fig.suptitle(
        "Same MA-bin probabilities and copy-number assumptions shown symmetrically",
        fontsize=12,
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png_path = OUT_DIR / f"{output_prefix}.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white", transparent=False)

    print(f"Figure saved: {png_path}")
    print(f"CSV saved: {OUT_DIR / f'{output_prefix}.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Make symmetric two-case phi figure.")
    parser.add_argument("--thetas", type=float, nargs="+", default=[-0.5, 0.0, 0.5])
    parser.add_argument("--output-prefix", default="phi_two_case_symmetric")
    parser.add_argument("--copy-ymin", type=float, default=0.1)
    parser.add_argument("--copy-ymax", type=float, default=10.0)
    parser.add_argument("--endpoint-copy-range", action="store_true")
    parser.add_argument("--copy-min", type=float, default=0.5)
    parser.add_argument("--copy-max", type=float, default=5.0)
    args = parser.parse_args()
    make_outputs(
        thetas=args.thetas,
        output_prefix=args.output_prefix,
        copy_ymin=args.copy_ymin,
        copy_ymax=args.copy_ymax,
        endpoint_copy_range=args.endpoint_copy_range,
        copy_min=args.copy_min,
        copy_max=args.copy_max,
    )


if __name__ == "__main__":
    main()
