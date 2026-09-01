from __future__ import annotations

from pathlib import Path

import pandas as pd

from helpers.data_extractors.all_experiments import AllExperiments
from helpers.plots.diversity_complexity_plots import plot_experiment_trajectories
import important_experiments as ie


SRC_DIR = Path(__file__).resolve().parent
LABBOOK = SRC_DIR / "labbook.csv"
CACHED_ANALYSIS = SRC_DIR / "all_experiments_analysis.pkl"
OUTPUT = SRC_DIR / "Figures_manuscript" / "Figure_5c"

FIGURE_5C_STYLES = [
    {
        "left_color": "black",
        "right_color": "#004c99",
        "linestyle": "-",
        "marker": "o",
        "fillstyle": "none",
    },
    {
        "left_color": "#2f2f2f",
        "right_color": "#0066cc",
        "linestyle": "--",
        "marker": "s",
        "fillstyle": "none",
    },
    {
        "left_color": "#4d4d4d",
        "right_color": "#1f77b4",
        "linestyle": ":",
        "marker": "^",
        "fillstyle": "none",
    },
    {
        "left_color": "#6b6b6b",
        "right_color": "#3399ff",
        "linestyle": "-.",
        "marker": "D",
        "fillstyle": "none",
    },
    {
        "left_color": "#8c8c8c",
        "right_color": "#66b2ff",
        "linestyle": (0, (3, 1, 1, 1)),
        "marker": "x",
        "fillstyle": "none",
    },
]


def main() -> None:
    labbook = AllExperiments().read_labbook(LABBOOK)
    all_data = AllExperiments(confidence=69.95, cache=True, labbook=labbook)
    all_data._full_analysis = pd.read_pickle(CACHED_ANALYSIS)

    labels = [
        str(
            labbook.loc[
                labbook.experiment_id == exp_id,
                "AA_composition",
            ].iloc[0]
        )
        for exp_id in ie.aa5_indiv
    ]

    plot_experiment_trajectories(
        all_data,
        exp_list=ie.aa5_indiv,
        ylim_right=(0, 2800),
        title="5 AAs wet-dry",
        box_anchor=(0.19, 0.70),
        save=str(OUTPUT),
        letter=None,
        show_legend=True,
        show_legend_items=True,
        xlabel="cycle number",
        left_ylabel="exploration ratio",
        right_ylabel="number of unique peptides",
        treat_exp_list_as_separate_trajectories=True,
        trajectory_styles=FIGURE_5C_STYLES,
        trajectory_labels=labels,
        legend_fontsize=7.4,
        legend_title_fontsize=8.3,
        legend_handlelength=2.2,
        dpi=600,
    )


if __name__ == "__main__":
    main()
