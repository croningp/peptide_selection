from __future__ import annotations
from helpers.plots.heatmap_visualizer import HeatmapVisualizer
from helpers.data_extractors.labbook_reader import LabBookHelper
import matplotlib.pyplot as plt
from typing import Iterator
from pathlib import Path

DEFAULT_PARAMS = {
    "patterns": ["G", "A", "V", "P", "H"],
    "figsize": (1.9, 1.9),
    "txt_position": (0.075, 0.81),
    "cbar_width": 0.05,
    "cbar_pad": 0.015,
    "cbar_y_pos": 0.155,
    "cbar_height": 0.99,
    "cmap": "Greys",
    "title_fontsize": 7,
    "label_fontsize": 7,
    "tick_fontsize": 7,
    "label_pad": 7,
    "annot_fontsize": 7,
    "adaptive_colors": True,
    "disable_title": True,
    "cbar_label_fontsize": 7,
    "automatic_abc_labeling": True,
    "segmented_cbar": False,
    "n_segments": 5,
    "abc_fontsize": 8,
    "dpi": 600,
    "bbox_inches": "tight",
    "zero_alpha": 0.5,  # Default alpha for zero values
}

FIGURE_CONFIG = {
    "small_size": {
        "patterns": ["G", "A", "V", "H", "P"],
        "figsize": (1.9, 1.9),
        "txt_position": (0.075, 0.81),
    },
    "large_size": {
        "patterns": [
            A2 + A1
            for A1 in ["G", "A", "V", "H", "P"]
            for A2 in ["G", "A", "V", "H", "P"]
        ],
        "figsize": (5, 5),
        "txt_position": (0.02, 0.88),
        "cbar_width": 0.02,
        "cbar_y_pos": 0.1,
        "cbar_height": 1.13,
    },
    "large_for_manuscript": {
        "patterns": [
            A2 + A1
            for A1 in ["G", "A", "V", "H", "P"]
            for A2 in ["G", "A", "V", "H", "P"]
        ],
        "figsize": (9 / 2.54, 9 / 2.54),  # Convert cm to inches
        "txt_position": (0.02, 0.88),
        "cbar_width": 0.02,
        "cbar_y_pos": 0.1,
        "cbar_height": 1.1,
        "automatic_abc_labeling": False,
        "annot_fontsize": 5,
        "label_fontsize": 6,
        "tick_fontsize": 6,
        "label_pad": 6,
        "cbar_label_fontsize": 6,
        "segmented_cbar": True,
    },
    "large_for_ESI": {
        "patterns": [
            A2 + A1
            for A1 in ["G", "A", "V", "H", "P"]
            for A2 in ["G", "A", "V", "H", "P"]
        ],
        "figsize": (8.3 / 2.54, 8.3 / 2.54),  # Convert cm to inches
        "txt_position": (0.02, 0.88),
        "cbar_width": 0.02,
        "cbar_y_pos": 0.1,
        "cbar_height": 1.1,
        "annot_fontsize": 5,
        "label_fontsize": 6,
        "tick_fontsize": 6,
        "label_pad": 6,
        "cbar_label_fontsize": 6,
        "segmented_cbar": True,
    },
}


def get_figure_params(config_name: str | None = None, **overrides) -> dict:
    """Get figure parameters by merging defaults with config and overrides."""
    params = DEFAULT_PARAMS.copy()

    if config_name and config_name in FIGURE_CONFIG:
        params.update(FIGURE_CONFIG[config_name])

    if overrides:
        params.update(overrides)

    return params


def create_esi_figure(
    experiment: str,
    folder_name: str,
    dataset: dict,
    lb,  # LabBook instance
    abc_iterator: Iterator[str] | None = None,
    config_name: str | None = None,
    **params,
) -> None:
    """Create a single ESI figure with the given configuration."""

    # Get final parameters
    final_params = get_figure_params(config_name, **params)

    # Create output directory
    Path(f"ESI_Figures/{folder_name}").mkdir(parents=True, exist_ok=True)

    # Extract non-HeatmapVisualizer params
    patterns = final_params.pop("patterns")
    txt_position = final_params.pop("txt_position")
    automatic_abc_labeling = final_params.pop("automatic_abc_labeling")
    abc_fontsize = final_params.pop("abc_fontsize")
    dpi = final_params.pop("dpi")
    bbox_inches = final_params.pop("bbox_inches")

    # Create the heatmap with remaining params
    fig, ax = HeatmapVisualizer(dataset, patterns=patterns).create_clean_plot(
        dataset_name=experiment, **final_params
    )

    # Add automatic labeling if enabled
    if automatic_abc_labeling and abc_iterator:
        fig.text(
            txt_position[0],
            txt_position[1],
            f"{next(abc_iterator)})",
            fontsize=abc_fontsize,
            ha="center",
            va="center",
            fontweight="bold",
        )

    # Save the figure
    plt.savefig(
        f"ESI_Figures/{folder_name}/heatmap_{experiment}.png",
        bbox_inches=bbox_inches,
        dpi=dpi,
    )
    plt.close(fig)


def figures_for_esi(
    trajectory_name: str,
    folder_name: str,
    config_name: str | None = None,
    df=None,  # DataFrame
    lb=None,  # LabBook instance
    **params,
) -> None:
    """Generate all ESI figures for a trajectory using the specified configuration."""

    # Get final parameters
    final_params = get_figure_params(config_name, **params)

    # Get dataset
    dataset = LabBookHelper(lb).get_experiment_data_dict(
        df, trajectory_name, value_column="observed_nodes"
    )

    # Initialize ABC iterator if automatic labeling is enabled
    abc_iterator = (
        iter("abcdefghijklmnopqrstuvwxyz")
        if final_params["automatic_abc_labeling"]
        else None
    )

    # Generate figures for each experiment
    for experiment in LabBookHelper(lb).get_experiment_sequence_names(
        trajectory_name
    ):
        create_esi_figure(
            # trajectory_name=trajectory_name,
            experiment=experiment,
            folder_name=folder_name,
            dataset=dataset,
            lb=lb,
            abc_iterator=abc_iterator,
            config_name=config_name,
            **params,
        )


def figure_for_esi(
    trajectory_name: str,
    experiment: str,
    folder_name: str,
    config_name: str | None = None,
    df=None,  # DataFrame
    lb=None,  # LabBook instance
    **params,
) -> None:
    dataset = LabBookHelper(lb).get_experiment_data_dict(
        df, trajectory_name, value_column="observed_nodes"
    )
    create_esi_figure(
        experiment=experiment,
        folder_name=folder_name,
        dataset=dataset,
        lb=lb,
        abc_iterator=None,  # No ABC labeling for single figures
        config_name=config_name,
        **params,
    )
