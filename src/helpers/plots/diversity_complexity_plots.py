from itertools import cycle
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from pathlib import Path
from matplotlib.ticker import MultipleLocator
from matplotlib.figure import Figure
from typing import Any
import pandas as pd


# TODO if column exploration_ratio - have y label explotration ratio
# TODO if right axis. set it blue colour
# TODO have better contorl over params


# Configure matplotlib globally
mpl.rcParams["font.family"] = ["sans-serif"]
mpl.rcParams["font.serif"] = ["Arial"]

# Modern type aliases
Array = list[float] | np.ndarray
AxisLimit = tuple[float, float] | list[float]

# Default plotting parameters
DEFAULT_PLOT_PARAMS = {
    "width": 75,  # mm
    "height": 75,  # mm
    "marker": "o",
    "default_color": "black",
    "linestyle": "-",
    "linewidth": 1.0,
    "markersize": 6,
    "fontsize": 10,
    "dpi": 300,
    "output_dir": "Figures",
    "bbox_inches": "tight",
    "fontweight": "normal",
    "fontfamily": "sans-serif",
    "capsize": 5,
    "left_axis_color": "black",
    "right_axis_color": "blue",
    "right_linestyle": "--",
}

# Predefined plot configurations
PLOT_CONFIGS = {
    "manuscript": {
        "width": 73,
        "height": 73,
        "fontsize": 10,
        "linewidth": 1.0,
        "markersize": 6,
        "dpi": 300,
        "output_dir": "ESI_Figures",
    },
    "ESI": {
        "width": 73,
        "height": 73,
        "fontsize": 10,
        "linewidth": 1.0,
        "markersize": 6,
        "dpi": 300,
        "output_dir": "ESI_Figures",
    },
}


def get_plot_params(
    config_name: str | None = None, **overrides
) -> dict[str, float | str | int]:
    """Get plot parameters by merging defaults with config and overrides."""
    params = DEFAULT_PLOT_PARAMS.copy()

    if config_name and config_name in PLOT_CONFIGS:
        params.update(PLOT_CONFIGS[config_name])

    if overrides:
        params.update(overrides)

    return params


class Plot:
    def __init__(self, config_name: str | None = None, **kwargs) -> None:
        """
        Initialize plot with config name or individual parameters.

        Args:
            config_name: Name of predefined configuration ('manuscript', 'presentation', etc.)
            **kwargs: Individual parameter overrides
        """
        # Get final parameters
        self.params = get_plot_params(config_name, **kwargs)

        self._setup_plot_properties()
        self._initialize_figure()

    def _setup_plot_properties(self) -> None:
        """Set up font and plotting properties from parameters."""
        self.label_fontdict: dict[str, str | int] = {
            "fontsize": int(self.params["fontsize"]),
            "fontweight": str(self.params["fontweight"]),
            "fontfamily": str(self.params["fontfamily"]),
        }

        self.plot_dict_props: dict[str, str | float | int] = {
            "marker": str(self.params["marker"]),
            "linewidth": float(self.params["linewidth"]),
            "markersize": int(self.params["markersize"]),
        }

        self.axis_colours: dict[str, str] = {
            "left": str(self.params["left_axis_color"]),
            "right": str(self.params["right_axis_color"]),
        }

        # Cycle through marker fill styles
        self.marker_shape = cycle(Line2D.fillStyles)

        # Ensure output directory exists
        Path(str(self.params["output_dir"])).mkdir(exist_ok=True)

    def _initialize_figure(self) -> None:
        """Create the matplotlib figure and primary axis."""
        mm = 1 / 25.4  # Convert mm to inches

        width = float(self.params["width"])
        height = float(self.params["height"])
        self.fig, self.ax1 = plt.subplots(figsize=(width * mm, height * mm))

    def add_trajectory(
        self,
        x_axis: Array | str,
        y_axis: Array,
        _marker_shape: str | None = None,
        yerr: Array | None = None,
        color: str | None = None,
        linestyle: str | tuple | None = None,
        marker: str | None = None,
        label: str | None = None,
        axis: str = "left",
        ylabel: str = "",
    ) -> None:
        """
        Add a trajectory to the plot.

        Args:
            x_axis: X data or 'cycle' for auto-generated cycle numbers
            y_axis: Y data array
            _marker_shape: Marker fill style (auto-cycles if None)
            yerr: Error bars for Y data
            color: Line color (uses default if None)
            linestyle: Line style override
            marker: Marker shape override
            label: Legend label override
            axis: 'left' or 'right' y-axis
            ylabel: Label for the y-axis
        """
        # Select axis and line style
        if axis == "left":
            ax = self.ax1
            line_style = self.params["linestyle"]
        elif axis == "right":
            if not hasattr(self, "ax2"):
                self.ax2 = self.ax1.twinx()
            ax = self.ax2
            line_style = self.params["right_linestyle"]
        else:
            raise ValueError("axis must be 'left' or 'right'")

        if linestyle is not None:
            line_style = linestyle

        # Handle marker shape
        if _marker_shape is None:
            _marker_shape = next(self.marker_shape)

        # Handle color
        if color is None:
            color = self.axis_colours.get(axis, self.params["default_color"])

        # Process x-axis data
        x_data = self._process_x_axis(x_axis, y_axis)

        plot_props = self.plot_dict_props.copy()
        if marker is not None:
            plot_props["marker"] = marker
        if label is not None:
            plot_props["label"] = label

        # Plot the trajectory
        ax.plot(
            x_data,
            y_axis,
            color=color,
            fillstyle=_marker_shape,
            linestyle=line_style,
            **plot_props,
        )

        # Set y-axis label
        if ylabel:
            ax.set_ylabel(ylabel, fontdict=self.label_fontdict)

        # Add error bars if provided
        if yerr is not None:
            self.add_error_bars_y(
                x_data,
                y_axis,
                yerr,
                axis_handle=ax,
                color=color,
                line_style=line_style,
            )

    def _process_x_axis(self, x_axis: Array | str, y_axis: Array) -> Array:
        """Process x-axis data, handling 'cycle' string input."""
        if isinstance(x_axis, str):
            if x_axis == "cycle":
                return np.arange(1, len(y_axis) + 1)
            else:
                raise ValueError("x_axis must be an array or 'cycle'")
        return x_axis

    def add_error_bars_y(
        self,
        x_axis: Array,
        y_axis: Array,
        yerr: Array,
        axis_handle: mpl.axes.Axes,
        color: str = "black",
        line_style: str = "-",
    ) -> None:
        """Add error bars to the specified axis."""
        axis_handle.errorbar(
            x=x_axis,
            y=y_axis,
            yerr=yerr,
            color=color,
            capsize=self.params["capsize"],
            linestyle=line_style,
            linewidth=self.params["linewidth"],
            label="_nolegend_",
        )

    def release(
        self,
        left_limit: AxisLimit | None = None,
        right_limit: AxisLimit | None = None,
        xlabel: str = "",
    ) -> mpl.figure.Figure:
        """
        Finalize the plot with limits and labels.

        Args:
            left_limit: Y-axis limits for left axis
            right_limit: Y-axis limits for right axis
            xlabel: Label for x-axis

        Returns:
            The matplotlib figure object
        """
        # Set axis limits
        if left_limit is not None:
            self.ax1.set_ylim(left_limit)

        if hasattr(self, "ax2") and right_limit is not None:
            self.ax2.set_ylim(right_limit)

        # Set labels
        self.ax1.set_xlabel(xlabel, fontdict=self.label_fontdict)

        # Style the right axis if it exists
        if hasattr(self, "ax2"):
            self.ax2.tick_params(
                axis="y", colors=self.params["right_axis_color"]
            )
            self.ax2.spines["right"].set_color(
                str(self.params["right_axis_color"])
            )

        return self.fig

    def save_figure(self, filename: str, **kwargs) -> None:
        """
        Save the figure to file.

        Args:
            filename: Output filename
            **kwargs: Additional arguments passed to savefig
        """
        filepath = Path(str(self.params["output_dir"])) / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs = {
            "dpi": int(self.params["dpi"]),
            "bbox_inches": str(self.params["bbox_inches"]),
            **kwargs,
        }
        self.fig.savefig(str(filepath), **save_kwargs)
        print(f"Saved figure: {filepath}")


def construct_plot(
    p,
    all_data,
    exp_list,
    crop_max_cycle,
    left_axis,
    right_axis,
    figure_style,
    ylabel,
    left_ylabel,
    right_ylabel,
    marker_shape,
    style: dict[str, Any] | None = None,
    label: str | None = None,
):
    style = style or {}
    y, yerr = all_data.combine_trajectories(
        exp_list, columns=[left_axis], crop_max_cycle=crop_max_cycle
    )

    # Use ylabel if provided, else left_ylabel, else left_axis

    left_y_label = (
        ylabel
        if ylabel is not None
        else (left_ylabel if left_ylabel is not None else left_axis)
    )

    p.add_trajectory(
        "cycle",
        y,
        yerr=yerr,
        ylabel=left_y_label,
        _marker_shape=style.get("fillstyle", marker_shape),
        color=style.get("left_color", style.get("color")),
        linestyle=style.get("linestyle"),
        marker=style.get("marker"),
        label=label,
    )

    div, diverr = all_data.combine_trajectories(
        exp_list, columns=[right_axis], crop_max_cycle=crop_max_cycle
    )
    p.add_trajectory(
        "cycle",
        div,
        yerr=diverr,
        axis="right",
        ylabel=right_ylabel if right_ylabel is not None else right_axis,
        _marker_shape=style.get("fillstyle", marker_shape),
        color=style.get("right_color", style.get("color")),
        linestyle=style.get("linestyle"),
        marker=style.get("marker"),
        label="_nolegend_" if label is not None else None,
    )
    return p


def plot_experiment_trajectories(
    all_data: Any,
    exp_list: list[str],
    ylim_right: tuple[float, float] = (0, 1000),
    ylim_left: tuple[float, float] = (0, 1),
    letter: str | None = None,
    title: str = "",
    box_anchor: tuple[float, float] = (0.5, 0.5),
    save: bool | str = False,
    crop_max_cycle: int = 6,
    left_axis: str = "exploration_ratio",
    right_axis: str = "diversity",
    figure_style: str = "manuscript",
    show_legend: bool = True,
    show_scatter_legend: bool = True,
    show_legend_items: bool = True,
    xlabel: str | None = None,
    left_ylabel: str | None = None,
    right_ylabel: str | None = None,
    ylabel: str | None = None,
    treat_exp_list_as_separate_trajectories: bool = False,
    marker_shape: str | None = None,
    trajectory_styles: list[dict[str, Any]] | None = None,
    trajectory_labels: list[str] | None = None,
    legend_fontsize: float = 10,
    legend_title_fontsize: float | None = None,
    legend_handlelength: float = 1,
    dpi: int | None = None,
) -> Figure | None:
    """
    Plot experiment trajectories with dual y-axes, optional letter label, and title.

    Args:
        all_data: Data object with combine_trajectories method.
        exp_list: List of experiment names.
        ylim_right: Y-axis limits for right axis.
        ylim_left: Y-axis limits for left axis.
        letter: Optional subplot letter label.
        title: Title for the legend.
        box_anchor: Anchor for the legend box.
        save: Whether to save the figure (bool or filename).
        crop_max_cycle: Max cycle for data cropping.
        left_axis: Column for left y-axis.
        right_axis: Column for right y-axis.
        figure_style: Plot style config.
        show_legend: Whether to display the legend.
        show_scatter_legend: Whether to include scatter point legends.
        show_legend_items: Whether to show legend items.
        xlabel: Label for the x-axis.
        left_ylabel: Label for the left y-axis.
        right_ylabel: Label for the right y-axis.
        ylabel: Alias for left_ylabel (overrides if provided).
        treat_exp_list_as_separate_trajectories: If True,
            plot each experiment separately with special symbol.
        trajectory_styles: Optional style dictionaries used for separate
            trajectories. Supported keys include color, linestyle, and marker.
        trajectory_labels: Optional legend labels used for separate trajectories.
        legend_fontsize: Font size for legend text.
        legend_title_fontsize: Font size for the legend title.
        legend_handlelength: Length of legend line handles.
        dpi: Optional output resolution override.

    Returns:
        Figure or None if saved.
    """
    plot_overrides = {"dpi": dpi} if dpi is not None else {}
    p = Plot(figure_style, **plot_overrides)

    # I will fix this horrible structure if I have time before submission
    if treat_exp_list_as_separate_trajectories:
        for i, exp in enumerate(exp_list):
            style = (
                trajectory_styles[i % len(trajectory_styles)]
                if trajectory_styles
                else None
            )
            label = (
                trajectory_labels[i]
                if trajectory_labels and i < len(trajectory_labels)
                else None
            )
            p = construct_plot(
                p,
                all_data,
                [exp],
                crop_max_cycle,
                left_axis,
                right_axis,
                figure_style,
                ylabel,
                left_ylabel,
                right_ylabel,
                next(p.marker_shape),
                style,
                label,
            )
    else:
        p = construct_plot(
            p,
            all_data,
            exp_list,
            crop_max_cycle,
            left_axis,
            right_axis,
            figure_style,
            ylabel,
            left_ylabel,
            right_ylabel,
            marker_shape,
        )

    # Make right y-label blue
    if hasattr(p, "ax2"):
        p.ax2.yaxis.label.set_color(
            str(p.params.get("right_axis_color", "blue"))
        )

    p.ax1.xaxis.set_major_locator(MultipleLocator(1))
    fig = p.release(left_limit=ylim_left, right_limit=ylim_right, xlabel=xlabel)

    if letter:
        fig.text(
            -0.24,
            0.96,
            f"{letter})",
            transform=fig.gca().transAxes,
            fontsize=12,
            fontweight="bold",
        )

    # Add legend with title and position if requested
    # Sorry for million nested ifs
    if show_legend:
        if show_legend_items:  # Only collect and show legend items if True
            handles, labels = [], []
            for ax in [p.ax1] + ([p.ax2] if hasattr(p, "ax2") else []):
                for line in ax.get_lines():
                    handles.append(line)
                    labels.append(line.get_label())
                if show_scatter_legend:
                    for col in ax.collections:
                        # Only add scatter points with a label not "_nolegend_"
                        if hasattr(col, "get_label"):
                            lab = col.get_label()
                            if lab and lab != "_nolegend_":
                                handles.append(col)
                                labels.append(lab)
            # Remove duplicates and "_nolegend_"
            legend_items = [
                (h, l) for h, l in zip(handles, labels) if l != "_nolegend_"
            ]
            # Remove scatter points if show_scatter_legend is False
            if not show_scatter_legend:
                from matplotlib.collections import PathCollection

                legend_items = [
                    (h, l)
                    for h, l in legend_items
                    if not isinstance(h, PathCollection)
                ]

            if legend_items:
                handles, labels = zip(*legend_items)
                fig.legend(
                    handles,
                    labels,
                    loc="upper left",
                    bbox_to_anchor=box_anchor,
                    frameon=False,
                    handlelength=legend_handlelength,
                    title=title if title else None,
                    title_fontsize=(
                        legend_title_fontsize
                        if legend_title_fontsize is not None
                        else legend_fontsize
                    ),
                    prop={"size": legend_fontsize},
                )
        else:
            # Show only title text without legend items
            if title:
                fig.text(
                    box_anchor[0],
                    box_anchor[1],
                    title,
                    transform=fig.transFigure,
                    fontsize=10,
                    ha="left",
                    va="top",
                )

    if save:
        if not isinstance(save, str):
            save = str(letter) if letter else "plot"
        p.save_figure(f"{save}.png")
        plt.close(fig)
        return None

    return fig


def plot_diversity_vs_column(
    all_data,
    exp_list: list[str],
    y_column: str = "A",
    y_label: str | None = None,
    x_column: str = "diversity",
    letter: str | None = None,
    save: bool | str = False,
    crop_max_cycle: int = 6,
    figure_style: str = "manuscript",
    ylim: tuple[float, float] | None = None,
    extend_from_origin: bool = False,
) -> Figure | None:
    """
    Plot a single y-axis column against diversity (or other x column) with error bars.

    Args:
        all_data: AllExperiments object containing the data
        exp_list: List of experiment names to plot
        y_column: Column name for y-axis (default: "A")
        y_label: Label for y-axis (defaults to y_column)
        x_column: Column name for x-axis (default: "diversity")
        letter: Optional letter label for subfigures
        save: Whether to save figure (bool) or filename (str)
        crop_max_cycle: Maximum cycle to include in analysis
        figure_style: Plot style configuration
        ylim: Y-axis limits as (min, max) tuple
        extend_from_origin: If True, extend trajectory from origin (0,0) to first point

    Returns:
        Figure object or None if saved
    """
    y_data, y_err = all_data.combine_trajectories(
        exp_list, columns=[y_column], crop_max_cycle=crop_max_cycle
    )
    x_data, x_err = all_data.combine_trajectories(
        exp_list, columns=[x_column], crop_max_cycle=crop_max_cycle
    )

    # Extend from origin if requested
    if extend_from_origin and len(x_data) > 0 and len(y_data) > 0:
        # Prepend origin point (0,0) to the data
        x_data = np.concatenate([[0], x_data])
        y_data = np.concatenate([[0], y_data])

        # Handle error bars - set to zero for origin point
        if y_err is not None:
            y_err = np.concatenate([[0], y_err])

    p = Plot(figure_style)

    if y_label is None:
        y_label = y_column

    # Add trajectory with error bars if more than one experiment
    if len(exp_list) > 1:
        p.add_trajectory(x_data, y_data, yerr=y_err, ylabel=y_label)
    else:
        # Single experiment - no error bars needed
        p.add_trajectory(x_data, y_data, ylabel=y_label)

    # Set y-axis limits if provided
    fig = p.release(left_limit=ylim, xlabel=x_column)

    # Add letter label if provided
    if letter:
        fig.text(
            -0.24,
            0.96,
            f"{letter})",
            transform=fig.gca().transAxes,
            fontsize=12,
            fontweight="bold",
        )

    # Save or return figure
    if save:
        if not isinstance(save, str):
            save = f"{y_column}_vs_{x_column}"
        p.save_figure(f"{save}.png")
        plt.close(fig)
        return None

    return fig


def plot_multiple_trajectories_vs_column(
    all_data,
    trajectory_groups: list[list[str]],
    trajectory_labels: list[str] | None = None,
    y_column: str = "A",
    y_label: str | None = None,
    x_column: str = "diversity",
    letter: str | None = None,
    save: bool | str = False,
    crop_max_cycle: int = 6,
    figure_style: str = "manuscript",
    ylim: tuple[float, float] | None = None,
    y_scale: str = "linear",
    colors: list[str] | None = None,
    extend_from_origin: bool = False,
    x_label: str | None = None,
) -> Figure | None:
    """
    Plot multiple trajectories on the same plot, each potentially with error bars.

    Args:
        all_data: AllExperiments object containing the data
        trajectory_groups: List of experiment lists, each group will be one trajectory
        trajectory_labels: Optional labels for each trajectory group
        y_column: Column name for y-axis (default: "A")
        y_label: Label for y-axis (defaults to y_column)
        x_column: Column name for x-axis (default: "diversity")
        letter: Optional letter label for subfigures
        save: Whether to save figure (bool) or filename (str)
        crop_max_cycle: Maximum cycle to include in analysis
        figure_style: Plot style configuration
        ylim: Y-axis limits as (min, max) tuple
        colors: Optional list of colors for each trajectory
        extend_from_origin: If True, extend each trajectory from origin (0,0) to first point
        x_label: Label for the x-axis (defaults to x_column)

    Returns:
        Figure object or None if saved

    Example:
        # Plot papain and bromelain as two separate trajectories
        plot_multiple_trajectories_vs_column(
            all_data,
            [important_experiments.pap, important_experiments.brom],
            trajectory_labels=["Papain", "Bromelain"],
            y_column="A",
            colors=["blue", "red"],
            extend_from_origin=True
        )
    """
    if y_label is None:
        y_label = y_column

    if x_label is None:
        x_label = x_column

    if trajectory_labels is None:
        trajectory_labels = [
            f"Trajectory {i + 1}" for i in range(len(trajectory_groups))
        ]

    if colors is None:
        # Default colors
        colors = [
            "blue",
            "red",
            "green",
            "orange",
            "purple",
            "brown",
            "pink",
            "gray",
        ]

    p = Plot(figure_style)

    # Plot each trajectory group
    for i, (exp_list, label, color) in enumerate(
        zip(trajectory_groups, trajectory_labels, colors)
    ):
        # Get data for this trajectory group
        y_data, y_err = all_data.combine_trajectories(
            exp_list, columns=[y_column], crop_max_cycle=crop_max_cycle
        )
        x_data, x_err = all_data.combine_trajectories(
            exp_list, columns=[x_column], crop_max_cycle=crop_max_cycle
        )

        # Extend from origin if requested
        if extend_from_origin and len(x_data) > 0 and len(y_data) > 0:
            # Prepend origin point (0,0) to the data
            x_data = np.concatenate([[0], x_data])
            y_data = np.concatenate([[0], y_data])

            # Handle error bars - set to zero for origin point
            if y_err is not None:
                y_err = np.concatenate([[0], y_err])

        # Prepare data for log scale if requested
        if y_scale == "log":
            # Mask or clip non-positive values since log scale can't display them
            # We'll replace non-positive y values with nan so matplotlib will skip them
            with np.errstate(invalid="ignore"):
                y_data = np.where(np.array(y_data) > 0, y_data, np.nan)
                if y_err is not None:
                    # Keep y_err as-is, but ensure lengths match if origin was prepended
                    y_err = np.array(y_err)

        # Add trajectory with error bars if more than one experiment in group
        if len(exp_list) > 1:
            p.add_trajectory(
                x_data,
                y_data,
                yerr=y_err,
                color=color,
                ylabel=y_label
                if i == 0
                else "",  # Only set ylabel for first trajectory
            )
        else:
            # Single experiment - no error bars needed
            p.add_trajectory(
                x_data, y_data, color=color, ylabel=y_label if i == 0 else ""
            )

    # Create legend if we have labels
    if any(
        label != f"Trajectory {i + 1}"
        for i, label in enumerate(trajectory_labels)
    ):
        # Add legend to the plot
        p.ax1.legend(trajectory_labels, loc="best")

    # Set y-axis limits if provided
    # Set axis scale (linear/log)
    if y_scale == "log":
        p.ax1.set_yscale("log")

        # If user provided ylim, ensure lower bound is positive for log scale
        if ylim is not None:
            low, high = ylim
            if low is not None and low <= 0:
                # adjust to a small positive number or the smallest positive y in data
                finite_ys = np.array(
                    [
                        v
                        for v in np.ravel(
                            [
                                np.array(
                                    all_data.combine_trajectories(
                                        g,
                                        columns=[y_column],
                                        crop_max_cycle=crop_max_cycle,
                                    )[0]
                                )
                                for g in trajectory_groups
                            ]
                        )
                        if np.isfinite(v) and v > 0
                    ]
                )
                min_positive = (
                    float(np.min(finite_ys)) if finite_ys.size > 0 else 1e-3
                )
                low = min_positive
            ylim = (low, high)

    fig = p.release(left_limit=ylim, xlabel=x_label)

    # Add letter label if provided
    if letter:
        fig.text(
            -0.24,
            0.96,
            f"{letter})",
            transform=fig.gca().transAxes,
            fontsize=12,
            fontweight="bold",
        )

    # Save or return figure
    if save:
        if not isinstance(save, str):
            save = f"multiple_{y_column}_vs_{x_column}"
        p.save_figure(f"{save}.png")
        plt.close(fig)
        return None

    return fig


def create_dual_axis_plot(
    x_data: Array | str,
    left_y: Array,
    right_y: Array,
    left_yerr: Array | None = None,
    right_yerr: Array | None = None,
    left_ylabel: str = "",
    right_ylabel: str = "",
    xlabel: str = "",
    left_limit: AxisLimit | None = None,
    right_limit: AxisLimit | None = None,
    config_name: str | None = None,
    **plot_kwargs,
) -> Figure:
    plot = Plot(config_name, **plot_kwargs)

    plot.add_trajectory(
        x_data, left_y, yerr=left_yerr, axis="left", ylabel=left_ylabel
    )
    plot.add_trajectory(
        x_data, right_y, yerr=right_yerr, axis="right", ylabel=right_ylabel
    )

    # Finalize and return
    return plot.release(left_limit, right_limit, xlabel)


def label_dataframe(all_data, labels: dict[str, list[str]]) -> pd.DataFrame:
    """
    Label the full analysis dataframe based on experiment names.

    Args:
        all_data: Data object with _full_analysis and experiment name utilities.
        labels: Dict mapping label name to list of experiment names/IDs.

    Returns:
        DataFrame with a new 'label' column.
    """
    all_data._assert_full_analysis_was_calculated()
    _df = all_data._full_analysis.copy()
    for label, names in labels.items():
        idx = all_data._name_present_in_df(
            all_data.get_list_of_experiment_names_from_ids(names)
        )
        _df.loc[idx, "label"] = label
    return _df


def plot_labeled_scatter_by_label(
    df: pd.DataFrame,
    x: str = "diversity",
    y: str = "exploration_ratio",
    label_col: str = "label",
    label_to_color: dict[str, str] | None = None,
    figure_style: str = "manuscript",
    xlabel: str | None = None,
    ylabel: str | None = None,
    x_scale: str = "linear",
    y_scale: str = "linear",
    legend_loc: str = "best",
    legend_bbox_to_anchor: tuple[float, float] | None = None,
    legend_handletextpad: float | None = None,
    legend_borderaxespad: float | None = None,
    save: bool | str = False,
    marker: str | None = None,
    xlabel_pad: float | None = None,
    ylabel_pad: float | None = None,
    tight_layout: bool = True,
    letter: str | None = None,
) -> Figure | None:
    """
    Scatter-plot x vs y colored by categorical labels from a DataFrame.

    Args:
        df: DataFrame containing at least x, y and label_col columns.
        x: Column name to use for the x-axis (default: 'diversity').
        y: Column name to use for the y-axis (default: 'exploration_ratio').
        label_col: Column name that contains category labels.
        label_to_color: Optional mapping from label -> color. If None a default
            qualitative palette will be used.
        figure_style: Name of the plot style config to use.
        xlabel, ylabel: Axis labels. If None the column names are used.
        x_scale, y_scale: Scale for the axes ('linear' or 'log').
        legend_loc: Legend location passed to ax.legend.
        legend_bbox_to_anchor: Tuple for legend bbox_to_anchor.
        legend_handletextpad: Padding between legend handle and text.
        legend_borderaxespad: Padding between legend and axes.
        save: If truthy, save filename or bool to auto-name file.
        marker: Matplotlib marker to use (defaults to Plot.params marker).
        xlabel_pad: Padding for x-axis label (distance from axis).
        ylabel_pad: Padding for y-axis label (distance from axis).
        tight_layout: Whether to call plt.tight_layout() to expand plot to fill figure.
        letter: Optional subplot letter label.

    Returns:
        Matplotlib Figure or None if saved to disk.
    """
    if xlabel is None:
        xlabel = x
    if ylabel is None:
        ylabel = y

    p = Plot(figure_style)
    ax = p.ax1

    # Prepare colors
    labels = list(df[label_col].dropna().unique())
    if label_to_color is None:
        cmap = plt.get_cmap("tab10")
        palette = {lab: cmap(i % cmap.N) for i, lab in enumerate(labels)}
    else:
        palette = label_to_color.copy()

    # Marker and size
    _marker = marker if marker is not None else str(p.params.get("marker", "o"))
    msize = int(p.params.get("markersize", 6))
    scatter_s = max(1, msize) ** 2

    handles: list[Line2D] = []
    legend_labels: list[str] = []

    for lab in labels:
        sub = df[df[label_col] == lab]
        if sub.empty:
            continue

        # Convert to numeric arrays safely
        xvals = pd.to_numeric(sub[x], errors="coerce").to_numpy(dtype=float)
        yvals = pd.to_numeric(sub[y], errors="coerce").to_numpy(dtype=float)

        mask = np.isfinite(xvals) & np.isfinite(yvals)
        if x_scale == "log":
            mask &= xvals > 0
        if y_scale == "log":
            mask &= yvals > 0

        if not np.any(mask):
            continue

        xplot = xvals[mask]
        yplot = yvals[mask]

        color = palette.get(lab, str(p.params.get("default_color", "black")))
        ax.scatter(
            xplot,
            yplot,
            s=scatter_s,
            c=color,
            marker=_marker,
            edgecolor=str(p.params.get("default_color", "black")),
            alpha=0.8,
        )

        handle = Line2D(
            [0],
            [0],
            marker=_marker,
            color="w",
            markerfacecolor=color,
            markeredgecolor=str(p.params.get("default_color", "black")),
            markersize=msize,
            linestyle="",
        )
        handles.append(handle)
        legend_labels.append(str(lab))

    if x_scale == "log":
        ax.set_xscale("log")
    if y_scale == "log":
        ax.set_yscale("log")

    # Set axis labels with optional padding
    ax.set_xlabel(
        xlabel,
        fontdict=p.label_fontdict,
        labelpad=xlabel_pad if xlabel_pad is not None else None,
    )
    ax.set_ylabel(
        ylabel,
        fontdict=p.label_fontdict,
        labelpad=ylabel_pad if ylabel_pad is not None else None,
    )

    if handles:
        # Allow user to control legend location and margin/padding
        legend_kwargs = {"loc": legend_loc}
        if legend_bbox_to_anchor is not None:
            legend_kwargs["bbox_to_anchor"] = legend_bbox_to_anchor
        if legend_handletextpad is not None:
            legend_kwargs["handletextpad"] = legend_handletextpad
        if legend_borderaxespad is not None:
            legend_kwargs["borderaxespad"] = legend_borderaxespad
        ax.legend(handles, legend_labels, **legend_kwargs)

    fig = p.release(left_limit=None, xlabel=xlabel)

    # Add letter label if provided
    if letter:
        fig.text(
            -0.24,
            0.96,
            f"{letter})",
            transform=fig.gca().transAxes,
            fontsize=12,
            fontweight="bold",
        )

    if tight_layout:
        plt.tight_layout()

    if save:
        if not isinstance(save, str):
            save = f"{y}_vs_{x}_by_label"
        p.save_figure(f"{save}.png")
        plt.close(fig)
        return None

    return fig


def plot_grouped_boxplot(
    all_data,
    experiment_groups: list[list[str]],
    group_labels: list[str] | None = None,
    column: str = "exploration_ratio",
    figure_style: str = "manuscript",
    showfliers: bool = False,
    widths: float | list[float] | None = None,
    xlabel: str = "",
    y_label: str | None = None,
    ylim: tuple[float, float] | None = None,
    letter: str | None = None,
    save: bool | str = False,
) -> Figure | None:
    p = Plot(figure_style)
    box_data: list[list[float]] = []
    for super_name in experiment_groups:
        group_vals: list[float] = []
        for exp_name in super_name:
            seqs: dict[str, str] = all_data.read_sequences(exp_name)
            single_trajectory: pd.DataFrame = all_data.single_trajectory(seqs)
            group_vals.extend(single_trajectory.loc[:, column].to_list())
        box_data.append(group_vals)

    if group_labels is None:
        group_labels = [f"Group {i + 1}" for i in range(len(box_data))]

    ax = p.ax1
    bp = ax.boxplot(
        box_data,
        showfliers=showfliers,
        widths=widths if widths is not None else 0.6,
        patch_artist=True,
    )

    lw = float(p.params.get("linewidth", 1.0))
    default_color = str(p.params.get("default_color", "black"))

    for box in bp.get("boxes", []):
        box.set_linewidth(lw)
        box.set_facecolor("white")

    for whisker in bp.get("whiskers", []):
        whisker.set_linewidth(lw)
        whisker.set_color(default_color)

    for cap in bp.get("caps", []):
        cap.set_linewidth(lw)
        cap.set_color(default_color)

    for median in bp.get("medians", []):
        median.set_linewidth(lw)
        median.set_color(default_color)

    for fl in bp.get("fliers", []):
        try:
            fl.set_marker("o")
            fl.set_alpha(0.6)
        except Exception:
            pass

    ax.set_xticklabels(group_labels, rotation=30, ha="right")
    ax.set_ylabel(
        y_label if y_label is not None else column, fontdict=p.label_fontdict
    )
    if ylim is not None:
        ax.set_ylim(ylim)
    fig = p.release(left_limit=None, right_limit=None, xlabel=xlabel)

    if letter:
        fig.text(
            -0.24,
            0.96,
            f"{letter})",
            transform=fig.gca().transAxes,
            fontsize=12,
            fontweight="bold",
        )

    plt.tight_layout()

    if save:
        if not isinstance(save, str):
            save = str(letter) if letter else "plot"
        p.save_figure(f"{save}.png")
        plt.close(fig)
        return None
    return fig
