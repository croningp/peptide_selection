import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.widgets import Button, Slider
from matplotlib.colors import ListedColormap, BoundaryNorm
from .bigram_analyzer import BigramAnalyzer


class HeatmapVisualizer:
    """
    Heatmap visualizer that works with BigramAnalyzer results.
    Focuses purely on visualization, leaving all computation to BigramAnalyzer.
    """

    def __init__(
        self,
        datasets=None,
        patterns=["G", "A", "V", "P", "H"],
        analyzer=None,
        bigram_matrices=None,
        global_vmin=None,
        global_vmax=None,
    ):
        """
        Initialize the heatmap visualizer.

        Args:
            datasets: Dictionary with dataset names as keys and data as values
            patterns: List of patterns to track (for backward compatibility with 'letters')
            analyzer: Pre-configured BigramAnalyzer instance
            bigram_matrices: Pre-computed bigram matrices (optional)
            global_vmin: Global minimum value for colormap (optional)
            global_vmax: Global maximum value for colormap (optional)
        """
        self.current_dataset_index = 0

        if analyzer is not None:
            # Use provided analyzer
            self.analyzer = analyzer
            self.patterns = analyzer.patterns
            self.letters = analyzer.letters  # For backward compatibility
            self.pattern_type = analyzer.pattern_type

            if datasets is not None:
                self.datasets = datasets
                analysis_results = analyzer.compare_datasets(datasets)
                self.bigram_matrices = analysis_results["matrices"]
                self.global_vmin = analysis_results["global_min"]
                self.global_vmax = analysis_results["global_max"]
            else:
                self.bigram_matrices = bigram_matrices or {}
                self.global_vmin = global_vmin or 0
                self.global_vmax = global_vmax or 1
                self.datasets = {
                    name: [] for name in self.bigram_matrices.keys()
                }

        elif datasets is not None:
            # Create analyzer from datasets using patterns parameter
            self.analyzer = BigramAnalyzer(patterns)
            self.datasets = datasets
            analysis_results = self.analyzer.compare_datasets(datasets)
            self.bigram_matrices = analysis_results["matrices"]
            self.global_vmin = analysis_results["global_min"]
            self.global_vmax = analysis_results["global_max"]
            self.patterns = self.analyzer.patterns
            self.letters = self.analyzer.letters
            self.pattern_type = self.analyzer.pattern_type

        elif bigram_matrices is not None:
            # Use pre-computed matrices
            self.analyzer = BigramAnalyzer(patterns)
            self.bigram_matrices = bigram_matrices
            self.global_vmin = global_vmin or 0
            self.global_vmax = global_vmax or 1
            self.datasets = {name: [] for name in bigram_matrices.keys()}
            self.patterns = patterns
            self.letters = patterns
            self.pattern_type = self.analyzer.pattern_type

        else:
            raise ValueError(
                "Must provide either datasets, analyzer, or bigram_matrices"
            )

    @classmethod
    def from_analysis_results(cls, analysis_results, datasets=None):
        """
        Create visualizer from BigramAnalyzer results.

        Args:
            analysis_results: Results from BigramAnalyzer.compare_datasets()
            datasets: Original datasets (optional)

        Returns:
            HeatmapVisualizer: Configured visualizer instance
        """
        instance = cls.__new__(cls)
        instance.current_dataset_index = 0
        instance.datasets = datasets
        instance.bigram_matrices = analysis_results["matrices"]
        instance.global_vmin = analysis_results["global_min"]
        instance.global_vmax = analysis_results["global_max"]
        instance.patterns = analysis_results["patterns"]
        instance.letters = analysis_results["letters"]
        instance.pattern_type = analysis_results["pattern_type"]
        instance.analyzer = None  # No need to store analyzer for this method
        return instance

    def _get_axis_labels(self):
        """Get appropriate axis labels based on pattern type."""
        if self.pattern_type == "single_char":
            return "Second Amino Acid", "First Amino Acid"
        elif self.pattern_type == "two_char":
            return (
                "Second Two-Amino Acid Pattern",
                "First Two-Amino Acid Pattern",
            )
        else:
            return "Second Pattern", "First Pattern"

    def _get_title_suffix(self):
        """Get title suffix based on pattern type."""
        if self.pattern_type == "single_char":
            return "Bigram Analysis"
        elif self.pattern_type == "two_char":
            return "Two-Letter Bigram Analysis"
        else:
            return "Pattern Bigram Analysis"

    def create_publication_ready_plot(
        self,
        dataset_name=None,
        figsize=(8, 6),
        style="whitegrid",
        context="paper",
        font_scale=1.2,
        cmap="viridis",
        title_fontsize=14,
        label_fontsize=12,
        tick_fontsize=10,
        show_values=True,
        linewidths=0.5,
        fmt=".0f",
        segmented_cbar=False,
        n_segments=5,
        transparent_zeros=False,
        zero_alpha=0.3,
    ):
        """Create a publication-ready heatmap plot with optional segmented colorbar."""
        # Set seaborn style
        sns.set_style(style)
        sns.set_context(context, font_scale=font_scale)

        # Get dataset
        if dataset_name is None:
            dataset_name = list(self.bigram_matrices.keys())[0]

        matrix = self.bigram_matrices[dataset_name]
        xlabel, ylabel = self._get_axis_labels()
        title_suffix = self._get_title_suffix()

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Handle segmented vs continuous colorbar
        if segmented_cbar:
            # Create segmented colormap
            segmented_cmap, norm, boundaries, tick_positions, tick_labels = (
                self._create_segmented_colormap(
                    cmap, n_segments, self.global_vmin, self.global_vmax
                )
            )

            # Create heatmap with segmented colormap
            sns.heatmap(
                matrix,
                annot=show_values,
                fmt=fmt,
                cmap=segmented_cmap,
                norm=norm,
                xticklabels=self.patterns,
                yticklabels=self.patterns,
                square=True,
                ax=ax,
                linewidths=linewidths,
                linecolor="white",
                vmin=self.global_vmin,
                vmax=self.global_vmax,
                cbar_kws={"label": "Bigram Count", "shrink": 0.8, "aspect": 20},
            )

            # Get the colorbar and customize it
            cbar = ax.collections[0].colorbar
            cbar.set_ticks(tick_positions)
            cbar.set_ticklabels(tick_labels)
            cbar.ax.tick_params(
                length=0,
                width=0,  # Remove tick marks completely
                which="both",  # Apply to both major and minor ticks
                left=False,  # Remove left ticks
                right=False,  # Remove right ticks
                top=False,  # Remove top ticks
                bottom=False,  # Remove bottom ticks
            )
            # Add border around colorbar
            for spine in cbar.ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.5)
                spine.set_edgecolor("black")

        else:
            # Create continuous heatmap (original behavior)
            sns.heatmap(
                matrix,
                annot=show_values,
                fmt=fmt,
                cmap=cmap,
                xticklabels=self.patterns,
                yticklabels=self.patterns,
                square=True,
                ax=ax,
                linewidths=linewidths,
                linecolor="white",
                vmin=self.global_vmin,
                vmax=self.global_vmax,
                cbar_kws={"label": "Bigram Count", "shrink": 0.8, "aspect": 20},
            )

        # Styling
        seg_suffix = " (Segmented)" if segmented_cbar else ""
        ax.set_title(
            f"{dataset_name.title()} - {title_suffix}{seg_suffix}",
            fontsize=title_fontsize,
            fontweight="bold",
            pad=20,
        )
        ax.set_xlabel(xlabel, fontsize=label_fontsize, labelpad=10)
        ax.set_ylabel(ylabel, fontsize=label_fontsize, labelpad=10)

        # Handle transparent zeros if enabled
        if transparent_zeros and show_values:
            for text in ax.texts:
                try:
                    value = float(text.get_text())
                    if value == 0:
                        text.set_alpha(zero_alpha)
                except ValueError:
                    pass

        # Tick styling
        ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
        plt.setp(
            ax.get_xticklabels(),
            rotation=45 if self.pattern_type == "two_char" else 0,
            ha="center",
        )
        plt.setp(ax.get_yticklabels(), rotation=0, va="center")

        plt.tight_layout()
        return fig, ax

    def create_horizontal_stack_plot(
        self, figsize=None, segmented_cbar=False, n_segments=5
    ):
        """Create horizontal stacked heatmaps for all datasets."""
        n_datasets = len(self.bigram_matrices)

        # Auto-adjust figure size based on number of datasets and pattern type
        if figsize is None:
            width_per_plot = 6 if self.pattern_type == "two_char" else 4
            figsize = (width_per_plot * n_datasets, 5)

        fig, axes = plt.subplots(1, n_datasets, figsize=figsize)

        if n_datasets == 1:
            axes = [axes]

        xlabel, ylabel = self._get_axis_labels()

        for idx, (name, matrix) in enumerate(self.bigram_matrices.items()):
            ax = axes[idx]

            if segmented_cbar:
                # Create segmented colormap
                segmented_cmap, norm, _, tick_positions, tick_labels = (
                    self._create_segmented_colormap(
                        "YlOrRd", n_segments, self.global_vmin, self.global_vmax
                    )
                )

                sns.heatmap(
                    matrix,
                    annot=True,
                    fmt=".0f",
                    cmap=segmented_cmap,
                    norm=norm,
                    xticklabels=self.patterns,
                    yticklabels=self.patterns,
                    square=True,
                    ax=ax,
                    cbar_kws={"label": "Bigram Count"},
                    vmin=self.global_vmin,
                    vmax=self.global_vmax,
                )

                # Customize colorbar if it exists
                if (
                    hasattr(ax.collections[0], "colorbar")
                    and ax.collections[0].colorbar is not None
                ):
                    cbar = ax.collections[0].colorbar
                    cbar.set_ticks(tick_positions)
                    cbar.set_ticklabels(tick_labels)
                    cbar.ax.tick_params(
                        length=0,
                        width=0,  # Remove tick marks completely
                        which="both",  # Apply to both major and minor ticks
                        left=False,  # Remove left ticks
                        right=False,  # Remove right ticks
                        top=False,  # Remove top ticks
                        bottom=False,  # Remove bottom ticks
                    )
                    # Add border around colorbar
                    for spine in cbar.ax.spines.values():
                        spine.set_visible(True)
                        spine.set_linewidth(0.5)
                        spine.set_edgecolor("black")
            else:
                sns.heatmap(
                    matrix,
                    annot=True,
                    fmt=".0f",
                    cmap="YlOrRd",
                    xticklabels=self.patterns,
                    yticklabels=self.patterns,
                    square=True,
                    ax=ax,
                    cbar_kws={"label": "Bigram Count"},
                    vmin=self.global_vmin,
                    vmax=self.global_vmax,
                )

            seg_suffix = " (Seg)" if segmented_cbar else ""
            ax.set_title(
                f"{name.title()}{seg_suffix}", fontsize=12, fontweight="bold"
            )
            ax.set_xlabel("Second Pattern", fontsize=10)
            if idx == 0:
                ax.set_ylabel("First Pattern", fontsize=10)
            else:
                ax.set_ylabel("")

            # Rotate labels for two-char patterns
            if self.pattern_type == "two_char":
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
                plt.setp(ax.get_yticklabels(), rotation=0)

        plt.tight_layout()
        return fig, axes

    def _create_segmented_colormap(self, base_cmap, n_segments, vmin, vmax):
        """
        Create a segmented colormap with discrete boundaries.

        Args:
            base_cmap: Base colormap name or object
            n_segments: Number of discrete segments
            vmin: Minimum value for the colormap
            vmax: Maximum value for the colormap

        Returns:
            tuple: (segmented_colormap, norm, boundaries, tick_positions, tick_labels)
        """
        # Handle edge cases where vmin == vmax
        if vmin == vmax:
            if vmin == 0:
                vmax = 1  # Default range for zero data
            else:
                # Expand range by 10% around the value
                delta = abs(vmin * 0.1) if vmin != 0 else 0.5
                vmin = vmin - delta
                vmax = vmax + delta

        # Ensure we have a valid range
        if vmax <= vmin:
            vmax = vmin + 1

        # Get base colormap
        if isinstance(base_cmap, str):
            base = plt.cm.get_cmap(base_cmap)
        else:
            base = base_cmap

        # Create boundaries for segments
        boundaries = np.linspace(vmin, vmax, n_segments + 1)

        # Calculate tick positions (middle of each segment)
        tick_positions = []
        tick_labels = []
        for i in range(n_segments):
            mid_point = (boundaries[i] + boundaries[i + 1]) / 2
            tick_positions.append(mid_point)
            # Format the label with no decimal places
            tick_labels.append(f"{mid_point:.0f}")

        # Create discrete colors
        colors = base(np.linspace(0, 1, n_segments))
        segmented_cmap = ListedColormap(colors)

        # Create boundary normalization with explicit handling of edge cases
        try:
            norm = BoundaryNorm(
                boundaries, segmented_cmap.N, clip=True, extend="neither"
            )
        except Exception as e:
            # Fallback to simple normalization if BoundaryNorm fails
            print(
                f"Warning: BoundaryNorm failed, using simple normalization. Error: {e}"
            )
            norm = plt.Normalize(vmin=vmin, vmax=vmax, clip=True)

        return segmented_cmap, norm, boundaries, tick_positions, tick_labels

    def create_clean_plot(
        self,
        dataset_name: str | None = None,
        figsize: tuple[float, float] = (6, 5),
        cmap: str = "viridis",
        tight_layout: bool = True,
        show_values: bool = True,
        fmt: str = ".0f",
        linewidths: float = 0.5,
        cbar_width: float = 0.03,
        cbar_height: float = 0.8,
        cbar_pad: float = 0.05,
        cbar_y_pos: float = 0.15,
        title_pad: int | float = 15,
        label_pad: int | float = 8,
        title_fontsize: int | float = 12,
        label_fontsize: int | float = 10,
        tick_fontsize: int | float = 9,
        annot_fontsize: int | float = 8,
        adaptive_colors: bool = False,
        disable_title: bool = False,
        cbar_label_fontsize: int | float = 10,
        segmented_cbar: bool = False,
        n_segments: int = 5,
        transparent_zeros: bool = False,
        zero_alpha: float = 0.3,
    ):
        """
        Create a clean, non-interactive heatmap plot with improved error handling.
        """
        # Get dataset
        if dataset_name is None:
            dataset_name = list(self.bigram_matrices.keys())[0]

        matrix = self.bigram_matrices[dataset_name]
        xlabel, ylabel = self._get_axis_labels()

        # Create figure and manually create colorbar axis
        fig, ax = plt.subplots(figsize=figsize)

        # Apply tight_layout BEFORE creating manual axes to avoid conflicts
        if tight_layout:
            plt.tight_layout()

        # Adjust main plot to make room for colorbar
        plt.subplots_adjust(right=0.85)

        # Create custom colorbar axis with desired width and position
        cbar_x = 0.85 + cbar_pad  # Move it right by cbar_pad amount
        cbar_ax = fig.add_axes(
            (cbar_x, cbar_y_pos, cbar_width, 0.7 * cbar_height)
        )

        # Determine color scale based on adaptive_colors parameter
        if adaptive_colors:
            # Use adaptive coloring - scale to this specific matrix
            vmin = float(matrix.min())
            vmax = float(matrix.max())
            cbar_label = "Count"
        else:
            # Use global coloring - scale across all datasets
            vmin = float(self.global_vmin)
            vmax = float(self.global_vmax)
            cbar_label = "Count"

        # Additional safety check for edge cases
        if np.isnan(vmin) or np.isnan(vmax) or np.isinf(vmin) or np.isinf(vmax):
            print(
                f"Warning: Invalid vmin/vmax values detected. vmin={vmin}, vmax={vmax}"
            )
            vmin = 0
            vmax = max(
                1, float(matrix.max()) if not np.isnan(matrix.max()) else 1
            )

        # Handle segmented vs continuous colorbar
        if segmented_cbar:
            try:
                # Create segmented colormap
                (
                    segmented_cmap,
                    norm,
                    boundaries,
                    tick_positions,
                    tick_labels,
                ) = self._create_segmented_colormap(
                    cmap, n_segments, vmin, vmax
                )

                # Create heatmap with segmented colormap
                sns.heatmap(
                    matrix,
                    annot=show_values,
                    fmt=fmt,
                    cmap=segmented_cmap,
                    norm=norm,
                    xticklabels=self.patterns,
                    yticklabels=self.patterns,
                    square=True,
                    ax=ax,
                    linewidths=linewidths,
                    linecolor="white",
                    vmin=vmin,
                    vmax=vmax,
                    cbar_ax=cbar_ax,
                    cbar_kws={"label": cbar_label},
                    annot_kws={"fontsize": annot_fontsize},
                )

                # Customize the segmented colorbar
                try:
                    cbar_ax.set_yticks(tick_positions)
                    cbar_ax.set_yticklabels(tick_labels)
                    cbar_ax.tick_params(
                        length=0,
                        width=0,  # Remove tick marks completely
                        which="both",  # Apply to both major and minor ticks
                        left=False,  # Remove left ticks
                        right=False,  # Remove right ticks
                        top=False,  # Remove top ticks
                        bottom=False,  # Remove bottom ticks
                    )
                    # Add border around colorbar
                    for spine in cbar_ax.spines.values():
                        spine.set_visible(True)
                        spine.set_linewidth(0.5)
                        spine.set_edgecolor("black")
                except Exception as e:
                    print(
                        f"Warning: Failed to customize segmented colorbar: {e}"
                    )

            except Exception as e:
                print(
                    f"Warning: Segmented colorbar failed, falling back to continuous. Error: {e}"
                )
                # Fallback to continuous colorbar
                sns.heatmap(
                    matrix,
                    annot=show_values,
                    fmt=fmt,
                    cmap=cmap,
                    xticklabels=self.patterns,
                    yticklabels=self.patterns,
                    square=True,
                    ax=ax,
                    linewidths=linewidths,
                    linecolor="white",
                    vmin=vmin,
                    vmax=vmax,
                    cbar_ax=cbar_ax,
                    cbar_kws={"label": cbar_label},
                    annot_kws={"fontsize": annot_fontsize},
                )

        else:
            # Create continuous heatmap (original behavior)
            sns.heatmap(
                matrix,
                annot=show_values,
                fmt=fmt,
                cmap=cmap,
                xticklabels=self.patterns,
                yticklabels=self.patterns,
                square=True,
                ax=ax,
                linewidths=linewidths,
                linecolor="white",
                vmin=vmin,
                vmax=vmax,
                cbar_ax=cbar_ax,
                cbar_kws={"label": cbar_label},
                annot_kws={"fontsize": annot_fontsize},
            )

        # Style the colorbar
        try:
            cbar_ax.tick_params(labelsize=tick_fontsize)
            cbar_ax.set_ylabel(cbar_label, fontsize=cbar_label_fontsize)
        except Exception as e:
            print(f"Warning: Failed to style colorbar: {e}")

        # Handle text color for grayscale colormaps
        if show_values and cmap in ["Greys", "gray", "grey"]:
            try:
                import matplotlib as mpl

                norm_for_text = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

                # Update text colors based on background darkness
                for text in ax.texts:
                    try:
                        value = float(text.get_text())
                        normalized_value = norm_for_text(value)
                        text_color = (
                            "white" if normalized_value > 0.5 else "black"
                        )
                        text.set_color(text_color)
                    except ValueError:
                        pass
            except Exception as e:
                print(f"Warning: Failed to adjust text colors: {e}")

        if not disable_title:
            # Clean styling
            title_suffix = " (Adaptive)" if adaptive_colors else ""
            seg_suffix = " (Segmented)" if segmented_cbar else ""

            ax.set_title(
                f"{dataset_name.title()}{title_suffix}{seg_suffix}",
                fontweight="bold",
                pad=title_pad,
                fontsize=title_fontsize,
            )
        ax.set_xlabel(xlabel, labelpad=label_pad, fontsize=label_fontsize)
        ax.set_ylabel(ylabel, labelpad=label_pad, fontsize=label_fontsize)

        # Set tick font sizes
        ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)

        # Handle label rotation for multi-character patterns
        if self.pattern_type == "two_char":
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            plt.setp(ax.get_yticklabels(), rotation=0, va="center")

        # Apply transparent zeros as the final step
        if transparent_zeros and show_values:
            try:
                # Now modify zero values without forcing interactive drawing
                for text in ax.texts:
                    try:
                        value_str = text.get_text().strip()

                        # Try different ways to detect zero
                        if value_str in ["0", "0.0", "0.", ".0"]:
                            text.set_alpha(zero_alpha)
                        elif value_str.replace(".", "").replace("0", "") == "":
                            # Handle cases like "0.000"
                            try:
                                if float(value_str) == 0.0:
                                    text.set_alpha(zero_alpha)
                            except ValueError:
                                pass

                    except (ValueError, AttributeError):
                        pass
            except Exception as e:
                print(f"Warning: Failed to apply transparent zeros: {e}")

        return fig, ax
