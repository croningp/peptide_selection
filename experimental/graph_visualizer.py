"""
Vertical Assembly Graph Visualization

A module for creating vertical layout visualizations of NetworkX graphs
based on assembly indices, with support for observed/unobserved node coloring
and curved edges for long-distance connections.
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math
from collections import defaultdict
import pandas as pd


def draw_curved_edges(
    graph: nx.Graph, 
    pos: dict[str, tuple[float, float]], 
    node_assemblies: dict[str, int], 
    ax,
    alpha: float = 0.1, 
    edge_color: str = 'gray', 
    width: float = 1.0
) -> tuple[list[float], list[float]]:
    """
    Draw curved edges that bend around intermediate levels for long-distance connections.
    
    Args:
        graph: NetworkX graph
        pos: Node positions as {node: (x, y)}
        node_assemblies: Node assembly indices as {node: assembly_index}
        ax: Matplotlib axes object
        alpha: Edge transparency
        edge_color: Edge color
        width: Edge line width
    
    Returns:
        Tuple of (all_x_coords, all_y_coords) for proper axis limiting
    """
    all_x_coords = []
    all_y_coords = []
    
    for edge in graph.edges():
        source, target = edge
        
        # Get assembly indices for source and target
        source_assembly = node_assemblies.get(source, 0)
        target_assembly = node_assemblies.get(target, 0)
        
        # Get positions
        source_pos = pos[source]
        target_pos = pos[target]
        
        # Calculate assembly difference
        assembly_diff = abs(source_assembly - target_assembly)
        
        if assembly_diff <= 1:
            # Short distance - draw straight line
            ax.plot([source_pos[0], target_pos[0]], 
                   [source_pos[1], target_pos[1]], 
                   color=edge_color, alpha=alpha, linewidth=width)
            # Add coordinates to bounds
            all_x_coords.extend([source_pos[0], target_pos[0]])
            all_y_coords.extend([source_pos[1], target_pos[1]])
        else:
            # Long distance - draw curved line
            if source_pos[1] < target_pos[1]:  # source is lower
                lower_pos, higher_pos = source_pos, target_pos
            else:
                lower_pos, higher_pos = target_pos, source_pos
            
            # Create curve that bends outward to avoid crossing nodes
            mid_y = (lower_pos[1] + higher_pos[1]) / 2
            x_diff = higher_pos[0] - lower_pos[0]
            
            # Calculate bend amount based on assembly difference
            bend_factor = min(assembly_diff * 2, 10)  # Cap the bend
            
            if x_diff >= 0:  # target is to the right, bend right
                control_x = max(lower_pos[0], higher_pos[0]) + bend_factor
            else:  # target is to the left, bend left
                control_x = min(lower_pos[0], higher_pos[0]) - bend_factor
            
            # Create bezier curve using parametric equations
            t_values = np.linspace(0, 1, 50)
            control_point = (control_x, mid_y)
            
            x_curve = []
            y_curve = []
            
            for t in t_values:
                x = (1-t)**2 * lower_pos[0] + 2*(1-t)*t * control_point[0] + t**2 * higher_pos[0]
                y = (1-t)**2 * lower_pos[1] + 2*(1-t)*t * control_point[1] + t**2 * higher_pos[1]
                x_curve.append(x)
                y_curve.append(y)
            
            ax.plot(x_curve, y_curve, color=edge_color, alpha=alpha, linewidth=width)
            
            # Add all curve coordinates to bounds
            all_x_coords.extend(x_curve)
            all_y_coords.extend(y_curve)
    
    return all_x_coords, all_y_coords


def vertical_assembly_layout(
    graph: nx.Graph, 
    df, 
    peptide_col: str = 'peptide', 
    assembly_col: str = 'assembly_index', 
    scale: float = 1.0, 
    horizontal_spread: float = 1.0, 
    reverse_order: bool = False, 
    observed_nodes: list[str] | None = None,
    width_by_node_count: bool = True, 
    width_by_observed: bool = True, 
    min_width_factor: float = 0.5, 
    max_width_factor: float = 3.0, 
    force_max_assembly: int | None = None
) -> tuple[dict[str, tuple[float, float]], dict[str, int]]:
    """
    Create vertical layout where Y position corresponds to assembly_index.
    Each assembly level is exactly 'scale' units apart vertically.
    
    Args:
        graph: NetworkX graph
        df: DataFrame with peptide and assembly_index columns
        peptide_col: Name of peptide column in dataframe
        assembly_col: Name of assembly index column in dataframe
        scale: Vertical distance between assembly levels
        horizontal_spread: Base horizontal spread factor
        reverse_order: If True, higher assembly index = lower Y position
        observed_nodes: List of observed node names for clustering
        width_by_node_count: Scale width by total node count at each level
        width_by_observed: Scale width by observed node count at each level
        min_width_factor: Minimum width multiplier
        max_width_factor: Maximum width multiplier
        force_max_assembly: Force normalization to this max assembly index
    
    Returns:
        Tuple of (node_positions, node_assemblies)
    """
    
    # Create mapping from peptide to assembly_index
    peptide_to_assembly = dict(zip(df[peptide_col], df[assembly_col]))
    
    # Get assembly indices for nodes that exist in both graph and dataframe
    node_assemblies = {}
    missing_nodes = []
    
    for node in graph.nodes():
        if node in peptide_to_assembly:
            node_assemblies[node] = peptide_to_assembly[node]
        else:
            missing_nodes.append(node)
    
    if missing_nodes:
        print(f"Warning: {len(missing_nodes)} nodes not found in dataframe. They will be placed at the top.")
        print(f"Sample missing nodes: {missing_nodes[:5]}")
    
    # Get assembly index range
    if node_assemblies:
        assembly_values = list(node_assemblies.values())
        max_assembly = max(assembly_values)
        min_assembly = min(assembly_values)
    else:
        max_assembly = min_assembly = 0
    
    # Group nodes by assembly index
    assembly_groups = defaultdict(list)
    for node, assembly_idx in node_assemblies.items():
        assembly_groups[assembly_idx].append(node)
    
    # Add missing nodes to top
    if missing_nodes:
        top_assembly = max_assembly + 1
        assembly_groups[top_assembly] = missing_nodes
    
    pos = {}
    observed_set = set(observed_nodes) if observed_nodes else set()
    
    # Sort assembly indices
    sorted_assemblies = sorted(assembly_groups.keys(), reverse=reverse_order)
    
    # Calculate width factors
    width_factors = {}
    
    if width_by_observed and observed_nodes:
        # Width based on number of observed nodes at each level
        level_observed_counts = {}
        for assembly_idx, nodes in assembly_groups.items():
            observed_count = len([node for node in nodes if node in observed_set])
            level_observed_counts[assembly_idx] = observed_count
        
        max_observed = max(level_observed_counts.values()) if level_observed_counts else 1
        min_observed = min(level_observed_counts.values()) if level_observed_counts else 0
        
        for assembly_idx in assembly_groups.keys():
            observed_count = level_observed_counts[assembly_idx]
            
            if max_observed == 0:
                width_factor = min_width_factor
            elif max_observed == min_observed:
                width_factor = (min_width_factor + max_width_factor) / 2
            else:
                normalized_count = (observed_count - min_observed) / (max_observed - min_observed)
                width_factor = min_width_factor + normalized_count * (max_width_factor - min_width_factor)
            
            width_factors[assembly_idx] = width_factor
            
    elif width_by_node_count:
        # Width based on total number of nodes at each level
        level_node_counts = {assembly_idx: len(nodes) for assembly_idx, nodes in assembly_groups.items()}
        max_nodes = max(level_node_counts.values()) if level_node_counts else 1
        min_nodes = min(level_node_counts.values()) if level_node_counts else 1
        
        for assembly_idx in assembly_groups.keys():
            node_count = level_node_counts[assembly_idx]
            
            if max_nodes == min_nodes:
                width_factor = (min_width_factor + max_width_factor) / 2
            else:
                normalized_count = (node_count - min_nodes) / (max_nodes - min_nodes)
                width_factor = min_width_factor + normalized_count * (max_width_factor - min_width_factor)
            
            width_factors[assembly_idx] = width_factor
    else:
        # Use fixed multipliers
        spread_multipliers = [1, 2, 3, 6, 10]
        for level_idx, assembly_idx in enumerate(sorted_assemblies):
            if level_idx < len(spread_multipliers):
                width_factors[assembly_idx] = spread_multipliers[level_idx]
            else:
                width_factors[assembly_idx] = spread_multipliers[-1]
    
    # Position nodes
    for level_idx, assembly_idx in enumerate(sorted_assemblies):
        nodes_at_assembly = assembly_groups[assembly_idx]
        
        # Calculate Y position - ABSOLUTE positioning (no normalization)
        y_pos = assembly_idx * scale
        
        # Handle missing nodes
        if assembly_idx not in [a for a in node_assemblies.values()]:
            max_actual_assembly = max(node_assemblies.values()) if node_assemblies else 0
            y_pos = (max_actual_assembly + 1) * scale
        
        # Calculate horizontal spread for this level
        level_spread = horizontal_spread * width_factors[assembly_idx]
        
        # Position nodes horizontally
        if observed_nodes:
            observed_at_level = [node for node in nodes_at_assembly if node in observed_set]
            unobserved_at_level = [node for node in nodes_at_assembly if node not in observed_set]
            
            # Unobserved nodes first (leftmost), then observed nodes (rightmost)
            # This puts unobserved "on top" visually when they overlap
            ordered_nodes = unobserved_at_level + observed_at_level
            
            # Distribute evenly across the horizontal range
            n_total = len(ordered_nodes)
            if n_total == 1:
                x_positions = [0]
            else:
                x_positions = np.linspace(-level_spread/2, level_spread/2, n_total)
            
            for j, node in enumerate(ordered_nodes):
                pos[node] = (x_positions[j], y_pos)
        else:
            # No clustering - spread all nodes evenly
            n_nodes = len(nodes_at_assembly)
            if n_nodes == 1:
                x_positions = [0]
            else:
                x_positions = np.linspace(-level_spread/2, level_spread/2, n_nodes)
            
            for j, node in enumerate(nodes_at_assembly):
                pos[node] = (x_positions[j], y_pos)
    
    return pos, node_assemblies


def visualize_vertical_assembly_graph(
    graph: nx.Graph,
    df,
    peptide_col: str = 'peptide',
    assembly_col: str = 'assembly_index',
    observed_nodes: list[str] | None = None,
    observed_color: str = 'lightcoral',
    unobserved_color: str = 'gray',
    node_size: float | str = 'auto',
    min_node_size: float = 30,
    max_node_size: float = 150,
    node_edge_width: float = 0.5,
    node_edge_color: str = 'black',
    node_shape: str = 'circle',
    figsize: tuple[float, float] = (4, 6),
    font_size: int = 6,
    reverse_order: bool = False,
    width_by_node_count: bool = True,
    width_by_observed: bool = True,
    min_width_factor: float = 0.5,
    max_width_factor: float = 3.0,
    vertical_scale: float = 1.0,
    force_ylim: tuple[float, float] | None = None,
    force_max_assembly: int | None = None,
    show_statistics: bool = True
) -> tuple[dict[str, tuple[float, float]], dict[str, int]]:
    """
    Visualize the graph with vertical layout based on assembly_index from dataframe.
    
    Args:
        graph: NetworkX graph to visualize
        df: DataFrame with peptide and assembly_index columns
        peptide_col: Name of peptide column in dataframe
        assembly_col: Name of assembly index column in dataframe
        observed_nodes: List of observed node names for coloring
        observed_color: Color for observed nodes (e.g., 'red', '#FF0000', (1,0,0))
        unobserved_color: Color for unobserved nodes (e.g., 'gray', '#808080')
        node_size: Node sizing method:
            - 'auto': Scale by assembly index (lower = larger)
            - 'uniform': All nodes same size (uses min_node_size)
            - float: All nodes this specific size
        min_node_size: Minimum node size (for 'auto' mode)
        max_node_size: Maximum node size (for 'auto' mode)
        node_edge_width: Thickness of node border/edge lines
        node_edge_color: Color of node border/edge lines
        node_shape: Shape of nodes ('circle' or 'rectangle')
        figsize: Figure size as (width, height)
        font_size: Font size for labels
        reverse_order: If True, higher assembly index = lower Y position
        width_by_node_count: Scale width by total node count at each level
        width_by_observed: Scale width by observed node count at each level
        min_width_factor: Minimum width multiplier
        max_width_factor: Maximum width multiplier
        vertical_scale: Vertical distance between assembly levels
        force_ylim: Force Y-axis limits as (min, max) for consistency
        force_max_assembly: Force normalization to this max assembly index
        show_statistics: Whether to print layer statistics
    
    Returns:
        Tuple of (node_positions, node_assemblies)
    """
    
    # Create layout using assembly indices
    pos, node_assemblies = vertical_assembly_layout(
        graph, df, peptide_col, assembly_col, 
        scale=vertical_scale, horizontal_spread=3.0, 
        reverse_order=reverse_order, observed_nodes=observed_nodes,
        width_by_node_count=width_by_node_count,
        width_by_observed=width_by_observed,
        min_width_factor=min_width_factor,
        max_width_factor=max_width_factor,
        force_max_assembly=force_max_assembly
    )
    
    # Get node colors based on observed/unobserved
    if observed_nodes is not None:
        observed_set = set(observed_nodes)
        node_colors = [observed_color if node in observed_set else unobserved_color 
                      for node in graph.nodes()]
    else:
        node_colors = [graph.nodes[node].get('color', 'gray') for node in graph.nodes()]
    
    # Calculate node sizes based on mode
    if isinstance(node_size, (int, float)):
        # Fixed size for all nodes
        node_sizes = [float(node_size)] * len(graph.nodes())
    elif node_size == 'uniform':
        # Uniform size for all nodes
        node_sizes = [min_node_size] * len(graph.nodes())
    elif node_size == 'auto' and node_assemblies:
        # Scale by assembly index (lower index = larger nodes)
        assembly_values = list(node_assemblies.values())
        min_assembly = min(assembly_values)
        max_assembly = max(assembly_values)
        
        node_sizes = []
        for node in graph.nodes():
            if node in node_assemblies:
                if max_assembly > min_assembly:
                    norm_assembly = (node_assemblies[node] - min_assembly) / (max_assembly - min_assembly)
                    size = min_node_size + (max_node_size - min_node_size) * (1 - norm_assembly)
                else:
                    size = (min_node_size + max_node_size) / 2
            else:
                size = min_node_size
            node_sizes.append(size)
    else:
        # Fallback to uniform size
        node_sizes = [50] * len(graph.nodes())
    
    plt.figure(figsize=figsize)
    ax = plt.gca()
    
    # Draw curved edges
    edge_x_coords, edge_y_coords = draw_curved_edges(
        graph, pos, node_assemblies, ax, alpha=0.1, edge_color='gray', width=1.0
    )
    
    # Draw nodes based on shape
    if node_shape == 'rectangle':
        # Use custom rounded rectangles - don't draw circles at all
        node_positions = [pos[node] for node in graph.nodes()]
        draw_rounded_rectangles(ax, node_positions, node_colors, node_sizes, 
                              node_edge_width, node_edge_color)
    else:
        # Use NetworkX circles with proper border handling
        if node_edge_width == 0:
            # No borders - set edgecolors to None
            nx.draw_networkx_nodes(
                graph, pos, 
                node_color=node_colors,
                node_size=node_sizes,
                alpha=0.8,
                edgecolors='none'
            )
        else:
            # With borders
            nx.draw_networkx_nodes(
                graph, pos, 
                node_color=node_colors,
                node_size=node_sizes,
                alpha=0.8,
                edgecolors=node_edge_color,
                linewidths=node_edge_width
            )
    
    plt.axis('off')
    
    # Set axis limits
    if pos:
        # Get coordinates for X-axis
        node_x_coords = [coord[0] for coord in pos.values()]
        all_x_coords = node_x_coords + edge_x_coords
        
        # Set X limits with margin
        x_margin = (max(all_x_coords) - min(all_x_coords)) * 0.1 if max(all_x_coords) != min(all_x_coords) else 1
        plt.xlim(min(all_x_coords) - x_margin, max(all_x_coords) + x_margin)
        
        # Set Y limits
        if force_ylim is not None:
            plt.ylim(force_ylim[0], force_ylim[1])
        else:
            node_y_coords = [coord[1] for coord in pos.values()]
            all_y_coords = node_y_coords + edge_y_coords
            y_margin = (max(all_y_coords) - min(all_y_coords)) * 0.05 if max(all_y_coords) != min(all_y_coords) else 1
            plt.ylim(min(all_y_coords) - y_margin, max(all_y_coords) + y_margin)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics about nodes per layer
    if show_statistics and node_assemblies:
        print("\n=== Assembly Layer Statistics ===")
        observed_set = set(observed_nodes) if observed_nodes else set()
        
        # Group by assembly index
        assembly_groups = defaultdict(list)
        for node, assembly_idx in node_assemblies.items():
            assembly_groups[assembly_idx].append(node)
        
        # Sort by assembly index
        sorted_assemblies = sorted(assembly_groups.keys(), reverse=reverse_order)
        
        total_observed = 0
        total_unobserved = 0
        
        for assembly_idx in sorted_assemblies:
            nodes_at_level = assembly_groups[assembly_idx]
            observed_count = len([node for node in nodes_at_level if node in observed_set])
            unobserved_count = len(nodes_at_level) - observed_count
            
            total_observed += observed_count
            total_unobserved += unobserved_count
            
            print(f"Assembly {assembly_idx}: {len(nodes_at_level)} total | {observed_count} observed | {unobserved_count} unobserved")
        
        print(f"\nTOTAL: {total_observed + total_unobserved} nodes | {total_observed} observed | {total_unobserved} unobserved")
        print(f"Observed percentage: {100 * total_observed / (total_observed + total_unobserved):.1f}%")
    
    return pos, node_assemblies


def add_default_node_colors(graph: nx.Graph, default_color: str = 'gray') -> None:
    """
    Add default color attribute to all nodes in the graph.
    
    Args:
        graph: NetworkX graph to modify
        default_color: Default color to assign to nodes
    """
    for node in graph.nodes():
        if 'color' not in graph.nodes[node]:
            graph.nodes[node]['color'] = default_color


def draw_rounded_rectangles(ax, positions, colors, sizes, edge_width, edge_color, alpha=0.8):
    """
    Draw rounded rectangles instead of circles for overlapping nodes.
    Creates horizontal rectangles that act as masks for overlapping circles.
    
    Args:
        ax: Matplotlib axes
        positions: Node positions
        colors: Node colors
        sizes: Node sizes
        edge_width: Border width
        edge_color: Border color
        alpha: Transparency
    """
    import matplotlib.patches as patches
    
    for i, ((x, y), color, size) in enumerate(zip(positions, colors, sizes)):
        # Convert circle size to rectangle dimensions
        # Make rectangles horizontal and appropriately sized
        width = size * 0.002  # Horizontal width based on size
        height = size * 0.0008  # Smaller height for horizontal rectangles
        
        # Create rounded rectangle
        rect = patches.FancyBboxPatch(
            (x - width/2, y - height/2), width, height,
            boxstyle="round,pad=0.0002",
            facecolor=color,
            edgecolor=edge_color if edge_width > 0 else None,
            linewidth=edge_width,
            alpha=alpha
        )
        ax.add_patch(rect)


class DataExtractor:
    """
    Helper class for extracting graph data from DataFrame by name.
    Simplifies the workflow of getting pathways and creating graphs.
    """
    
    def __init__(self, df: pd.DataFrame, parent_folder_path: str = "../MolecularAssembly/"):
        """
        Initialize the data extractor.
        
        Args:
            df: DataFrame containing name, all_nodes, and observed_nodes columns
            parent_folder_path: Path to the parent folder for StringToPaths
        """
        self.df = df
        self.parent_folder_path = parent_folder_path
        self._string_to_paths = None
    
    @property
    def string_to_paths(self):
        """Lazy loading of StringToPaths instance."""
        if self._string_to_paths is None:
            try:
                from helpers.data_extractors.pathway_helper import StringToPaths
                self._string_to_paths = StringToPaths(parent_folder_path=self.parent_folder_path)
            except ImportError:
                raise ImportError(
                    "Could not import StringToPaths. Please ensure the pathway_helper module is available."
                )
        return self._string_to_paths
    
    def extract_data_by_name(
        self, 
        name: str, 
        check_file_exists: bool = True
    ) -> tuple[pd.DataFrame, list[str], nx.Graph]:
        """
        Extract pathways DataFrame, observed nodes, and graph for a given name.
        
        Args:
            name: Name to search for in the DataFrame (e.g., 'ripper_ARM_07_07')
            check_file_exists: Whether to check if pathway files exist
        
        Returns:
            Tuple of (df_paths, observed_nodes, graph)
        
        Raises:
            ValueError: If name not found in DataFrame
            IndexError: If data structure is unexpected
        """
        # Find the row with the specified name
        matching_rows = self.df.loc[self.df.name == name]
        
        if matching_rows.empty:
            available_names = self.df.name.unique()[:10]  # Show first 10 names
            raise ValueError(
                f"Name '{name}' not found in DataFrame. "
                f"Available names include: {list(available_names)}"
            )
        
        if len(matching_rows) > 1:
            print(f"Warning: Multiple rows found for name '{name}'. Using the first one.")
        
        row = matching_rows.iloc[0]
        
        # Extract sequences (all_nodes)
        try:
            sequences = list(row.all_nodes[0])  # Assuming all_nodes is a list of lists
        except (IndexError, TypeError) as e:
            raise IndexError(
                f"Could not extract sequences from all_nodes for '{name}'. "
                f"Expected structure: list of lists. Got: {type(row.all_nodes)}"
            ) from e
        
        # Extract observed nodes
        try:
            observed_nodes = list(row.observed_nodes[0])  # Assuming observed_nodes is a list of lists
        except (IndexError, TypeError) as e:
            raise IndexError(
                f"Could not extract observed_nodes for '{name}'. "
                f"Expected structure: list of lists. Got: {type(row.observed_nodes)}"
            ) from e
        
        # Get pathways DataFrame
        df_paths = self.string_to_paths.get_pathways_dataframe(
            sequences=sequences,
            check_of_file_exists=check_file_exists
        )
        
        # Create graph
        graph = self.string_to_paths.create_graph_from_pathways(df_paths)
        
        return df_paths, observed_nodes, graph
    
    def get_available_names(self) -> list[str]:
        """Get list of available names in the DataFrame."""
        return self.df.name.unique().tolist()
    
    def get_name_info(self, name: str) -> dict[str, any]:
        """
        Get basic information about a name entry.
        
        Args:
            name: Name to get info for
        
        Returns:
            Dictionary with info about the entry
        """
        matching_rows = self.df.loc[self.df.name == name]
        
        if matching_rows.empty:
            return {"found": False, "error": f"Name '{name}' not found"}
        
        row = matching_rows.iloc[0]
        
        try:
            num_sequences = len(row.all_nodes[0]) if row.all_nodes else 0
            num_observed = len(row.observed_nodes[0]) if row.observed_nodes else 0
        except (IndexError, TypeError):
            num_sequences = "Unknown (data structure issue)"
            num_observed = "Unknown (data structure issue)"
        
        return {
            "found": True,
            "name": name,
            "num_sequences": num_sequences,
            "num_observed_nodes": num_observed,
            "has_multiple_entries": len(matching_rows) > 1
        }


def extract_and_visualize(
    df: pd.DataFrame,
    name: str,
    parent_folder_path: str = "../MolecularAssembly/",
    observed_color: str = 'lightcoral',
    unobserved_color: str = 'gray',
    node_size: float | str = 'auto',
    node_edge_width: float = 0.5,
    node_edge_color: str = 'black',
    node_shape: str = 'circle',
    figsize: tuple[float, float] = (4, 6),
    vertical_scale: float = 1.0,
    force_ylim: tuple[float, float] | None = (0, 7),
    show_statistics: bool = True
) -> tuple[dict[str, tuple[float, float]], dict[str, int], pd.DataFrame, list[str], nx.Graph]:
    """
    One-step function to extract data and create visualization.
    
    Args:
        df: DataFrame containing name, all_nodes, and observed_nodes columns
        name: Name to search for (e.g., 'ripper_ARM_07_07')
        parent_folder_path: Path to the parent folder for StringToPaths
        observed_color: Color for observed nodes
        unobserved_color: Color for unobserved nodes
        node_size: Node sizing method or fixed size
        node_edge_width: Thickness of node border lines
        node_edge_color: Color of node border lines
        node_shape: Shape of nodes ('circle' or 'rectangle')
        figsize: Figure size
        vertical_scale: Vertical distance between assembly levels
        force_ylim: Force Y-axis limits for consistency
        show_statistics: Whether to print layer statistics
    
    Returns:
        Tuple of (positions, node_assemblies, df_paths, observed_nodes, graph)
    
    Examples:
        # Basic usage
        pos, mapping, df_paths, obs_nodes, graph = extract_and_visualize(df, 'ripper_ARM_07_07')
        
        # Custom styling with rectangles
        observed_color = 'darkred'
        unobserved_color = 'lightgray'
        pos, mapping, df_paths, obs_nodes, graph = extract_and_visualize(
            df, 'ripper_ARM_07_07',
            observed_color=observed_color,
            unobserved_color=unobserved_color,
            node_size=80,
            node_edge_width=0,  # No borders
            node_shape='rectangle',  # Use rectangles
            figsize=(6, 8)
        )
        # Note: Unobserved nodes will appear on the left, observed on the right
    """
    # Extract data
    extractor = DataExtractor(df, parent_folder_path)
    df_paths, observed_nodes, graph = extractor.extract_data_by_name(name)
    
    # Create visualization
    positions, node_assemblies = visualize_vertical_assembly_graph(
        graph=graph,
        df=df_paths,
        observed_nodes=observed_nodes,
        observed_color=observed_color,
        unobserved_color=unobserved_color,
        node_size=node_size,
        figsize=figsize,
        vertical_scale=vertical_scale,
        force_ylim=force_ylim,
        show_statistics=show_statistics,
        node_edge_width=node_edge_width,
        node_shape=node_shape
    )
    
    return positions, node_assemblies, df_paths, observed_nodes, graph