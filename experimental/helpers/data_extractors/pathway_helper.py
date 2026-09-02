from __future__ import annotations
import subprocess
import signal
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from ..assembly_path_parser import string_pathway as sp
from .memory import Hash


@dataclass(frozen=True)
class AssemblyConfig:
    """Configuration for AssemblyGo execution."""

    assembly_go_path: str = "assemblygo/bin/assembly.exe"
    parent_folder_path: str = "./MolecularAssembly/"
    timeout: float = 120.0
    workers: int = 64
    verbose: bool = False
    confidence: float | None = None


@dataclass(frozen=True)
class AssemblyPathway:
    """Represents an assembly pathway for a sequence."""
    sequence: str
    assembly_index: float
    node_list: np.ndarray
    edge_list: np.ndarray

    @classmethod
    def empty(cls, sequence: str) -> AssemblyPathway:
        """Create empty pathway for sequence with no assembly."""
        return cls(
            sequence=sequence,
            assembly_index=0,
            node_list=np.array([]),
            edge_list=np.array([]),
        )


@dataclass(frozen=True)
class AssemblyMetrics:
    """Metrics calculated for assembly pathways."""
    name: str
    complexity: int = 0
    diversity: int = 0
    A: float = 0.0
    max_MA: float = 0.0
    observed_nodes: np.ndarray | None = None
    all_nodes: np.ndarray | None = None
    exploration_ratio: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame creation."""
        return {
            "name": self.name,
            "complexity": self.complexity,
            "diversity": self.diversity,
            "A": self.A,
            "max_MA": self.max_MA,
            "observed_nodes": [self.observed_nodes]
            if self.observed_nodes is not None
            else None,
            "all_nodes": [self.all_nodes]
            if self.all_nodes is not None
            else None,
            "exploration_ratio": self.exploration_ratio,
        }

    @classmethod
    def empty(cls, name: str) -> AssemblyMetrics:
        """Create empty metrics."""
        return cls(name=name)


class StringToPaths:
    """Service class to calculate assembly pathways for sequences using AssemblyGo."""
    def __init__(
        self,
        config: AssemblyConfig | None = None,
        sequences: dict[str, list[str]] | None = None,
        parquet_path: str | Path | None = None,
        # Add backward compatibility parameters
        parent_folder_path: str | None = None,
        confidence: float | None = None,
        assembly_go_path: str | None = None,
        timeout: float | None = None,
        workers: int | None = None,
        verbose: bool | None = None,
    ):
        """Initialize the StringToPaths processor.

        Args:
            config: Configuration for AssemblyGo execution
            sequences: Dictionary mapping group names to lists of sequences
            parquet_path: Optional path to a Parquet table (columns: sequence,
                assembly_index, node_list, edge_list) used as a fast lookup
                before falling back to on-disk .txt logs / AssemblyGo.
            parent_folder_path: Path for assembly outputs (backward compatibility)
            confidence: Confidence level (backward compatibility)
            assembly_go_path: Path to AssemblyGo executable (backward compatibility)
            timeout: Timeout for AssemblyGo execution (backward compatibility)
            workers: Number of workers (backward compatibility)
            verbose: Verbose output (backward compatibility)
        """
        # If individual parameters are provided, create config from them
        if any(
            [
                parent_folder_path,
                confidence,
                assembly_go_path,
                timeout,
                workers,
                verbose is not None,
            ]
        ):
            self.config = AssemblyConfig(
                assembly_go_path=assembly_go_path
                or "assemblygo/bin/assembly.exe",
                parent_folder_path=parent_folder_path or "./MolecularAssembly/",
                timeout=timeout or 120.0,
                workers=workers or 64,
                verbose=verbose or False,
                confidence=confidence,
            )
        else:
            # Use provided config or default
            self.config = config or AssemblyConfig()

        self.sequences = sequences or {}
        self._hash = Hash(self.config.confidence)

        # Convert paths to Path objects
        self._assembly_go_path = Path.home() / self.config.assembly_go_path
        self._parent_folder_path = Path(self.config.parent_folder_path)

        # Optional fast-path lookup table, opt-in and backward compatible
        self._parquet_lookup: dict[str, dict] = {}
        if parquet_path is not None and Path(parquet_path).exists():
            self._parquet_lookup = (
                pd.read_parquet(parquet_path)
                .set_index("sequence")
                .to_dict(orient="index")
            )

    @property
    def assembly_go_path(self) -> Path:
        """Get the AssemblyGo executable path."""
        return self._assembly_go_path

    @property
    def parent_folder_path(self) -> Path:
        """Get the parent folder path for outputs."""
        return self._parent_folder_path

    def add_sequences(self, group_name: str, sequences: list[str]) -> None:
        """Add sequences to a group."""
        self.sequences[group_name] = sequences

    def get_all_unique_sequences(self) -> set[str]:
        """Get all unique sequences from the sequences dictionary."""
        return {seq for seq_list in self.sequences.values() for seq in seq_list}

    def get_output_path(self, sequence: str) -> Path:
        """Get the output file path for a sequence."""
        return self.parent_folder_path / f"{sequence}.txt"

    def should_skip_calculation(
        self, sequence: str, skip_if_existing: bool = True
    ) -> bool:
        """Check if calculation should be skipped for a sequence."""
        return skip_if_existing and self.get_output_path(sequence).exists()

    def calculate_assembly_for_individual_sequence(self, sequence: str) -> None:
        """Calculate assembly for a single sequence using AssemblyGo."""
        self.parent_folder_path.mkdir(parents=True, exist_ok=True)

        output_file = self.get_output_path(sequence)

        cmd = [
            str(self.assembly_go_path),
            "-workers",
            str(self.config.workers),
            "-log",
            f"-logfile={output_file}",
            "-string",
            sequence,
        ]

        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL)

        try:
            proc.wait(timeout=self.config.timeout)
        except subprocess.TimeoutExpired:
            proc.send_signal(signal.SIGINT)
            if self.config.verbose:
                print(f"Timed out, killing process for {sequence}")

    def calculate_assembly_for_all_sequences(
        self, skip_if_existing: bool = True
    ) -> StringToPaths:
        """Calculate assembly for all sequences in the sequences dictionary."""
        unique_sequences = self.get_all_unique_sequences()

        for sequence in tqdm(unique_sequences, desc="Calculating assemblies"):
            if self.should_skip_calculation(sequence, skip_if_existing):
                continue
            self.calculate_assembly_for_individual_sequence(sequence)

        return self

    def parse_assembly_output(
        self, sequence: str, check_if_file_exists: bool = False
    ) -> AssemblyPathway:
        """Parse AssemblyGo output for a sequence.

        Looks up the sequence in three tiers: the Parquet lookup table
        (if provided), then an existing .txt log on disk, then finally
        runs AssemblyGo to create the .txt log if neither is available.
        """
        cached = self._parquet_lookup.get(sequence)
        if cached is not None:
            return AssemblyPathway(
                sequence=sequence,
                assembly_index=cached["assembly_index"],
                node_list=np.array(cached["node_list"]),
                edge_list=np.array(cached["edge_list"]),
            )

        file_path = self.get_output_path(sequence)

        # This was added ex post when I started manipulating the networkX
        # graphs. Some nodes might not be observed and not generated in the
        # union, for some reason it happens for single letters. So this is
        # a quick fix.
        if not file_path.exists():
            if check_if_file_exists:
                print(f"Output file does not exist for {sequence}")
                return AssemblyPathway.empty(sequence)
            # Not cached anywhere yet: compute it now (writes the .txt log)
            self.calculate_assembly_for_individual_sequence(sequence)
            if not file_path.exists():
                return AssemblyPathway.empty(sequence)

        construction_object = sp.generate_string_pathway(file_path)
        assembly_index = construction_object.ma

        if assembly_index == 0:
            return AssemblyPathway.empty(sequence)

        # Create rules for node/edge name replacement
        rules = {}
        for i, atom in enumerate(construction_object.atoms_list):
            rules[f"atom{i}"] = atom[1]
        for i, step in enumerate(construction_object.steps_str):
            rules[f"step{i + 1}"] = step

        # Get and transform nodes/edges
        node_list, edge_list = sp.get_graph_string(construction_object)
        node_list, edge_list = np.asarray(node_list), np.asarray(edge_list)

        replace_func = np.vectorize(lambda val: rules.get(val, val))
        transformed_nodes = replace_func(node_list)
        transformed_edges = replace_func(edge_list)

        return AssemblyPathway(
            sequence=sequence,
            assembly_index=assembly_index,
            node_list=transformed_nodes,
            edge_list=transformed_edges,
        )

    def _remove_duplicate_peptides(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate peptides from pathways DataFrame.

        Keeps the first occurrence of each peptide sequence.

        Args:
            df: DataFrame with 'peptide' column that may contain duplicates

        Returns:
            pd.DataFrame: DataFrame with duplicate peptides removed
        """
        if df.empty:
            return df

        initial_count = len(df)
        df_deduplicated = df.drop_duplicates(
            subset=["peptide"], keep="first"
        ).reset_index(drop=True)
        final_count = len(df_deduplicated)

        if initial_count != final_count and self.config.verbose:
            print(
                f"Removed {initial_count - final_count} duplicate peptides. "
                f"Kept {final_count} unique peptides."
            )

        return df_deduplicated

    def get_pathways_dataframe(
        self, sequences: list[str], check_of_file_exists: bool = False
    ) -> pd.DataFrame:
        """Get assembly pathways as a DataFrame."""
        pathways = []

        for sequence in sequences:
            pathway = self.parse_assembly_output(
                sequence, check_if_file_exists=check_of_file_exists
            )
            pathways.append(
                {
                    "peptide": pathway.sequence,
                    "assembly_index": pathway.assembly_index,
                    "node_list": pathway.node_list,
                    "edge_list": pathway.edge_list,
                }
            )

        df = pd.DataFrame(pathways)
        return self._remove_duplicate_peptides(df)

    @staticmethod
    def create_graph_from_pathways(df: pd.DataFrame) -> nx.DiGraph | None:
        """Create a NetworkX graph from pathway DataFrame."""
        if df.empty:
            return None

        graphs = []
        for _, pathway in df.iterrows():
            graph = nx.DiGraph()
            for edge in pathway["edge_list"]:
                if len(edge) >= 2:
                    graph.add_edge(str(edge[0]), str(edge[1]))
            graphs.append(graph)

        return nx.compose_all(graphs) if graphs else None

    def calculate_metrics_for_sequences(
        self, sequences: list[str], name: str = "calculated"
    ) -> AssemblyMetrics:
        """Calculate metrics for a list of sequences."""
        df_paths = self.get_pathways_dataframe(sequences)
        graph = self.create_graph_from_pathways(df_paths)

        if graph is None or df_paths.empty:
            return AssemblyMetrics.empty(name)

        observed_nodes = df_paths["peptide"].unique()
        all_nodes = np.unique(
            np.concatenate([nodes for nodes in df_paths["node_list"]])
        )

        return AssemblyMetrics(
            name=name,
            complexity=graph.number_of_edges(),
            diversity=len(df_paths),
            A=np.mean(df_paths["assembly_index"].apply(np.exp)),
            max_MA=df_paths["assembly_index"].max(),
            observed_nodes=observed_nodes,
            all_nodes=all_nodes,
            exploration_ratio=len(observed_nodes) / len(all_nodes)
            if len(all_nodes) > 0
            else 0,
        )

    def calculate_metrics_with_caching(
        self,
        key: str,
        force_recalc: bool = False,
        confidence: float | None = None,
    ) -> AssemblyMetrics:
        """Calculate metrics with caching support."""
        # Use provided confidence or fall back to config confidence
        effective_confidence = (
            confidence if confidence is not None else self.config.confidence
        )

        hash_key = self._hash.generate_hash_key(
            key=key, confidence=effective_confidence
        )

        if not force_recalc and self._hash.check_if_analysis_run_before(
            hash_key
        ):
            cached_df = self._hash.load_pickled_analysis(hash_key)
            # Convert DataFrame back to AssemblyMetrics
            row = cached_df.iloc[0]
            return AssemblyMetrics(
                name=row["name"],
                complexity=row["complexity"],
                diversity=row["diversity"],
                A=row["A"],
                max_MA=row["max_MA"],
                observed_nodes=row["observed_nodes"][0]
                if row["observed_nodes"] is not None
                else None,
                all_nodes=row["all_nodes"][0]
                if row["all_nodes"] is not None
                else None,
                exploration_ratio=row["exploration_ratio"],
            )

        metrics = self.calculate_metrics_for_sequences(
            self.sequences[key], name=key
        )

        # Cache the result
        df = pd.DataFrame([metrics.to_dict()], index=[0])
        self._hash.pickle_current_analysis(df, hash_key)

        return metrics

    def get_metrics_for_all(
        self, force_recalc: bool = False, confidence: float | None = None
    ) -> pd.DataFrame:
        """Calculate metrics for all sequence groups."""
        all_metrics = []

        for key in tqdm(self.sequences.keys(), desc="Calculating metrics"):
            metrics = self.calculate_metrics_with_caching(
                key, force_recalc, confidence
            )
            all_metrics.append(metrics.to_dict())

        return pd.DataFrame(all_metrics)

    def get_metrics_for_trajectory(
        self, trajectory_names: list[str]
    ) -> pd.DataFrame:
        """Calculate metrics for a specific trajectory."""
        all_metrics = []

        for name in tqdm(trajectory_names, desc="Processing trajectory"):
            if name in self.sequences:
                metrics = self.calculate_metrics_for_sequences(
                    self.sequences[name], name=name
                )
                all_metrics.append(metrics.to_dict())

        return pd.DataFrame(all_metrics)


# Factory functions for easy creation
def create_string_to_paths(
    assembly_go_path: str = "assemblygo/bin/assembly.exe",
    parent_folder: str = "./Calculated_assemblies/",
    timeout: float = 120.0,
    sequences: dict[str, list[str]] | None = None,
    confidence: float | None = None,
    verbose: bool = False,
) -> StringToPaths:
    """Factory function to create StringToPaths with configuration."""
    config = AssemblyConfig(
        assembly_go_path=assembly_go_path,
        parent_folder_path=parent_folder,
        timeout=timeout,
        verbose=verbose,
        confidence=confidence,
    )

    return StringToPaths(config=config, sequences=sequences)


def create_assembly_config(**kwargs) -> AssemblyConfig:
    """Factory function to create AssemblyConfig with custom parameters."""
    return AssemblyConfig(**kwargs)
