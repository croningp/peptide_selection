from typing import Any, Optional
from .data_reader import OligossReader, DataFilter  # noqa
from .pathway_helper import StringToPaths  # noqa
import pandas as pd  # noqa
from pathlib import Path
from .memory import Hash  # noqa
import matplotlib.pyplot as plt
import json
import plotly.express as px
from plotly import graph_objects as go
import numpy as np
import functools
import operator

# Type aliases for better readability
type ExperimentData = dict[str, pd.DataFrame]
type SequenceDict = dict[int, str]
type SequenceList = dict[str, list[str]]
type LabBookDict = dict[str, dict[str, Any]]


class AllExperiments:
    def __init__(
        self,
        confidence: float = 70.0,
        folder_path_for_assembly: str = "../MolecularAssembly",
        cache: bool = False,
        labbook: Optional[pd.DataFrame] = None,
    ) -> None:
        """Helper class to load all experiments and calculate the assembly

        Args:
            confidence: Cutoff for the confidence scoring from the OLIGOSS output.
                This roughly corresponds to percentage of fragments that has to be
                identified among all possible fragments for the sequence assuming
                b/y fragmentation. Default set to 70.0.
            folder_path_for_assembly: Name of the folder in which the outputs of
                the assembly calculations are stored. Defaults "MolecularAssembly".
            cache: Whether to use caching for calculations.
            labbook: Optional DataFrame containing experiment information.
        """
        self.experiments: ExperimentData = {}
        self.confidence: float = confidence
        self.folder_path_for_assembly: str = folder_path_for_assembly
        self._full_analysis: Optional[pd.DataFrame] = None
        self.cache: bool = cache
        if self.cache:
            Hash().create_cache_folder()
        self.labbook: Optional[pd.DataFrame] = labbook
        self.read_parents: list[str] = []

    @staticmethod
    def convert_to_dict(s: str) -> dict[int, str]:
        """Convert a string representation of a dictionary to an actual dictionary."""
        return {int(k): v for k, v in json.loads(s.replace("'", '"')).items()}

    def read_labbook(
        self,
        labbook_path: str | Path = "./labbook.csv",
        verbose: bool = False,
    ) -> pd.DataFrame:
        """Reads the labbook from the given path. The labbook is a CSV file
        with the following columns: experiment_id, sequences, description,
        monomer_type, experiment_type, key_sequences. The sequences are
        stored as a string with the following format: "seq1; seq2; seq3".
        The key_sequences are stored as a string with the following format:
        "{0: 'seq1', 1: 'seq2', 2: 'seq3'}". The function returns a
        DataFrame with the following columns: experiment_id, sequences,
        description, monomer_type, experiment_type, key_sequences.

        Args:
            labbook_path: Path to the labbook CSV file
            verbose: Whether to print detailed loading information

        Returns:
            DataFrame with processed labbook data
        """
        labbook_path = Path(labbook_path)
        if not labbook_path.is_file():
            raise FileNotFoundError(f"Labbook file not found: {labbook_path}")

        labbook = pd.read_csv(labbook_path)
        labbook["sequences"] = labbook["sequences"].apply(
            lambda x: self.convert_to_dict(x)
        )

        if verbose:
            self._analyze_labbook(labbook)

        return labbook

    def _analyze_labbook(
        self,
        labbook: pd.DataFrame,
    ) -> None:
        """Provides all experiment IDs with descriptions in compact format.

        Args:
            labbook: The loaded labbook DataFrame
        """
        for idx, row in labbook.iterrows():
            exp_id = row["experiment_id"]
            desc = ""
            if "description" in row and pd.notna(row["description"]):
                desc = f" | {str(row['description'])}"
            print(f"  {exp_id}{desc}")

    def add_experiment(self, data: ExperimentData) -> None:
        """Append a new experiment to the dictionary of all experiments."""
        self.experiments.update(data)

    def add_methyl_ester_experiment(self, data: ExperimentData) -> None:
        """Special method to load experiments from Papain
        catalysed oligomerisation of amino acid methyl esters.
        One has to remove sequences that have none or more
        than one methyl ester group <m>. Also the m has to be
        at the end of the sequence."""
        for key in data:
            valid_df = self.return_valid_methyl_ester_peptides(data, key)
            valid_df = self.drop_m_at_end(valid_df)
            self.add_experiment({key: valid_df})

    def smart_read(self, names: list[str]) -> None:
        """This checks whether the list of experiments is empty are
        experiment ids or parents. And read them all, avoiding
        unnecessary reading of the same data."""
        if self.labbook is None:
            raise ValueError(
                "Labbook is not initialized. Please provide a labbook."
            )

        for name in names:
            if (prnt := self.what_is_parent_name(name)) in self.read_parents:
                continue
            # determine type of the name
            match self.labbook.loc[
                self.labbook.parent == prnt
            ].monomer_type.iloc[0]:
                case "methyl ester":
                    self.add_methyl_ester_experiment(
                        OligossReader.read_from_labbook(
                            prnt, self.labbook
                        ).return_all_data()
                    )
                case "amino acid":
                    self.add_experiment(
                        OligossReader.read_from_labbook(
                            prnt, self.labbook
                        ).return_all_data()
                    )
                case _:
                    raise ValueError(
                        f"Unknown experiment type for {name}. "
                        "Please check the labbook."
                    )
            self.read_parents.append(prnt)

    def what_is_parent_name(self, name: str) -> str:
        """This checks whether the name is parent or experiment id.
        If it's experiment id, it returns the parent name."""
        if self.labbook is None:
            raise ValueError(
                "Labbook is not initialized. Please provide a labbook."
            )

        if not (_prnt := self.labbook.loc[self.labbook.experiment_id == name]).empty:  # fmt: skip
            return _prnt.iloc[0].parent
        elif not (_prnt := self.labbook.loc[self.labbook.parent == name]).empty:  # fmt: skip
            return _prnt.iloc[0].parent
        else:
            raise ValueError(f"Experiment {name} not found in labbook")

    def return_sequences(
        self, experiment: list[str] | None = None
    ) -> dict[str, list[str]]:
        """Return all sequences for the given experiments.
        Returns rich format of pd.DataFrame with sequences and confidence"""
        return DataFilter(
            data=self.experiments, exp_names=experiment
        ).filter_by_confidence(self.confidence)

    def return_full_analysis(
        self,
        experiment: str | list[str] | list[list[str]] | None = None,
        force_recalc: bool = False,
    ) -> pd.DataFrame:
        """Calculates the assembly for all sequences and returns a
        DataFrame with all metrics. The metrics are ensable assembyly,
        number of edges in JAS, diversity and exploration ratio.
        It uses confidence accrding to self.confidence."""

        experiment_list: list[str] | None = self.unify_type(experiment)
        sequences: dict[str, list[str]] = self.return_sequences(experiment_list)
        self._full_analysis = (
            StringToPaths(
                parent_folder_path=self.folder_path_for_assembly,
                sequences=sequences,
                confidence=self.confidence,
            )
            .calculate_assembly_for_all_sequences()
            .get_metrics_for_all(
                force_recalc=force_recalc, confidence=self.confidence
            )
        )
        return self._full_analysis

    def unify_type(self, experiment: Any) -> list[str] | None:
        """Helper function unify the type of the list of experiment names."""
        if isinstance(experiment, str):
            return [experiment]
        if isinstance(experiment, list):
            if isinstance(experiment[0], list):
                return self.flatten(experiment)
            else:
                return experiment
        return None
        # TODO use proper typing instead of Any

    def ends_with_one_methyl_ester_group(self, peptide: str) -> bool:
        """Return True if the peptide ends with one methyl ester group
        and if there is exaclty one methyl ester group. This is the only
        valid peptide sequence with methyl ester on C terminus."""
        return (peptide.count("m") == 1) and (peptide[-1] == "m")

    def _assert_full_analysis_was_calculated(self) -> bool:
        """Check if the full analysis has been calculated."""

        if self._full_analysis is None:
            raise ValueError(
                "Full analysis is not calculated. Please run "
                "return_full_analysis first."
            )
        return True

    @staticmethod
    def flatten(list_of_lists: list[list[str]]) -> list[str]:
        """Flatten a list of lists into a single list."""

        return functools.reduce(operator.iconcat, list_of_lists, [])

    def single_trajectory(
        self,
        seq: dict[int, str],
    ) -> pd.DataFrame:
        """Reads internally stored analysis and
        return a DataFrame with the metrics for a single trajectory."""

        self._assert_full_analysis_was_calculated()

        _df = pd.DataFrame(
            index=range(len(seq)), columns=self._full_analysis.columns
        )

        for k, v in seq.items():
            lookup: pd.DataFrame = self._full_analysis.loc[
                self._full_analysis.name == seq[k]
            ]
            assert len(lookup) == 1, (
                f"Error: {k, v}. Maybe your forgot to run analysis first?"
            )
            if len(lookup) != 1:
                print(k)
            _df.iloc[int(k)] = lookup.iloc[0]
        return _df

    def get_list_of_experiment_names_from_ids(
        self,
        experiment_ids: list[str],
    ) -> list[str]:
        """Get list of experiment names from experiment IDs."""
        return self.flatten(
            [
                list(
                    self.labbook.loc[
                        self.labbook.experiment_id == exp_id, "sequences"
                    ]
                    .values[0]
                    .values()
                )
                for exp_id in experiment_ids
            ]
        )

    def filter_analysed_df(self, experiment_names: list[str]) -> pd.DataFrame:
        """experiment_names is a list of single measurement experiment names like 'ripper_MKJ_56_B_06'

        NONO I assume big EXP name like MKJ_56


        """
        self._assert_full_analysis_was_calculated()

        # first do I provide list of experiment super names of indivudal measurments?

        _exp_names = self.get_list_of_experiment_names_from_ids(
            experiment_names
        )

        return self._full_analysis[self._name_present_in_df(_exp_names)]

    def _name_present_in_df(self, _exp_names: list[str]) -> pd.Series:
        return self._full_analysis.name.isin(_exp_names)





    def return_valid_methyl_ester_peptides(
        self,
        _raw_data: dict[str, pd.DataFrame],
        _key: str,
    ) -> pd.DataFrame:
        """Filters data for valid peptides with one methyl ester group."""
        return (
            _raw_data[_key]
            .where(
                _raw_data[_key].sequence.apply(
                    self.ends_with_one_methyl_ester_group
                )
            )
            .dropna()
        )

    def drop_m_at_end(self, df: pd.DataFrame) -> pd.DataFrame:
        """Methyl ester pepides have at the end of sequences
        m, this method removes it."""
        df.sequence = df.sequence.apply(lambda x: x[:-1])
        return df

    @staticmethod
    def what_is_in_dataset(data: pd.DataFrame) -> None:
        """Prints the number of peptides in the dataset and the longest and
        shortest peptide. Just helper showing the basic statistics."""
        ordered_peptides: pd.DataFrame = data.sort_values(
            by="sequence", key=lambda x: x.str.len()
        )
        longest_peptide: str = ordered_peptides.sequence.iloc[-1]
        shortest_peptide: str = ordered_peptides.sequence.iloc[0]
        print(
            f"There is {len(data)} peptides in the dataset."
            f"\nThe longest peptide is {longest_peptide}."
            f"\nThe shortest peptide is {shortest_peptide}."
        )

    @staticmethod
    def vial_scheme(first: str, last: str, n: int) -> list[str]:
        """Return a list of vial names from first to last with n vials."""
        lst: list[str] = [f"{i}{x}" for i in "ABCDEFG" for x in range(1, 10)]
        indices_to_remove = set(range(17, len(lst), 18))
        lst = [x for i, x in enumerate(lst) if i not in indices_to_remove]
        assert len(lst) == 60
        return lst[lst.index(first) : lst.index(last) + 1 : n]

    def generate_color_palette(self, num_colors):
        """Generate a color palette with the specified number of colors."""
        cmap = plt.get_cmap("tab10")
        return [cmap(i) for i in range(num_colors)]

    def combine_trajectories(
        self,
        experiments: list[str],
        columns: list[str],
        crop_max_cycle: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """This assumes that all experiments are of the
        same type and should be avaraged per cycle.
        Primarely, I will care about exploration ratio,
        as it is bit less dependent on the diversity
        and general conversion.

        experiments: list of experiment names (of the whole trajectory)

        """

        dataframes: list[pd.DataFrame] = [
            self.single_trajectory(self.read_sequences(exp_name))
            for exp_name in experiments
        ]

        means = pd.DataFrame(index=dataframes[0].index)
        stds = pd.DataFrame(index=dataframes[0].index)

        # Calculate mean and std for each column
        for col in columns:
            # Stack values for this column from all dataframes
            stacked_values = pd.concat([df[col] for df in dataframes], axis=1)
            means[col] = stacked_values.mean(axis=1)
            stds[col] = stacked_values.std(axis=1)

        # Crop the dataframes if crop_max_cycle is provided
        if crop_max_cycle is not None:
            means = means[means.index < crop_max_cycle]
            stds = stds[stds.index < crop_max_cycle]

        return means.to_numpy().flatten(), stds.to_numpy().flatten()

    @staticmethod
    def find_type(key: str, labbook: dict[str, dict[str, Any]]) -> str:
        """Simple helper function to find the monomer type"""
        for i in labbook.values():
            if key in list(i["key_sequences"].values()):
                try:
                    mtype = i["experiment_type"]
                    return mtype
                except KeyError:
                    return "unknown"
        return "unknown"

    def read_sequences(self, name: str) -> SequenceDict:
        """Reads the sequences from the labbook. This is a helper function
        that returns the sequences for the given experiment (trajectory).

        Expected return format:
        {0: 'ripper_MKJ_56_B_01',
         1: 'ripper_MKJ_56_B_10',
         2: 'ripper_MKJ_56_B_19',
         ...}
        """
        if self.labbook is None:
            raise ValueError(
                "Labbook is not initialized. Please provide a labbook."
            )

        return self.labbook.loc[
            self.labbook["experiment_id"] == name, "sequences"
        ].values[0]

    def merge_dataframes(
        self, df1: pd.DataFrame, df2: pd.DataFrame
    ) -> pd.DataFrame:
        # unite two dataframes by specific column and if
        # conflicting value in oonether column choose higher value

        # I hardcode the names of the columns. Not ideal, but it works.
        # I expect columns: sequence | confidence | composition

        _temporal_merge: pd.DataFrame = pd.merge(
            df1,
            df2,
            how="outer",
            on=["sequence"],
            suffixes=("_df1", "_df2"),
        )

        # drop duplicates in column sequence
        _temporal_merge.drop_duplicates(subset=["sequence"], inplace=True)

        # Create a new DataFrame with just the sequence column
        result: pd.DataFrame = _temporal_merge[
            ["sequence", "composition_df1"]
        ].copy()
        result.rename(columns={"composition_df1": "composition"}, inplace=True)

        result = pd.concat(
            [
                result,
                _temporal_merge[["confidence_df1", "confidence_df2"]]
                .max(axis=1, skipna=True)
                .rename("confidence"),
            ],
            axis=1,
        )
        return result

    def plot_trajectories_plotly(
        self,
        list_of_experiments: list[str],
        x_axis: str = "diversity",
        y_axis: str = "exploration_ratio",
        title: str = "default",
        xaxis_title: str = "default",
        yaxis_title: str = "default",
        font_size: int = 14,
        fig_size: tuple[int, int] = (800, 600),
        show_peptides_in_plot: bool = False,
        y_range: Optional[list[float]] = None,
        marker_size: int = 8,
    ) -> go.Figure:
        """Plotting tool for quick inspection. Not nice palettes for publication."""

        fig = px.line()
        for name in list_of_experiments:
            # check if not cumulative
            # if ends with _cummul, remove it
            if "_cummul" in name:
                name = name.split("_cummul")[0]
                old_name: SequenceDict = self.read_sequences(name)
                # add _cummul everywhere
                sequences_cummul: SequenceDict = {
                    k: f"{v}_cummul" for k, v in old_name.items()
                }
                df_local = self.single_trajectory(sequences_cummul)
            else:
                df_local = self.single_trajectory(self.read_sequences(name))

            def format_sequences() -> list[str]:
                """Formatting the string to be printed in the plot"""
                new_list = []
                for peptide_list in df_local.observed_nodes:
                    new_nice_string = ""
                    line_count = 0
                    for peptide in peptide_list:
                        if line_count > 60:
                            new_nice_string += "<br>"
                            line_count = 0
                        new_nice_string += f"{peptide}, "
                        line_count += len(peptide) + 2
                    new_list.append(new_nice_string[:1900])
                return new_list

            list_to_print = (
                format_sequences() if show_peptides_in_plot else None
            )

            if x_axis == "cycle":
                fig.add_trace(
                    go.Scatter(
                        x=df_local.index,
                        y=df_local[y_axis],
                        name=name,
                        mode="lines+markers",
                        text=list_to_print,
                        marker=dict(
                            size=marker_size,
                            line=dict(width=1),
                        ),
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=df_local[x_axis],
                        y=df_local[y_axis],
                        name=name,
                        mode="lines+markers",
                        text=list_to_print,
                        marker=dict(
                            size=marker_size,
                            line=dict(width=1),
                        ),
                    )
                )

            if xaxis_title == "default":
                xaxis_title = x_axis
            if yaxis_title == "default":
                yaxis_title = y_axis

            fig.update_layout(
                width=fig_size[0],
                height=fig_size[1],
                font_family="Arial",
                font_color="black",
                xaxis_title=xaxis_title,
                yaxis_title=yaxis_title,
                font_size=font_size,
                title=title,
                yaxis_range=y_range,
            )
        return fig
