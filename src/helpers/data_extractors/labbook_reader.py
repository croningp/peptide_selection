import pandas as pd
import json
from typing import Any


class LabBookHelper:
    """Helper class for working with labbook experiments and sequence data."""

    def __init__(self, labbook_df: pd.DataFrame):
        """
        Initialize with a labbook DataFrame.

        Args:
            labbook_df: DataFrame containing experiment data with columns:
                       experiment_id, sequences, etc.
        """
        self.labbook_df = labbook_df

    def get_experiment_sequences(self, experiment_id: str) -> dict[str, str]:
        """
        Get the sequences dictionary for a given experiment ID.

        Args:
            experiment_id: The experiment ID (e.g., 'MKJ_56_4')

        Returns:
            Dictionary mapping sequence indices to sequence names

        Example:
            >>> helper.get_experiment_sequences('MKJ_56_4')
            {"0": "ripper_MKJ_56_B_04", "1": "ripper_MKJ_56_B_13", ...}
        """
        # Find the experiment row
        experiment_row = self.labbook_df[
            self.labbook_df["experiment_id"] == experiment_id
        ]

        if experiment_row.empty:
            raise ValueError(
                f"Experiment ID '{experiment_id}' not found in labbook"
            )

        # Get the sequences field
        sequences_data = experiment_row["sequences"].iloc[0]

        if pd.isna(sequences_data):
            return {}

        # Handle both dict (auto-parsed by pandas) and string (JSON) formats
        if isinstance(sequences_data, dict):
            return sequences_data
        elif isinstance(sequences_data, str):
            try:
                return json.loads(sequences_data)
            except json.JSONDecodeError as e:
                # If it's not valid JSON, maybe it's a simple comma-separated list
                # or some other format - handle gracefully
                raise ValueError(
                    f"Could not parse sequences field for {experiment_id}. {e}. "
                    f"Expected JSON dict but got: {sequences_data[:100]}..."
                )
        else:
            raise ValueError(
                f"Unexpected sequences data type for {experiment_id}: "
                f"{type(sequences_data)} - {sequences_data}"
            )

    def get_experiment_sequence_names(self, experiment_id: str) -> list[str]:
        """
        Get just the sequence names (values) for a given experiment ID.

        Args:
            experiment_id: The experiment ID (e.g., 'MKJ_56_4')

        Returns:
            List of sequence names

        Example:
            >>> helper.get_experiment_sequence_names('MKJ_56_4')
            ['ripper_MKJ_56_B_04', 'ripper_MKJ_56_B_13', ...]
        """
        sequences_dict = self.get_experiment_sequences(experiment_id)
        return list(sequences_dict.values())

    def filter_by_sequence_names(
        self,
        data_df: pd.DataFrame,
        experiment_id: str,
        name_column: str = "name",
    ) -> pd.DataFrame:
        """
        Filter a DataFrame to only include rows matching the experiment's sequence names.

        Args:
            data_df: DataFrame to filter (e.g., with columns 'name', 'complexity', etc.)
            experiment_id: The experiment ID to get sequence names for
            name_column: Column name in data_df that contains sequence names

        Returns:
            Filtered DataFrame containing only rows matching the experiment's sequences

        Example:
            >>> filtered_df = helper.filter_by_sequence_names(all_data, 'MKJ_56_4')
        """
        sequence_names = self.get_experiment_sequence_names(experiment_id)
        return data_df[data_df[name_column].isin(sequence_names)]

    def create_experiment_lookup(
        self,
        data_df: pd.DataFrame,
        experiment_ids: list[str] | None = None,
        name_column: str = "name",
        value_column: str = "observed_nodes",
    ) -> dict[str, any]:
        """
        Create a lookup dictionary mapping experiment IDs to their aggregated data.

        Args:
            data_df: DataFrame containing the data to lookup
            experiment_ids: List of experiment IDs to include. If None, uses all experiments
            name_column: Column name in data_df that contains sequence names
            value_column: Column name in data_df to extract values from

        Returns:
            Dictionary mapping experiment IDs to their data values

        Example:
            >>> lookup = helper.create_experiment_lookup(data_df, ['MKJ_56_4', 'MKJ_56_5'])
            >>> # Returns: {'MKJ_56_4': array([...]), 'MKJ_56_5': array([...])}
        """
        if experiment_ids is None:
            experiment_ids = self.list_experiments()

        lookup_dict = {}

        for exp_id in experiment_ids:
            try:
                # Get the sequence names for this experiment
                sequence_names = self.get_experiment_sequence_names(exp_id)

                # Filter data to only include rows from this experiment
                exp_data = data_df[data_df[name_column].isin(sequence_names)]

                if not exp_data.empty and value_column in exp_data.columns:
                    # If there's only one row, return the value directly
                    if len(exp_data) == 1:
                        lookup_dict[exp_id] = exp_data[value_column].iloc[0]
                    else:
                        # If multiple rows, you might want to aggregate or return all
                        # For now, let's return all values as a list
                        lookup_dict[exp_id] = exp_data[value_column].tolist()
                else:
                    # No data found for this experiment
                    lookup_dict[exp_id] = None

            except ValueError as e:
                # Experiment not found in labbook or other error
                print(f"Warning: Could not process experiment {exp_id}: {e}")
                lookup_dict[exp_id] = None

        return lookup_dict

    def get_experiment_info(self, experiment_id: str) -> pd.Series:
        """
        Get all information for a given experiment ID.

        Args:
            experiment_id: The experiment ID

        Returns:
            Series containing all experiment information
        """
        experiment_row = self.labbook_df[
            self.labbook_df["experiment_id"] == experiment_id
        ]

        if experiment_row.empty:
            raise ValueError(
                f"Experiment ID '{experiment_id}' not found in labbook"
            )

        return experiment_row.iloc[0]

    def list_experiments(self, pattern: str | None = None) -> list[str]:
        """
        List all experiment IDs, optionally filtered by pattern.

        Args:
            pattern: Optional string pattern to filter experiment IDs

        Returns:
            List of experiment IDs

        Example:
            >>> helper.list_experiments('MKJ_56')
            ['MKJ_56_1', 'MKJ_56_2', 'MKJ_56_3', 'MKJ_56_4', ...]
        """
        experiment_ids = self.labbook_df["experiment_id"].tolist()

        if pattern:
            experiment_ids = [
                exp_id for exp_id in experiment_ids if pattern in exp_id
            ]

        return experiment_ids

    def get_experiment_data_dict(
        self,
        data_df: pd.DataFrame,
        experiment_id: str,
        name_column: str = "name",
        value_column: str = "observed_nodes",
    ) -> dict[str, Any]:
        """
        Get a dictionary mapping sequence names to their data values for a specific experiment,
        ordered by the sequence indices from the experiment.

        Args:
            data_df: DataFrame containing the data
            experiment_id: The experiment ID to get sequences for
            name_column: Column name in data_df that contains sequence names
            value_column: Column name in data_df to extract values from

        Returns:
            Dictionary mapping sequence names to their data values, ordered by sequence indices

        Example:
            >>> data_dict = helper.get_experiment_data_dict(df, 'ARM_07_2', value_column='observed_nodes')
            >>> # Returns: {'ripper_ARM_07_01': array([...]), 'ripper_ARM_07_02': array([...]), ...}
        """
        # Get all sequences for the experiment (this returns {0: 'name1', 1: 'name2', ...})
        experiment_sequences = self.get_experiment_sequences(experiment_id)

        # Filter DataFrame to only include rows matching the experiment sequences
        filtered_df = data_df[
            data_df[name_column].isin(experiment_sequences.values())
        ]

        # Create a mapping from sequence name to data value
        name_to_value = filtered_df.set_index(name_column)[
            value_column
        ].to_dict()

        # Create ordered dictionary following the sequence indices
        ordered_dict = {}
        for index in sorted(
            experiment_sequences.keys()
        ):  # Sort by index (0, 1, 2, ...)
            sequence_name = experiment_sequences[index]
            if sequence_name in name_to_value:
                ordered_dict[sequence_name] = name_to_value[sequence_name]

        return ordered_dict
