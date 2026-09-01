import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
from .memory import Hash
from typing import Self


@dataclass
class DefaultValues:
    CONFIDENCE: float = 75


class OligossReader:
    def __init__(
        self,
        path_to_parent_folder: str | Path,
        confidence: float = DefaultValues.CONFIDENCE,
    ):
        """This class is used to read the data from the OLIGOSS
        output files."""

        self.path_to_parent_folder: Path = Path(path_to_parent_folder)
        # path to the folder with all data folders. Typically .../extracted
        self.listdir_in_parent_folder: list[Path] = list(
            self.path_to_parent_folder.iterdir()
        )
        self.confidence: float = confidence
        self.hash = Hash(confidence=self.confidence)
        # if labbook exists, then read the data from there

    @classmethod
    def read_from_labbook(
        cls,
        _name: str,
        labbook: pd.DataFrame,
    ) -> Self:
        """Labbbook contians both experimetn ids and also
        parent, which is other type of id for the experiements
        according how they were run together in the MS instrument.
        Primarily, I expect to get parent id. If not found,
        I will try to search for the experiment id and get
        associated parent.

        _name: expected parent id, but can be also experiment id

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        # if labbook_location is string, convert it to Path
        # lb: pd.DataFrame = pd.read_csv(labbook_location)  # noqa

        # is it parent name?
        if not (_prnt := labbook.loc[labbook.parent == _name]).empty:
            location: str = _prnt.location.iloc[0]
        # if not, then check if it is experiment id?
        elif not (_prnt := labbook.loc[labbook.experiment_id == _name]).empty:
            location: str = _prnt.location.iloc[0]
        # I cannot find the experiment id
        else:
            ValueError(f"Experiment {_name} not found in labbook")
        return cls(path_to_parent_folder=location)

    def run_all(
        self,
        confidence: float = DefaultValues.CONFIDENCE,
        string_only: bool = False,
    ) -> pd.DataFrame:
        """Returns a dataframe with the name of the folder,
        the assembly and the diversity of the sequences.

        Args:
            confidence (float, optional): Defaults DefaultValues.CONFIDENCE.
            string_only (bool, optional): Defaults False.

        Returns:
            _type_: pandas.DataFrame with columns: "name", "assembly", "diversity"
        """

        df_to_return = pd.DataFrame(columns=["name", "assembly", "diversity"])
        for x in tqdm(self.listdir_in_parent_folder):
            total_path = self.path_to_parent_folder / x
            path_to_csv = self.prepare_data(total_path)
            df = pd.read_csv(path_to_csv)
            seq = self.filter_confidence(df, confidence=confidence)
            diversity = len(seq)
            if string_only:
                A = np.exp(seq.apply(self.peptide_to_string_assembly)).mean()
            else:
                A = np.exp(
                    seq.apply(self.peptide_to_assembly)
                ).mean()  # ensamble assembly

            new_df = pd.DataFrame(
                {"name": str(x), "assembly": [A], "diversity": [diversity]}
            )
            df_to_return = pd.concat([df_to_return, new_df], axis=0)
        return df_to_return

    def run_all_sequences(
        self, confidence: float = DefaultValues.CONFIDENCE
    ) -> list[pd.DataFrame]:
        """Returns a list of dataframes with sequences, cut by confidence.

        Args:
            confidence (float, optional): Defaults 70.

        Returns:
            list[pd.DataFrame]: List of dataframes with sequences.
        """
        list_of_dataframes = []
        for x in tqdm(self.listdir_in_parent_folder):
            total_path = self.path_to_parent_folder / x
            path_to_csv = self.prepare_data(total_path)
            df = pd.read_csv(path_to_csv)
            seq = self.filter_confidence(df, confidence=confidence)
            list_of_dataframes.append(seq)
        return list_of_dataframes

    def return_sequences(
        self, confidence: float | None = None
    ) -> dict[str, list[str]]:
        """Returns a dictionary with the name of the folder as key
        and the list of sequences as value. Sequences are cut by confidence.

        Args:
            confidence (float | None, optional): Set the cut for confidence.
            If None, then the default value of the instance is used.
            (default in 75)

        Returns:
            dict[str, list[str]]: keys are names of the folders,
            values are lists of sequences
        """

        confidence = self.confidence if confidence is None else confidence
        df_to_return: dict = {}
        for folder in tqdm(self.listdir_in_parent_folder):
            total_path: Path = self.path_to_parent_folder / folder
            df: pd.DataFrame = self.read_data(total_path)
            seq = self.filter_confidence(df, confidence=confidence)
            seq2save = list(seq)
            df_to_return.update({folder.name: seq2save})
        self.sequences = df_to_return
        return df_to_return

    def return_all_data(self, disable_progress_bar: bool = False) -> dict[str, pd.DataFrame]:
        """Returns all data from all files in the folder,
        including confidences. The parent folder is defined
        when the instance is created.

        Returns:
            dict[str, pd.DataFrame]: _description_
        """

        dict_to_return: dict[str, pd.DataFrame] = {}
        print_name: str = self.nice_printout(self.path_to_parent_folder)
        for folder in tqdm(
            self.listdir_in_parent_folder,
            desc=f"Reading data from folder {print_name}",
            disable=disable_progress_bar,
        ):
            # check if the analysis was run before
            hash_key: str = self.hash.generate_hash_key(
                key=folder.name, reading_data=True
            )
            if self.hash.check_if_analysis_run_before(hash_key):
                df = self.hash.load_pickled_analysis(hash_key)
                dict_to_return.update({folder.name: df})
                continue

            total_path: Path = self.path_to_parent_folder / folder
            df: pd.DataFrame = self.read_data(total_path)
            dict_to_return.update({folder.name: df})
            self.hash.pickle_current_analysis(df, hash_key)
            # save the analysis
        return dict_to_return

    def nice_printout(self, path: Path) -> str:
        """Simple helper formatting the path to a nice printout.
        example:

            path = Path(".../extracted/2021-09-01/2021-09-01_12-00-00")
            nice_printout(path) -> "extracted>2021-09-01>12-00-00"""

        number_of_subfolders: int = len(path.parts)
        if number_of_subfolders > 4:
            print_level: int = 4
        else:
            print_level: int = number_of_subfolders
        print_name: str = ">".join(
            self.path_to_parent_folder.parent.parts[-print_level:-1]
        )
        return print_name

    def prepare_data(
        self,
        filepath: str | Path,
    ) -> Path:
        """Returns path to csv file to read. If multiple files are created by
        OLIGOSS, all data are combined into one 'result.csv' file."""
        filepath_to_folder: Path = Path(filepath)

        resultpath: Path = filepath_to_folder / "result.csv"

        if resultpath.exists():
            return resultpath

        filelist: list[str] = [x.name for x in filepath_to_folder.iterdir()]
        filelist.sort()

        if len(filelist) == 1 and filelist[0] == "sequencing_summary_1.csv":
            return filepath_to_folder / "sequencing_summary_1.csv"

        elif len(filelist) < 11:
            data_merge: pd.DataFrame = pd.DataFrame()
            for index, csvname in enumerate(filelist):
                d_temp = pd.read_csv(filepath_to_folder / csvname)
                if index == 0:
                    data_merge = d_temp
                else:
                    data_merge = pd.concat([data_merge, d_temp], axis=0)
            data_merge.to_csv(filepath_to_folder / "result.csv")
            return filepath_to_folder / "result.csv"

        elif len(filelist) >= 11:
            with open(filepath_to_folder / "result.csv", "wb") as outfile:
                for f in filelist:
                    with open(filepath_to_folder / f, "rb") as infile:
                        outfile.write(infile.read())
            if resultpath.exists():
                return resultpath
            else:
                raise ValueError("result.csv was not created.")
        else:
            raise ValueError(
                f"{filepath_to_folder} does not contain any valid csv files."
            )

    def read_data(self, filepath_to_read: str | Path) -> pd.DataFrame:
        """Reads OLIGOSS result from csv file in the folder to
        which one provides the path. Return Dataframe with sequences
        and confidences."""
        return pd.read_csv(self.prepare_data(filepath_to_read)).loc[
            ::, ["sequence", "confidence", "composition"]
        ]

    @classmethod
    def filter_confidence(
        cls, df: pd.DataFrame, confidence: float = DefaultValues.CONFIDENCE
    ) -> pd.DataFrame:
        """Filters the dataframe by confidence and returns only sequences
        with confidence higher than the provided value.
        """
        return df.loc[
            (df.loc[:, "confidence"] >= confidence)
            & (df.loc[:, "sequence"].apply(len) > 0),
            "sequence",
        ]

    @staticmethod
    def read_json(file_path: Path) -> dict:
        """Helper reading json files and returning dictionary."""
        with open(file_path, "r") as f:
            return json.load(f)

    @staticmethod
    def get_default_datapath(experiment_name: str) -> Path:
        def get_output_path(parameter_file: dict) -> Path:
            return Path(parameter_file["output_folder"])

        experiment_inputparameters: Path = Path(
            f"../../experiments/{experiment_name}/inputparameters.json"
        )
        return (
            get_output_path(OligossReader.read_json(experiment_inputparameters))
            / "extracted"
        )

    @staticmethod
    def filter_by_confidence(
        df: pd.DataFrame, confidence: float
    ) -> pd.DataFrame:
        return df.loc[df["confidence"] >= confidence, ::]

    @staticmethod
    def dataframe_to_list(df: pd.DataFrame, confidence: int) -> list[str]:
        return list(OligossReader.filter_by_confidence(df, confidence).sequence)


class DataFilter:
    def __init__(
        self,
        data: dict[str, pd.DataFrame],
        exp_names: list[str] | None = None,
    ):
        self.data: dict[str, pd.DataFrame] = data
        self.drop_composition()
        self.exp_names: str | list[str] | None = exp_names
        self.filtered_data: dict[str, pd.DataFrame] = {}

    def drop_composition(self) -> None:
        self.data = {
            key: value.drop(columns=["composition"])
            for key, value in self.data.items()
        }

    def filter_by_keys(self, keys: list[str] | None) -> dict[str, pd.DataFrame]:
        """Takes the large dictionary of data and filters them by names
        provided as list of strings.

        Args:
            keys (list[str]): list of keys to filter by.
            If not in data, it will be ignored.

        Returns:
            dict[str, pd.DataFrame]: same data structure as the large dict,
            but filtered by keys.
        """
        if keys is None:
            return self.data
        return {key: self.data[key] for key in keys if key in self.data}

    def _filter_by_confidence(
        self, df: pd.DataFrame, confidence: int | float
    ) -> pd.DataFrame:
        return df.loc[df.confidence >= confidence, ::]

    def _order_peptides(self, df: pd.DataFrame) -> pd.DataFrame:
        # first order by size, than alphabetically.
        try:
            # df["sequence_length"] = df.sequence.apply(len)
            df = df.sort_values(by="sequence", key=lambda x: x.str.len())
        except Exception as e:
            print(f"Error in ordering peptides: {e}")
        return df

    def filter_by_confidence(
        self, confidence: int | float = 70
    ) -> dict[str, list[str]]:
        self.filtered_data = self.filter_by_keys(self.exp_names)
        self.filtered_data = {
            k: self._filter_by_confidence(v, confidence)
            for k, v in self.filtered_data.items()
        }
        self.filtered_data = {
            k: self._order_peptides(v) for k, v in self.filtered_data.items()
        }

        return self.df_to_list(self.filtered_data)

    @classmethod
    def series_to_list(cls, data: pd.Series) -> list[str]:
        return [str(x) for x in list(data)]

    @classmethod
    def df_to_list(cls, data: dict[str, pd.DataFrame]) -> dict[str, list[str]]:
        return {k: cls.series_to_list(v.sequence) for k, v in data.items()}


class SortExperiments:
    """This splits experimetns that were run
    together into separate elements of a list"""

    def __init__(
        self,
        sequences: dict[str, list[str | None]],
        number_of_independent_experiments: int,
        total_number_of_datapoints: int,
    ):
        self.sequences = sequences
        self.rxns = number_of_independent_experiments
        self.total = total_number_of_datapoints

        self.common_prefix = self.return_common_prefix()
        self.number_keys_int = self.return_number_keys_as_int()

        self.sanity_check_numbering()
        self.index_matrix = self.generate_index_matrix()
        self.key_matrix_to_return = self.key_matrix()

    def return_ordered_sequences(self) -> list[list[dict[str, str | None]]]:
        """Returns a list of dictionaries with the
        sequences ordered by experiment"""
        return [
            [{j: self.sequences[j]} for j in i]
            for i in self.key_matrix_to_return
        ]

    def generate_index_matrix(self) -> list[list[int]]:
        return [
            [i for i in range(k, self.total + 1, self.rxns)]
            for k in range(1, self.rxns + 1, 1)
        ]

    def return_common_prefix(self) -> str:
        return os.path.commonprefix(list(self.sequences.keys()))

    def sanity_check_numbering(self) -> None:
        if self.number_keys_int[0] != 1:
            raise ValueError("First key is not 1")
        if self.number_keys_int[-1] != self.total:
            raise ValueError("Last key is not same as expected length")
        if len(self.number_keys_int) != self.total:
            raise ValueError("Length of keys is not same as expected length")

    def return_number_keys_as_int(self) -> set[int]:
        return sorted(
            {
                int(x)
                for x in {
                    k[len(self.common_prefix) :] for k in self.sequences.keys()
                }
            }
        )

    def decimals(self, index_matrix: list[list[int]]) -> int:
        """Returns the number of decimals required to
        represent the largest number in the index matrix
        """
        return len(str(max(max(index_matrix))))

    def key_matrix(self) -> list[list[str]]:
        """Returns a matrix of keys with the same shape as the index matrix"""
        return [
            [
                f"{self.common_prefix}{i:0{self.decimals(self.index_matrix)}}"
                for i in k
            ]
            for k in self.index_matrix
        ]

    # split_experiment_keys = key_matrix(index_matrix)


def multiple_trajectories(
    array_of_array_of_sequences: list[list[dict[str, list[str]]]],
) -> list[list[str]]:
    return [return_cummulative(i) for i in array_of_array_of_sequences]


def return_cummulative(
    array_of_sequences: list[dict[str, list[str]]],
) -> list[list[str]]:
    """This is for single trajectory,
    for sake of clarity and readability"""
    # TODO: add sanity checks (distance between points to make sure)
    cumulative_list: list = []
    k: set = set()
    for j in array_of_sequences:
        k = k.union(list(j.values())[0])
        cumulative_list.append(list(k))
    return cumulative_list


def return_cumulative_for_given_keys(
    sequences: dict[str, list[str]], keys: list[str]
) -> dict[str, list[str]]:
    """
    Takes a dictionary of sequences and
    returns a dictionary of cumulative sequences.

    Args:
        sequences (dict[str, list[str]]): sequences
        keys (list[str]): keys, for which (and in which order)
        the cumulative sequences are returned

    Raises:
        ValueError: Raise error if not all keys are in sequences.

    Returns:
        dict[str, list[str]]: Cumulative sequences
    """
    if not all(key in sequences for key in keys):
        for key in keys:
            if key not in sequences:
                print(f"Key: {key} was not found in {sequences.keys()}")
        raise ValueError("Not all keys are in sequences. ")
    cumulative_dict: dict[str, list[str]] = {}
    k: set = set()  # temporal set to store all sequences
    for key in keys:
        seqs = sequences[key]
        new_key_name: str = "cumul_" + key
        k = k.union(seqs)
        cumulative_dict.update(
            {new_key_name: sorted(list(k), key=lambda x: (len(x), x))}
        )
    return cumulative_dict
