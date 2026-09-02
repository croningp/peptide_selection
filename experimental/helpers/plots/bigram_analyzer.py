import numpy as np


class BigramAnalyzer:
    """
    Analysis for both single-character and multi-character patterns.
    Detects pattern type and applies appropriate analysis method.
    """

    def __init__(self, patterns: list[str] = ["G", "A", "V", "P", "H"]):
        """
        Initialize the bigram analyzer.

        Args:
            patterns: List of patterns to track.
                     - Single characters: ["G", "A", "V", "P", "H"] (default)
                     - Two-character patterns: ["GA", "AV", "VP", "PG"]
        """
        self.patterns = patterns
        self.pattern_to_idx = {
            pattern: i for i, pattern in enumerate(self.patterns)
        }

        # Determine pattern type
        self._analyze_pattern_type()

        # For backward compatibility - keep 'letters' attribute
        self.letters = self.patterns

    def _analyze_pattern_type(self):
        """Analyze and determine if patterns are single or multi-character."""
        lengths = [len(str(pattern)) for pattern in self.patterns]
        unique_lengths = set(lengths)

        if len(unique_lengths) == 1:
            length = unique_lengths.pop()
            if length == 1:
                self.pattern_type = "single_char"
            elif length == 2:
                self.pattern_type = "two_char"
            else:
                self.pattern_type = "multi_char"
                raise ValueError(
                    "Patterns must be single or two-character. "
                    f"Detected length: {length}."
                )
        else:
            # Mixed lengths - default to multi_char handling
            self.pattern_type = "multi_char"
            raise ValueError(
                "Mixed pattern lengths detected, this is not implemented."
                " Please ensure all patterns are of the same length."
            )

    def _find_pattern_positions(self, string, pattern):
        """Find all positions where a pattern occurs."""
        string = str(string)
        pattern = str(pattern)
        positions = []
        for i in range(len(string) - len(pattern) + 1):
            if string[i : i + len(pattern)] == pattern:
                positions.append((i, i + len(pattern)))
        return positions

    def _compute_single_char_bigram_matrix(self, strings):
        """Compute bigram matrix for single-character patterns."""
        n = len(self.patterns)
        bigram_matrix = np.zeros((n, n))

        for string in strings:
            string = str(string)
            for i in range(len(string) - 1):
                char1, char2 = string[i], string[i + 1]
                if (
                    char1 in self.pattern_to_idx
                    and char2 in self.pattern_to_idx
                ):
                    bigram_matrix[
                        self.pattern_to_idx[char1], self.pattern_to_idx[char2]
                    ] += 1

        return bigram_matrix

    def _compute_multi_char_bigram_matrix(self, strings):
        """Compute bigram matrix for multi-character patterns."""
        n = len(self.patterns)
        bigram_matrix = np.zeros((n, n))

        for string in strings:
            string = str(string)

            # Find all pattern matches with positions
            matches = []
            for pattern in self.patterns:
                positions = self._find_pattern_positions(string, pattern)
                for start, end in positions:
                    matches.append((start, end, pattern))

            # Sort by position and remove overlaps
            matches.sort(key=lambda x: x[0])
            non_overlapping = []
            last_end = -1

            for start, end, pattern in matches:
                if start >= last_end:
                    non_overlapping.append((start, end, pattern))
                    last_end = end

            # Create bigrams from consecutive patterns
            for i in range(len(non_overlapping) - 1):
                pattern1 = non_overlapping[i][2]
                pattern2 = non_overlapping[i + 1][2]

                if (
                    pattern1 in self.pattern_to_idx
                    and pattern2 in self.pattern_to_idx
                ):
                    bigram_matrix[
                        self.pattern_to_idx[pattern1],
                        self.pattern_to_idx[pattern2],
                    ] += 1

        return bigram_matrix

    def compute_bigram_matrix(self, strings):
        """
        Compute bigram matrix using the appropriate method based on pattern type.

        Args:
            strings: List of strings to analyze

        Returns:
            numpy.ndarray: Square matrix of bigram counts
        """
        # Handle original nested format for backward compatibility
        if (
            isinstance(strings, list)
            and len(strings) > 0
            and isinstance(strings[0], (list, np.ndarray))
        ):
            actual_strings = strings[0]
        else:
            actual_strings = strings

        if self.pattern_type == "single_char":
            return self._compute_single_char_bigram_matrix(actual_strings)
        else:
            return self._compute_multi_char_bigram_matrix(actual_strings)

    def compute_multiple_datasets(self, datasets):
        """
        Compute bigram matrices for multiple datasets.

        Args:
            datasets: Dictionary with dataset names as keys and data as values

        Returns:
            dict: Dictionary with dataset names as keys and bigram matrices as values
        """
        bigram_matrices = {}
        for name, strings in datasets.items():
            bigram_matrices[name] = self.compute_bigram_matrix(strings)

        return bigram_matrices

    def get_global_min_max(self, bigram_matrices):
        """Calculate global minimum and maximum values across all matrices."""
        all_values = []
        for matrix in bigram_matrices.values():
            all_values.extend(matrix.flatten())

        return min(all_values) if all_values else 0, max(
            all_values
        ) if all_values else 1

    def analyze_single_dataset(self, strings, dataset_name="Dataset"):
        """
        Analyze a single dataset and return comprehensive results.

        Args:
            strings: Data to analyze (handles various input formats)
            dataset_name: Name for the dataset

        Returns:
            dict: Analysis results including matrix, statistics, and metadata
        """
        bigram_matrix = self.compute_bigram_matrix(strings)

        # Calculate statistics
        total_bigrams = np.sum(bigram_matrix)
        most_common_bigram = np.unravel_index(
            np.argmax(bigram_matrix), bigram_matrix.shape
        )
        most_common_count = bigram_matrix[most_common_bigram]

        # Convert indices back to patterns
        most_common_patterns = (
            self.patterns[most_common_bigram[0]],
            self.patterns[most_common_bigram[1]],
        )

        # Handle nested format for string count
        if (
            isinstance(strings, list)
            and len(strings) > 0
            and isinstance(strings[0], (list, np.ndarray))
        ):
            string_count = len(strings[0])
        else:
            string_count = len(strings) if isinstance(strings, list) else 1

        return {
            "name": dataset_name,
            "matrix": bigram_matrix,
            "total_bigrams": int(total_bigrams),
            "most_common_bigram": most_common_patterns,
            "most_common_count": int(most_common_count),
            "patterns": self.patterns,
            "letters": self.patterns,  # For backward compatibility
            "pattern_type": self.pattern_type,
            "string_count": string_count,
        }

    def compare_datasets(self, datasets):
        """
        Compare multiple datasets and return comprehensive analysis.

        Args:
            datasets: Dictionary with dataset names as keys and data as values

        Returns:
            dict: Comprehensive comparison results
        """
        bigram_matrices = self.compute_multiple_datasets(datasets)
        global_min, global_max = self.get_global_min_max(bigram_matrices)

        # Analyze each dataset
        dataset_analyses = {}
        for name, strings in datasets.items():
            dataset_analyses[name] = self.analyze_single_dataset(strings, name)

        return {
            "matrices": bigram_matrices,
            "global_min": global_min,
            "global_max": global_max,
            "dataset_analyses": dataset_analyses,
            "patterns": self.patterns,
            "letters": self.patterns,  # For backward compatibility
            "pattern_type": self.pattern_type,
            "dataset_count": len(datasets),
        }

    def print_analysis_summary(self, dataset_name, analysis_result):
        """Print a detailed summary of the analysis."""
        result = analysis_result["dataset_analyses"][dataset_name]
        matrix = result["matrix"]

        print(f"\n=== Analysis Summary: {dataset_name.title()} ===")
        print(f"Pattern type: {result['pattern_type']}")
        print(f"Patterns analyzed: {result['patterns']}")
        print(f"Total bigrams found: {result['total_bigrams']}")
        print(f"String count: {result['string_count']}")

        if result["total_bigrams"] > 0:
            print(
                f"Most common bigram: {result['most_common_bigram'][0]} → {result['most_common_bigram'][1]} ({result['most_common_count']} times)"
            )

            # Top 5 transitions
            flat_indices = np.argsort(matrix.flatten())[::-1]
            print("\nTop 5 transitions:")
            for i, flat_idx in enumerate(flat_indices[:5]):
                row, col = np.unravel_index(flat_idx, matrix.shape)
                count = matrix[row, col]
                if count > 0:
                    print(
                        f"  {i + 1}. {self.patterns[row]} → {self.patterns[col]}: {int(count)} times"
                    )
        print()

    @staticmethod
    def create_oriented_pairwise_matrix(
        strings, letters=["G", "A", "V", "P", "H"]
    ):
        """
        Static method to create oriented pairwise bigram matrix.

        Args:
            strings: List of strings to analyze (handles original format)
            letters: List of letters to track

        Returns:
            numpy.ndarray: Bigram matrix
        """
        analyzer = BigramAnalyzer(letters)
        return analyzer.compute_bigram_matrix(strings)
