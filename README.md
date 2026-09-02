# Quantifying Directedness in Chemical Reaction Networks Using Assembly Theory

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Code for the paper.

Michael Jirasek, Abhishek Sharma, Mary Wong, Jennifer Munro, Leroy Cronin\*
School of Chemistry, University of Glasgow

## About the paper

This work uses [Assembly Theory](https://doi.org/10.1038/s41586-023-06600-9) to quantify how directed vs. undirected an open-ended reaction network is, using the **exploration ratio (ER)** and **ensemble assembly (A)**. Applied to peptide ensembles, non-specific activation (heat-driven wet–dry cycling, CDI coupling) gives high ER (broad, undirected exploration), while sequence-selective proteases (papain, bromelain, trypsin, chymotrypsin) impose measurably lower ER and higher A, consistent with their independently characterised substrate selectivity. This repository builds the Joint Assembly Space (JAS) from experimentally observed peptide sequences, computes ER and A, and produces the manuscript and Supporting Information figures.

## Pipeline

```
HPLC-MS/MS raw data
        │  (OLIGOSS: b/y fragment matching, ≥70% coverage threshold)
        ▼
  annotated peptide sequences  ──►  labbook.csv (experiment metadata & sequences)
        │
        ▼
  AssemblyGo  (external binary, github.com/croningp/assembly_go)
        │  shortest construction pathway per sequence
        ▼
  MolecularAssembly/  (per-sequence pathway cache, gitignored)
        │
        ▼
  helpers.data_extractors  ──►  JAS union, ER, ensemble assembly A, diversity
        │
        ▼
  helpers.plots / graph_visualizer  ──►  manuscript & ESI figures
```

## Requirements

- Python 3.12 with `networkx` 2.8.8, `pandas`, `numpy`, `matplotlib`, `plotly`, `rdkit`, `tqdm`, `pyarrow`
- [AssemblyGo](https://github.com/croningp/assembly_go) binary, for computing shortest construction pathways
- Wolfram Mathematica 14 for the notebooks in `mathematica_notebooks/`

## Usage

1. Place OLIGOSS-annotated sequence data and experiment metadata as referenced in `labbook.csv`.
2. Point `AssemblyConfig`/`AllExperiments` (in `helpers/data_extractors/pathway_helper.py` and `all_experiments.py`) at a local build of AssemblyGo and a `MolecularAssembly/` working directory.
3. Run `plots_static.ipynb` to compute ER, ensemble assembly A, and diversity for each experiment group and reproduce Figures 5 and 6 and the corresponding ESI figures.
4. Use `visualise_JAS.ipynb` / `graph_visualizer.py` to draw individual JAS graphs (Figure 6), and the scripts in `copy_number_sensitivity/` for the bounded abundance-gradient sensitivity check.

## Related repositories

- [assembly_go](https://github.com/croningp/assembly_go) — assembly-index/pathway calculation for experimentally observed sequences
- [assemblycpp-v5](https://github.com/croningp/assemblycpp-v5) — assembly-pathway implementation used for the simulated sequence-space model
- [molecular_complexity](https://github.com/croningp/molecular_complexity) — companion study relating molecular assembly index to NMR/IR/MS spectroscopic complexity

## License

Released under the [MIT License](LICENSE).
