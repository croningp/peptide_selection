"""Parse simulation SelectiveFac_* data into per-folder JAS metrics.

Scope: SelectiveFac_1, _10, _25, _50, _75, _100, all string_data_* folders
within each -> one metrics row per (factor, string_data).

node_list/edge_list are looked up by sequence from MolecularAssembly.parquet
(no `helpers` package dependency); assembly_index comes fresh from each
folder's seqList_*.txtOut summary.

Usage:
    python parse_simulation_to_jas.py \
        --base /path/to/PeptideSelectionDataFinal \
        --out  jas_final.csv \
        --workers 48
        """
from __future__ import annotations

import argparse
import csv
import json
import re
from multiprocessing import Pool
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PARQUET = SCRIPT_DIR / "MolecularAssembly.parquet"

FACTORS = [
    "SelectiveFac_1",
    "SelectiveFac_10",
    "SelectiveFac_25",
    "SelectiveFac_50",
    "SelectiveFac_75",
    "SelectiveFac_100",
]
STRING_DATA_RANGE = range(1, 26)  # string_data_1 .. string_data_25
MA_LINE = re.compile(r"^(.*) has assembly index: (\d+)$")


# --- per-worker globals (populated by _init_worker) -------------------------
# Each pool worker loads its own copy of MolecularAssembly.parquet into a
# sequence -> {assembly_index, node_list, edge_list} dict; this is the only
# source of node_list/edge_list data in the whole script.
_LOOKUP: dict[str, dict] | None = None
_PARQUET_PATH: Path = DEFAULT_PARQUET


def _init_worker(parquet_path: str):
    """Pool initializer: load the MolecularAssembly.parquet lookup for this worker."""
    global _LOOKUP, _PARQUET_PATH
    _PARQUET_PATH = Path(parquet_path)
    _LOOKUP = (
        pd.read_parquet(_PARQUET_PATH)
        .set_index("sequence")
        .to_dict(orient="index")
    )


def parse_ma(folder: Path) -> dict[str, int]:
    """Ground-truth assembly index per sequence, from seqList_*.txtOut summaries."""
    ma: dict[str, int] = {}
    for log in folder.glob("seqList_*.txtOut"):
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            m = MA_LINE.match(line)
            if m:
                ma[m.group(1)] = int(m.group(2))
    return ma


def observed_sequence(text: str) -> str:
    """Sequence identifier for one *_Pathway JSON (plain field access, no helpers)."""
    return json.loads(text)["file_graph"][0]["Fragments"][0]


def create_graph_from_pathways(df: pd.DataFrame) -> nx.DiGraph | None:
    """Union DiGraph from per-pathway edge_lists (mirrors StringToPaths.create_graph_from_pathways)."""
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


def process_folder(task: tuple[str, str, str]) -> dict:
    """Full per-folder pipeline: parse -> df_paths -> union graph -> metrics.

    node_list/edge_list come from the local MolecularAssembly.parquet lookup
    (no `helpers` package needed); assembly_index is still the ground-truth
    value from this folder's seqList_*.txtOut summary.
    """
    factor, sd, folder_str = task
    folder = Path(folder_str)
    ma = parse_ma(folder)

    rows = []
    errors = 0
    for path in folder.glob("*_Pathway"):
        try:
            observed = observed_sequence(path.read_text(encoding="utf-8"))
        except Exception:
            errors += 1
            continue
        if observed not in ma:
            errors += 1
            continue
        cached = _LOOKUP.get(observed)
        if cached is None:
            errors += 1
            continue
        rows.append(
            {
                "peptide": observed,
                "assembly_index": ma[observed],
                "node_list": np.asarray(cached["node_list"]),
                "edge_list": np.asarray(cached["edge_list"]),
            }
        )

    df = (
        pd.DataFrame(rows, columns=["peptide", "assembly_index", "node_list", "edge_list"])
        .drop_duplicates(subset=["peptide"])
        .reset_index(drop=True)
    )

    graph = create_graph_from_pathways(df)
    if graph is None or df.empty:
        return {
            "selective_fac": factor, "string_data": sd, "pathway_files": len(rows),
            "diversity": 0, "complexity": 0,
            "observed_nodes": 0, "all_nodes": 0, "exploration_ratio": 0.0,
            "A": 0.0, "max_MA": 0, "errors": errors,
        }

    observed_nodes = df["peptide"].unique()
    all_nodes = np.unique(np.concatenate([nodes for nodes in df["node_list"]]))

    return {
        "selective_fac": factor,
        "string_data": sd,
        "pathway_files": len(rows),
        "diversity": int(len(df)),
        "complexity": int(graph.number_of_edges()),
        "observed_nodes": int(len(observed_nodes)),
        "all_nodes": int(len(all_nodes)),
        "exploration_ratio": float(len(observed_nodes) / len(all_nodes)) if len(all_nodes) > 0 else 0.0,
        "A": float(np.mean(df["assembly_index"].apply(np.exp))),
        "max_MA": int(df["assembly_index"].max()),
        "errors": errors,
    }


def natural(value: str) -> int:
    try:
        return int(value.rsplit("_", 1)[1])
    except Exception:
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="./PeptideSelectionDataFinal")
    parser.add_argument("--out", default="jas_survey_fac1_100.csv")
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--parquet", default=str(DEFAULT_PARQUET))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = Path(args.base)

    tasks: list[tuple[str, str, str]] = []
    for factor in FACTORS:
        for i in STRING_DATA_RANGE:
            folder = base / factor / f"string_data_{i}"
            if folder.is_dir():
                tasks.append((factor, f"string_data_{i}", str(folder)))
    print(f"folders to process: {len(tasks)}  workers: {args.workers}", flush=True)

    fields = [
        "selective_fac", "string_data", "pathway_files", "diversity", "complexity",
        "observed_nodes", "all_nodes", "exploration_ratio", "A", "max_MA", "errors",
    ]

    results: list[dict] = []
    # Write incrementally so a crash mid-run still leaves partial output.
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        with Pool(processes=args.workers, initializer=_init_worker, initargs=(args.parquet,)) as pool:
            for done, row in enumerate(pool.imap_unordered(process_folder, tasks), start=1):
                results.append(row)
                writer.writerow(row)
                handle.flush()
                print(
                    f"[{done}/{len(tasks)}] {row['selective_fac']}/{row['string_data']} "
                    f"files={row['pathway_files']} obs={row['observed_nodes']} all={row['all_nodes']} "
                    f"exp_ratio={row['exploration_ratio']} errors={row['errors']}",
                    flush=True,
                )

    # Rewrite sorted for a tidy final file.
    results.sort(key=lambda r: (natural(r["selective_fac"]), natural(r["string_data"])))
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    print(f"done. {len(results)} rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
