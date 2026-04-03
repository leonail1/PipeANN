#!/usr/bin/env python3
"""Generate exact-selectivity labels for the SIFT1M hybrid rebuild."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from exact_hybrid_common import (
    canonical_selectivities,
    count_labels_by_id,
    data_labels_path,
    format_sel,
    labels_dir,
    query_labels_path,
    selectivity_map_path,
    write_single_label_query_spmat,
    write_spmat_csr,
)


def generate_data_spmat(ndata: int, selectivities: list[float], seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    row_counts = np.zeros(ndata, dtype=np.int32)
    chosen_by_label: list[np.ndarray] = []

    for sel in selectivities:
        count = int(round(sel * ndata))
        if sel == 1.0:
            chosen = np.arange(ndata, dtype=np.int32)
        else:
            chosen = rng.choice(ndata, size=count, replace=False).astype(np.int32)
        chosen_by_label.append(chosen)
        row_counts[chosen] += 1
        print(f"label sel={format_sel(sel)} -> {count} vectors")

    indptr = np.empty(ndata + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(row_counts, out=indptr[1:])
    indices = np.empty(int(indptr[-1]), dtype=np.int32)
    cursor = indptr[:-1].copy()

    for label_id, chosen in enumerate(chosen_by_label):
        positions = cursor[chosen]
        indices[positions] = label_id
        cursor[chosen] += 1

    data = np.ones(indices.shape[0], dtype=np.float32)
    return indptr, indices, data


def write_selectivity_map(path: Path, selectivities: list[float], indices: np.ndarray) -> None:
    counts = count_labels_by_id(indices, len(selectivities))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# label_id\tcount\tselectivity\n")
        for label_id, sel in enumerate(selectivities):
            handle.write(f"{label_id}\t{int(counts[label_id])}\t{sel:.6f}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(labels_dir()))
    parser.add_argument("--ndata", type=int, default=1_000_000)
    parser.add_argument("--nqueries", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selectivities", default=None,
                        help="Comma-separated selectivities. Defaults to the canonical 9-point exact set.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selectivities = canonical_selectivities(args.selectivities)

    indptr, indices, data = generate_data_spmat(args.ndata, selectivities, args.seed)
    ncol = len(selectivities)

    canonical_output_dir = labels_dir().resolve()
    out_data = data_labels_path() if output_dir == canonical_output_dir else output_dir / "data_labels_exact.spmat"
    write_spmat_csr(out_data, args.ndata, ncol, indptr, indices, data)
    print(f"data labels -> {out_data}")

    for label_id, sel in enumerate(selectivities):
        if output_dir == canonical_output_dir:
            out_query = query_labels_path(sel)
        else:
            out_query = output_dir / f"query_labels_exact_sel{format_sel(sel)}.spmat"
        write_single_label_query_spmat(out_query, args.nqueries, label_id, ncol)
        print(f"query labels sel={format_sel(sel)} -> {out_query}")

    out_map = selectivity_map_path() if output_dir == canonical_output_dir else output_dir / "selectivity_map.tsv"
    write_selectivity_map(out_map, selectivities, indices)
    print(f"selectivity map -> {out_map}")


if __name__ == "__main__":
    main()
