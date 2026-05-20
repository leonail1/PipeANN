#!/usr/bin/env python3
"""Measure original sparse-label files against mixed densebit sidecars.

This is intentionally a file-size accounting experiment: it does not infer
semantic quality from the files, and every reported number comes from stat(2).
"""

from __future__ import annotations

import csv
import json
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent

PAIRS = [
    (
        "Fashion-MNIST784",
        "experiments/qps_4sets/fashion_mnist784/workloads/uniform_exact_selectivity/base.uniform_exact_selectivity.spmat",
        "data/fashion_mnist784/fashion_mnist784_qps4_pipeann_labels.densebit",
    ),
    (
        "GIST960",
        "experiments/qps_4sets/gist960/workloads/uniform_exact_selectivity/base.uniform_exact_selectivity.spmat",
        "data/gist960/gist960_qps4_pipeann_labels.densebit",
    ),
    (
        "GloVe100",
        "experiments/qps_4sets/glove100/workloads/uniform_exact_selectivity/base.uniform_exact_selectivity.spmat",
        "data/glove100/glove100_qps4_pipeann_labels.densebit",
    ),
    (
        "YFCC10M",
        "data/yfcc100M/base.metadata.10M.spmat",
        "data/yfcc100M/yfcc10m_pipeann_labels.densebit",
    ),
    (
        "SIFT1M/r116",
        "experiments/r116_suite/labels/base_1m.spmat",
        "experiments/r116_suite/exp6_aris_cpu_clean/tmp/direct_1m_labels.densebit",
    ),
]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read_spmat_header(path: Path) -> dict[str, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) != 24:
        raise ValueError(f"{path} is too small to be an spmat file")
    nrow, ncol, nnz = struct.unpack("<qqq", header)
    return {"spmat_rows": nrow, "spmat_cols": ncol, "spmat_nnz": nnz}


def read_densebit_header(path: Path) -> dict[str, int]:
    with path.open("rb") as handle:
        header = handle.read(48)
    if len(header) != 48:
        raise ValueError(f"{path} is too small to be a densebit sidecar")
    _magic, version, npoints, nlabels, words_per_label, nnz = struct.unpack("<QQQQQQ", header)
    return {
        "densebit_version": version,
        "densebit_points": npoints,
        "densebit_labels": nlabels,
        "densebit_words_per_label": words_per_label,
        "densebit_nnz": nnz,
    }


def mib(value: int) -> float:
    return value / (1024.0 * 1024.0)


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, original_rel, processed_rel in PAIRS:
        original = ROOT / original_rel
        processed = ROOT / processed_rel
        if not original.exists():
            raise FileNotFoundError(original)
        if not processed.exists():
            raise FileNotFoundError(processed)

        original_bytes = original.stat().st_size
        processed_bytes = processed.stat().st_size
        ratio = processed_bytes / original_bytes
        row: dict[str, Any] = {
            "dataset": dataset,
            "original_label_path": rel(original),
            "processed_label_path": rel(processed),
            "original_bytes": original_bytes,
            "processed_bytes": processed_bytes,
            "original_mib": round(mib(original_bytes), 6),
            "processed_mib": round(mib(processed_bytes), 6),
            "processed_over_original": round(ratio, 8),
            "processed_over_original_percent": round(ratio * 100.0, 4),
            "space_reduction_percent": round((1.0 - ratio) * 100.0, 4),
        }
        row.update(read_spmat_header(original))
        row.update(read_densebit_header(processed))
        rows.append(row)
    return rows


def write_csv(rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "dataset",
        "original_label_path",
        "processed_label_path",
        "original_bytes",
        "processed_bytes",
        "original_mib",
        "processed_mib",
        "processed_over_original",
        "processed_over_original_percent",
        "space_reduction_percent",
        "spmat_rows",
        "spmat_cols",
        "spmat_nnz",
        "densebit_version",
        "densebit_points",
        "densebit_labels",
        "densebit_words_per_label",
        "densebit_nnz",
    ]
    with (OUT_DIR / "table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_plot(rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(rows, key=lambda row: row["processed_over_original_percent"])
    datasets = [row["dataset"] for row in ordered]
    ratios = [row["processed_over_original_percent"] for row in ordered]

    fig, ax = plt.subplots(figsize=(8.6, 4.5), dpi=240)
    colors = ["#4E79A7", "#59A14F", "#9C755F", "#F28E2B", "#E15759"]
    bars = ax.barh(datasets, ratios, color=colors[: len(ratios)])
    ax.set_xlabel("Processed / original label size (%)", fontsize=11)
    ax.set_title("Mixed Label Sidecar Space", fontsize=13, weight="bold")
    ax.set_xlim(0, max(ratios) * 1.18)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=10)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    for bar, ratio in zip(bars, ratios):
        ax.text(
            bar.get_width() + max(ratios) * 0.018,
            bar.get_y() + bar.get_height() / 2,
            f"{ratio:.2f}%",
            va="center",
            fontsize=10,
        )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "label_space_ratio.png")


def write_summary(rows: list[dict[str, Any]]) -> None:
    summary = {
        "evaluation_type": "file_size_accounting",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Original .spmat label files versus processed mixed densebit sidecar files.",
        "row_count": len(rows),
        "min_processed_over_original_percent": min(
            row["processed_over_original_percent"] for row in rows
        ),
        "max_processed_over_original_percent": max(
            row["processed_over_original_percent"] for row in rows
        ),
        "inputs": [
            {
                "dataset": row["dataset"],
                "original_label_path": row["original_label_path"],
                "processed_label_path": row["processed_label_path"],
            }
            for row in rows
        ],
        "rows": rows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_readme(rows: list[dict[str, Any]]) -> None:
    max_ratio = max(rows, key=lambda row: row["processed_over_original_percent"])
    min_ratio = min(rows, key=lambda row: row["processed_over_original_percent"])
    lines = [
        "# Label Space Ratio Experiment",
        "",
        "ARIS-style experiment record.",
        "",
        "- Evaluation type: `file_size_accounting`.",
        "- Scope: original `.spmat` label files versus processed mixed densebit sidecar files.",
        "- Measurement: byte counts from filesystem `stat`; no synthetic ground truth and no normalized score.",
        "- Reproduce: run `python3 experiments/label_space_ratio/measure_label_space.py` from the repo root on node6.",
        "",
        "## Outputs",
        "",
        "- `table.csv`: per-dataset byte counts and ratios.",
        "- `summary.json`: machine-readable summary and exact input paths.",
        "- `label_space_ratio.png`: high-resolution visualization used by the PPT.",
        "",
        "## Result",
        "",
        f"- Minimum processed/original ratio: {min_ratio['dataset']} at {min_ratio['processed_over_original_percent']:.2f}%.",
        f"- Maximum processed/original ratio: {max_ratio['dataset']} at {max_ratio['processed_over_original_percent']:.2f}%.",
        "- The processed label sidecar is smaller than the original `.spmat` label file for every measured dataset.",
        "",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = build_rows()
    write_csv(rows)
    write_summary(rows)
    write_readme(rows)
    write_plot(rows)
    print(f"wrote {len(rows)} rows to {OUT_DIR}")


if __name__ == "__main__":
    main()
