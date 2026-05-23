#!/usr/bin/env python3
"""Plot ARIS PQ drift experiment outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ensure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save(fig, figures: Path, name: str) -> list[str]:
    figures.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in ("png", "pdf"):
        path = figures / f"{name}.{suffix}"
        fig.savefig(path, dpi=180)
        written.append(str(path))
    return written


def cycle_idx(row: dict[str, Any]) -> int | None:
    if row.get("cycle_idx") is not None:
        return int(row["cycle_idx"])
    text = " ".join(str(row.get(key, "")) for key in ["dest_prefix", "source_prefix", "cycle", "prefix"])
    import re

    match = re.search(r"cycle[_-]?0*([0-9]+)", text)
    return int(match.group(1)) if match else None


def selected_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in [
        root / "raw/phaseB_selected_route_l.jsonl",
        root / "raw/phaseC_selected_route_l.jsonl",
        root / "raw/phase4_selected_route_l.jsonl",
    ]:
        for row in read_jsonl(path):
            selected = row.get("selected", row)
            merged = dict(selected)
            for key in ["phase", "variant", "cycle_idx", "selector_type", "bucket"]:
                if key in row and key not in merged:
                    merged[key] = row[key]
            rows.append(merged)
    if not rows:
        rows = read_jsonl(root / "raw/selected_route_l.jsonl")
    return rows


def plot(root: Path) -> list[str]:
    plt = ensure_matplotlib()
    figures = root / "figures"
    written: list[str] = []

    deletes = read_jsonl(root / "raw/phaseC_delete_steps.jsonl")
    inserts = read_jsonl(root / "raw/phaseC_no_retrain_cycles.jsonl")
    if deletes or inserts:
        cycles = sorted({cycle_idx(r) for r in deletes + inserts if cycle_idx(r) is not None})
        fig, ax = plt.subplots(figsize=(7.6, 4.4))
        if deletes:
            ax.plot([cycle_idx(r) for r in deletes], [r["delete_elapsed_s"] for r in deletes], marker="o", label="delete API")
            ax.plot([cycle_idx(r) for r in deletes], [r["merge_elapsed_s"] for r in deletes], marker="o", label="delete-side merge")
        if inserts:
            ax.plot([cycle_idx(r) for r in inserts], [r["insert_elapsed_s"] for r in inserts], marker="o", label="insert no-retrain")
            ax.plot([cycle_idx(r) for r in inserts], [r["merge_elapsed_s"] for r in inserts], marker="o", label="insert-side merge")
        ax.set_xticks(cycles)
        ax.set_xlabel("cycle")
        ax.set_ylabel("seconds")
        ax.set_title("Delete / Insert / Merge Cost")
        ax.legend()
        fig.tight_layout()
        written += save(fig, figures, "phaseC_delete_insert_merge_cost")
        plt.close(fig)

    selected = selected_rows(root)
    if selected:
        fig, ax = plt.subplots(figsize=(8.4, 4.6))
        ordered = sorted(selected, key=lambda r: (str(r.get("phase", "")), int(r.get("cycle_idx") or 0), str(r.get("variant", "")), str(r.get("selector_type", "")), str(r.get("bucket", ""))))
        x = list(range(1, len(ordered) + 1))
        ax.plot(x, [r.get("recall@10", 0) for r in ordered], marker="o", linewidth=1)
        ax.axhline(98.0, color="#666666", linestyle="--", linewidth=1)
        ax.set_title("Selected Recall Across PQ Drift Runs")
        ax.set_xlabel("selected point order")
        ax.set_ylabel("recall@10 (%)")
        fig.tight_layout()
        written += save(fig, figures, "selected_recall")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.4, 4.6))
        ax.plot(x, [r.get("avg_latency_us", 0) / 1000.0 for r in ordered], marker="o", label="avg")
        ax.plot(x, [r.get("p95_latency_us", 0) / 1000.0 for r in ordered], marker="s", label="p95")
        ax.set_title("Selected Latency Across PQ Drift Runs")
        ax.set_xlabel("selected point order")
        ax.set_ylabel("latency (ms)")
        ax.legend()
        fig.tight_layout()
        written += save(fig, figures, "selected_latency")
        plt.close(fig)

    penalty = read_jsonl(root / "raw/phaseC_penalty.jsonl") + read_jsonl(root / "raw/phaseB_penalty.jsonl")
    if penalty:
        fig, ax = plt.subplots(figsize=(8.0, 4.5))
        labels = [f"{r.get('phase') or 'phase'}-{r.get('cycle_idx', '')}\n{r.get('selector_type')}-{r.get('bucket')}" for r in penalty]
        values = [r.get("selected_feasible_delta_ms") for r in penalty]
        colors = ["#2878b5" if v is not None and v <= 0 else "#c85200" for v in values]
        ax.bar(range(len(values)), [0 if v is None else v for v in values], color=colors)
        ax.axhline(0, color="#444444", linewidth=1)
        ax.set_xticks(range(len(values)), labels, rotation=55, ha="right", fontsize=7)
        ax.set_ylabel("no-retrain minus retrain avg latency (ms)")
        ax.set_title("No-Retrain Selected Feasible Latency Delta")
        fig.tight_layout()
        written += save(fig, figures, "no_retrain_selected_feasible_latency_delta")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.0, 4.5))
        matched_values = [r.get("matched_reference_recall_delta_ms") for r in penalty]
        colors = ["#999999" if v is None else ("#2878b5" if v <= 0 else "#c85200") for v in matched_values]
        ax.bar(range(len(matched_values)), [0 if v is None else v for v in matched_values], color=colors)
        ax.axhline(0, color="#444444", linewidth=1)
        ax.set_xticks(range(len(matched_values)), labels, rotation=55, ha="right", fontsize=7)
        ax.set_ylabel("no-retrain matched-reference delta (ms)")
        ax.set_title("No-Retrain Matched Reference Recall Delta")
        fig.tight_layout()
        written += save(fig, figures, "no_retrain_matched_reference_recall_delta")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.0, 4.5))
        values = [r.get("l_uplift") for r in penalty]
        ax.bar(range(len(values)), [0 if v is None else v for v in values], color="#2878b5")
        ax.axhline(0, color="#444444", linewidth=1)
        ax.set_xticks(range(len(values)), labels, rotation=55, ha="right", fontsize=7)
        ax.set_ylabel("no-retrain L minus retrain L")
        ax.set_title("Selected L Uplift")
        fig.tight_layout()
        written += save(fig, figures, "selected_l_uplift")
        plt.close(fig)

    core = read_jsonl(root / "raw/phaseD_pq_core_sweep.jsonl")
    if core:
        core = sorted(core, key=lambda r: r.get("core_count", 0))
        fig, ax = plt.subplots(figsize=(6.8, 4.3))
        ax.plot([r["core_count"] for r in core], [r.get("pq_train_wall_s") or 0 for r in core], marker="o", label="PQ train")
        ax.plot([r["core_count"] for r in core], [r.get("pq_recode_wall_s") or 0 for r in core], marker="o", label="PQ recode")
        ax.plot([r["core_count"] for r in core], [r.get("build_wall_s") or 0 for r in core], marker="o", label="build wall")
        ax.set_xlabel("CPU cores")
        ax.set_ylabel("seconds")
        ax.set_title("PQ Retrain/Recode Core Sweep")
        ax.legend()
        fig.tight_layout()
        written += save(fig, figures, "phaseD_pq_core_sweep")
        plt.close(fig)

    (root / "figures_manifest.json").write_text(json.dumps({"figures": written}, indent=2) + "\n", encoding="utf-8")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    written = plot(args.root)
    print(json.dumps({"figures": written}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
