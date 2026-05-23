#!/usr/bin/env python3
"""Summarize existing dynamic delete / PQ drift ARIS evidence.

This script is read-only with respect to evidence roots. It writes a markdown
summary and lightweight figures into a new summary directory.
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def has_fields(row: dict[str, Any], fields: list[str]) -> bool:
    return bool(row) and all(field in row and row[field] is not None for field in fields)


def pass_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if row.get("supports_recall_claim") is True)


def evidence_roots(repo: Path) -> dict[str, Path]:
    exp = repo / "experiments"
    return {
        "phase12": exp / "dynamic_delete_pq_drift_aris_20260522_phase12_rerun",
        "phase3_5cycle": exp / "dynamic_delete_pq_drift_aris_20260522_phase3_5cycle_bigann6m",
        "phase4_flatfinal": exp / "dynamic_delete_pq_drift_aris_20260522_phase4_flatfinal",
    }


def ensure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def write_existing_figures(out_dir: Path, roots: dict[str, Path]) -> list[Path]:
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt = ensure_matplotlib()
    written: list[Path] = []

    phase12 = roots["phase12"]
    p1 = read_jsonl(phase12 / "raw/phase1_delete_only.jsonl")
    p2 = read_jsonl(phase12 / "raw/phase2_delete_then_merge.jsonl")
    if p1 and p2:
        delete_ms = p1[0]["delete_wall_s"] * 1000.0 / p1[0]["delete_count"]
        labels = ["delete API\nms/vector", "merge\nseconds"]
        values = [delete_ms, p2[0]["merge_wall_s"]]
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        bars = ax.bar(labels, values, color=["#2878b5", "#c85200"])
        ax.set_title("Phase1/2 Delete API and Merge Cost")
        ax.set_ylabel("mixed units")
        ax.bar_label(bars, fmt="%.4g")
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            path = figures / f"phase12_delete_merge.{suffix}"
            fig.savefig(path, dpi=180)
            written.append(path)
        plt.close(fig)

    phase3 = roots["phase3_5cycle"]
    deletes = read_jsonl(phase3 / "raw/phase3_delete_steps.jsonl")
    inserts = read_jsonl(phase3 / "raw/phase3_cycles.jsonl")
    selected = read_jsonl(phase3 / "raw/selected_route_l.jsonl")
    if deletes and inserts:
        cycles = list(range(1, max(len(deletes), len(inserts)) + 1))
        fig, ax = plt.subplots(figsize=(7.4, 4.4))
        ax.plot(cycles[: len(deletes)], [d["delete_elapsed_s"] for d in deletes], marker="o", label="delete API")
        ax.plot(cycles[: len(deletes)], [d["merge_elapsed_s"] for d in deletes], marker="o", label="delete-side merge")
        ax.plot(cycles[: len(inserts)], [d["insert_elapsed_s"] for d in inserts], marker="o", label="insert")
        ax.plot(cycles[: len(inserts)], [d["merge_elapsed_s"] for d in inserts], marker="o", label="insert-side merge")
        ax.set_title("Phase3 5-Cycle Costs")
        ax.set_xlabel("cycle")
        ax.set_ylabel("seconds")
        ax.legend()
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            path = figures / f"phase3_5cycle_costs.{suffix}"
            fig.savefig(path, dpi=180)
            written.append(path)
        plt.close(fig)
    if selected:
        fig, ax1 = plt.subplots(figsize=(8.0, 4.6))
        x = list(range(1, len(selected) + 1))
        ax1.plot(x, [r["recall@10"] for r in selected], marker="o", label="recall@10", color="#2878b5")
        ax1.axhline(98.0, color="#666666", linestyle="--", linewidth=1)
        ax1.set_ylabel("recall@10 (%)")
        ax1.set_xlabel("selected point order")
        ax2 = ax1.twinx()
        ax2.plot(x, [r["avg_latency_us"] / 1000.0 for r in selected], marker="s", label="avg latency", color="#c85200")
        ax2.set_ylabel("avg latency (ms)")
        ax1.set_title("Phase3 Selected Recall and Latency")
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            path = figures / f"phase3_selected_recall_latency.{suffix}"
            fig.savefig(path, dpi=180)
            written.append(path)
        plt.close(fig)

    phase4 = roots["phase4_flatfinal"]
    drift = read_jsonl(phase4 / "raw/phase4_pq_drift.jsonl")
    phase4_selected = read_jsonl(phase4 / "raw/phase4_selected_route_l.jsonl")
    if drift:
        fig, ax = plt.subplots(figsize=(7.4, 4.2))
        labels = [d["variant"] for d in drift]
        train = [d.get("pq_train_wall_s") or 0 for d in drift]
        recode = [d.get("pq_recode_wall_s") or 0 for d in drift]
        x = range(len(labels))
        ax.bar(x, train, label="PQ train", color="#2878b5")
        ax.bar(x, recode, bottom=train, label="PQ recode", color="#c85200")
        ax.set_xticks(list(x), labels, rotation=12, ha="right")
        ax.set_ylabel("seconds")
        ax.set_title("Phase4 100k PQ Train/Recode Smoke")
        ax.legend()
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            path = figures / f"phase4_pq_train_recode.{suffix}"
            fig.savefig(path, dpi=180)
            written.append(path)
        plt.close(fig)
    if phase4_selected:
        rows = [r["selected"] for r in phase4_selected if "selected" in r]
        fig, ax = plt.subplots(figsize=(7.8, 4.4))
        labels = [f"{r['cycle'].replace('phase4_', '')}\n{r['selector_type']}-{r['bucket']}" for r in rows]
        ax.bar(range(len(rows)), [r["avg_latency_us"] / 1000.0 for r in rows], color="#2878b5")
        ax.set_xticks(range(len(rows)), labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("avg latency (ms)")
        ax.set_title("Phase4 100k Selected Latency")
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            path = figures / f"phase4_selected_latency.{suffix}"
            fig.savefig(path, dpi=180)
            written.append(path)
        plt.close(fig)

    return written


def summarize(repo: Path, out_dir: Path) -> None:
    roots = evidence_roots(repo)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = write_existing_figures(out_dir, roots)

    phase12_p1 = read_jsonl(roots["phase12"] / "raw/phase1_delete_only.jsonl")
    phase12_p2 = read_jsonl(roots["phase12"] / "raw/phase2_delete_then_merge.jsonl")
    phase3_deletes = read_jsonl(roots["phase3_5cycle"] / "raw/phase3_delete_steps.jsonl")
    phase3_inserts = read_jsonl(roots["phase3_5cycle"] / "raw/phase3_cycles.jsonl")
    phase3_selected = read_jsonl(roots["phase3_5cycle"] / "raw/selected_route_l.jsonl")
    phase4_zero = read_jsonl(roots["phase4_flatfinal"] / "raw/phase4_zero_insert.jsonl")
    phase4_drift = read_jsonl(roots["phase4_flatfinal"] / "raw/phase4_pq_drift.jsonl")
    phase4_selected = [r["selected"] for r in read_jsonl(roots["phase4_flatfinal"] / "raw/phase4_selected_route_l.jsonl") if "selected" in r]

    phase3_delete_ms = [d["delete_elapsed_s"] * 1000.0 / d["delete_count"] for d in phase3_deletes]
    phase3_min_recall = min((r.get("recall@10", 0) for r in phase3_selected), default=None)
    phase3_max_avg_ms = max((r.get("avg_latency_us", 0) / 1000.0 for r in phase3_selected), default=None)
    phase3_max_p95_ms = max((r.get("p95_latency_us", 0) / 1000.0 for r in phase3_selected), default=None)
    phase4_min_recall = min((r.get("recall@10", 0) for r in phase4_selected), default=None)
    phase3_pass = pass_count(phase3_selected)
    phase4_pass = pass_count(phase4_selected)

    lines = [
        "# Dynamic Delete / Merge / PQ Drift ARIS Summary",
        "",
        f"Created UTC: `{now_stamp()}`",
        "",
        "## Evidence Roots",
        "",
    ]
    for name, path in roots.items():
        lines.append(f"- `{name}`: `{path}`")
    lines += [
        "",
        "## Claim-by-Claim Status",
        "",
        "| # | Claim | Status | Evidence-backed summary | Do not overclaim |",
        "|---|---|---|---|---|",
    ]
    p1 = phase12_p1[0] if phase12_p1 else {}
    p2 = phase12_p2[0] if phase12_p2 else {}
    delete_ms = p1.get("delete_wall_s", 0) * 1000.0 / p1.get("delete_count", 1) if p1 else None
    claim1_status = "PASS" if has_fields(p1, ["delete_count", "delete_wall_s"]) else "UNSUPPORTED"
    claim2_status = "WARN" if has_fields(p2, ["merge_wall_s", "cpu_cap", "cpu_cap_enforced"]) else "UNSUPPORTED"
    claim3_status = (
        "WARN"
        if phase3_deletes
        and phase3_inserts
        and len(phase3_deletes) == len(phase3_inserts)
        and phase3_selected
        else "UNSUPPORTED"
    )
    claim4_status = "WARN" if phase4_zero and phase4_drift and phase4_selected else "UNSUPPORTED"
    claim5_status = "PASS" if phase3_selected or phase4_selected else "UNSUPPORTED"
    claim6_status = (
        "WARN"
        if has_fields(p2, ["main_index_label_size", "label_sidecar_loadable"])
        and p2.get("main_index_label_size") == 0
        else "UNSUPPORTED"
    )

    lines.append(
        "| 1 | Current delete is mark/lazy delete; 60% delete is sub-ms/vector | PASS | "
        f"`delete_count={p1.get('delete_count')}`, `delete_wall_s={fmt(p1.get('delete_wall_s'), 6)}`, "
        f"`avg={fmt(delete_ms, 6)} ms/vector`. | Claim registry C1 is still not a full paper-grade static proof. |"
    .replace("| PASS |", f"| {claim1_status} |", 1))
    lines.append(
        f"| 2 | Mark-delete materialization/merge time under resource cap | {claim2_status} | "
        f"`merge_wall_s={fmt(p2.get('merge_wall_s'), 3)}`, `wall_s={fmt(p2.get('wall_s'), 3)}`, "
        f"`cpu_cap={p2.get('cpu_cap')}`, `allowed_cpus={p2.get('cpu_affinity_allowed_cpus')}`. | "
        "No watt/power measurement; state CPU cap only. |"
    )
    lines.append(
        f"| 3 | Repeated 60% delete + equal insert | {claim3_status} | "
        f"`cycles={len(phase3_deletes)}`, delete live counts `{[d.get('live_point_count') for d in phase3_deletes]}`, "
        f"insert live counts `{[d.get('live_point_count') for d in phase3_inserts]}`; "
        f"selected min recall `{fmt(phase3_min_recall, 2)}`, max avg latency `{fmt(phase3_max_avg_ms, 2)} ms`. | "
        "This is a 1M-live BigANN/SIFT-6M-prefix experiment, not full SIFT100M. |"
    )
    zero = phase4_zero[0] if phase4_zero else {}
    drift_by_variant = {r.get("variant"): r for r in phase4_drift}
    direct = drift_by_variant.get("direct_build", {})
    no_retrain = drift_by_variant.get("zero_insert_seed_pq_no_retrain", {})
    lines.append(
        f"| 4 | PQ drift from zero insert | {claim4_status} | "
        f"100k smoke: direct train `{fmt(direct.get('pq_train_wall_s'), 3)}s`, recode `{fmt(direct.get('pq_recode_wall_s'), 3)}s`; "
        f"zero insert `{fmt(zero.get('insert_wall_s'), 3)}s`, merge `{fmt(zero.get('merge_wall_s'), 3)}s`, "
        f"zero min recall `{fmt(phase4_min_recall, 2)}`. | "
        "100k flat-until-final smoke only; not 1M or long online insertion. |"
    )
    lines.append(
        f"| 5 | Recall means retuned search parameters can reach 98% | {claim5_status} | "
        f"Selected rows use `post_hoc_retuned_fastest_feasible_recall_ge_98`; "
        f"Phase3 has `{phase3_pass}/{len(phase3_selected)}` selected points passing, "
        f"Phase4 has `{phase4_pass}/{len(phase4_selected)}`. | "
        "Do not claim fixed-parameter graph quality is unchanged or held-out generalization. |"
    )
    lines.append(
        f"| 6 | Labels stored in sidecar, main index label payload removed | {claim6_status} | "
        f"Phase2 has `main_index_label_size={p2.get('main_index_label_size')}`, "
        f"`label_sidecar_loadable={p2.get('label_sidecar_loadable')}`. | "
        "Current evidence is sidecar smoke with disk format v1; full v2 format remains unsupported. |"
    )

    lines += [
        "",
        "## Key Metrics",
        "",
        f"- Phase3 delete API ms/vector: `{[round(x, 6) for x in phase3_delete_ms]}`; average `{fmt(statistics.mean(phase3_delete_ms) if phase3_delete_ms else None, 6)}`.",
        f"- Phase3 delete-side merge seconds: `{[round(d.get('merge_elapsed_s', 0), 3) for d in phase3_deletes]}`.",
        f"- Phase3 insert seconds: `{[round(d.get('insert_elapsed_s', 0), 3) for d in phase3_inserts]}`.",
        f"- Phase3 insert-side merge seconds: `{[round(d.get('merge_elapsed_s', 0), 3) for d in phase3_inserts]}`.",
        f"- Phase3 selected recall min/max: `{fmt(phase3_min_recall, 2)}` / `{fmt(max((r.get('recall@10', 0) for r in phase3_selected), default=None), 2)}`.",
        f"- Phase3 selected latency max avg/p95: `{fmt(phase3_max_avg_ms, 3)} ms` / `{fmt(phase3_max_p95_ms, 3)} ms`.",
        "",
        "## Figures",
        "",
    ]
    for figure in figures:
        lines.append(f"- `{figure.relative_to(out_dir)}`")
    lines += [
        "",
        "## ARIS Review Position",
        "",
        "- Existing evidence supports smoke/local claims with raw JSONL and logs.",
        "- New 1M PQ drift work must create a fresh claim registry and bind direct commands, input hashes, pivot hashes, raw calibration rows, and review reports.",
        "- Claims without frozen evidence should be labelled `UNSUPPORTED`; claims with 100k-only evidence should remain `SMOKE`.",
        "",
    ]
    (out_dir / "previous_results_summary.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "summary_manifest.json").write_text(
        json.dumps(
            {
                "created_utc": now_stamp(),
                "repo": str(repo),
                "evidence_roots": {k: str(v) for k, v in roots.items()},
                "figures": [str(p) for p in figures],
                "output": str(out_dir / "previous_results_summary.md"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/mnt/bak3/lzg/PipeANN-github"))
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    out_dir = args.out_dir or args.repo / "experiments" / f"dynamic_delete_pq_drift_aris_summary_{now_stamp()}"
    summarize(args.repo, out_dir)
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
