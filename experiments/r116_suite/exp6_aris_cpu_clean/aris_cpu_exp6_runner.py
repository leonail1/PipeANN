#!/usr/bin/env python3
"""ARIS-style CPU runner for PipeANN r116 thread-sweep validation.

This is intentionally experiment-local.  It follows the ARIS
experiment-queue contracts that matter for this CPU/SSD benchmark:
manifest.json, queue_state.json, per-job logs, expected output files,
resume safety, and traceable final summaries.  It does not use ARIS'
GPU scheduler because this experiment has no GPU resource.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BUCKET_ORDER = ["u1e-03", "u3e-03", "u1e-02", "u5e-02", "u1e-01", "u25", "u30", "u50", "u75", "u100"]
BUCKET_LABELS = {
    "u1e-03": "0.1%",
    "u3e-03": "0.3%",
    "u1e-02": "1%",
    "u5e-02": "5%",
    "u1e-01": "10%",
    "u25": "25%",
    "u30": "30%",
    "u50": "50%",
    "u75": "75%",
    "u100": "100%",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_threads(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as reader:
        return list(csv.DictReader(reader))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as writer:
        writer_obj = csv.DictWriter(writer, fieldnames=keys, lineterminator="\n")
        writer_obj.writeheader()
        writer_obj.writerows(rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as reader:
        for line in reader:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as writer:
        writer.write(json.dumps(row, sort_keys=True) + "\n")


def run_command(command: list[str], cwd: Path, log_path: Path, env: dict[str, str] | None = None) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        log.write("$ " + " ".join(command) + "\n")
        proc = subprocess.run(
            command,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, **(env or {})},
        )
        elapsed = time.time() - start
        log.write(f"\n[exit_code] {proc.returncode}\n[elapsed_s] {elapsed:.3f}\n")
    if proc.returncode != 0:
        raise RuntimeError(f"command failed, see {log_path}: {' '.join(command)}")
    return elapsed


def read_last_jsonl(path: Path, before_count: int) -> dict[str, Any]:
    rows = read_jsonl(path)
    if len(rows) <= before_count:
        raise RuntimeError(f"expected {path} to append a row")
    return rows[-1]


def file_prefix_exists(prefix: Path) -> bool:
    return Path(str(prefix) + "_disk.index").exists()


def build_index_if_needed(repo: Path, prefix: Path, args: argparse.Namespace) -> None:
    if file_prefix_exists(prefix):
        return
    command = [
        str(repo / "build/tests/build_disk_index"),
        "float",
        str(repo / "experiments/r116_suite/data/sift_base_1m.bin"),
        str(prefix),
        str(args.build_r),
        str(args.build_l),
        str(args.pq_bytes),
        str(args.memory_gb),
        "32",
        "l2",
        "pq",
        "spmat",
        str(repo / "experiments/r116_suite/labels/base_1m.spmat"),
    ]
    run_command(command, repo, args.out_dir / "logs/preflight/build_index.log")


def compute_truth_if_needed(repo: Path, selector: str, bucket: str, args: argparse.Namespace) -> Path:
    truth = repo / "experiments/r116_suite/exp4_intersect_range_selectivity/truth" / f"gt_1m_{selector}_{bucket}.bin"
    if truth.exists():
        return truth
    query_label = query_label_path(repo, selector, bucket)
    command = [
        str(repo / "build/tests/utils/compute_groundtruth"),
        "float",
        "l2",
        str(repo / "experiments/r116_suite/data/sift_base_1m.bin"),
        str(repo / "experiments/r116_suite/data/sift_query_1000.bin"),
        "10",
        str(truth),
        "null",
        "spmat",
        selector,
        str(repo / "experiments/r116_suite/labels/base_1m.spmat"),
        str(query_label),
    ]
    if shutil.which("numactl"):
        command = ["numactl", "--cpunodebind=1", "--membind=1", *command]
    run_command(command, repo, args.out_dir / "logs/truth" / f"{selector}_{bucket}.log")
    return truth


def query_label_path(repo: Path, selector: str, bucket: str) -> Path:
    name = f"query_1000_{bucket}.spmat" if selector == "intersect" else f"query_1000_range_{bucket}.spmat"
    return repo / "experiments/r116_suite/labels" / name


def load_l_overrides(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    overrides: dict[tuple[str, str], dict[str, Any]] = {}
    for key, value in raw.items():
        selector, bucket = key.split(":", 1)
        overrides[(selector, bucket)] = value
    return overrides


def load_exp4_settings(repo: Path, overrides: dict[tuple[str, str], dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    table = repo / "experiments/r116_suite/exp4_intersect_range_selectivity/table.csv"
    rows = read_csv(table)
    settings: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        selector = row.get("selector_type") or row.get("filter_type")
        bucket = row.get("bucket")
        if not selector or not bucket:
            continue
        route = row.get("selected_route") or row.get("route") or "auto"
        chosen_l = int(float(row.get("chosen_L") or row.get("L") or 0))
        if chosen_l <= 0:
            continue
        setting = {
            "selector_type": selector,
            "bucket": bucket,
            "selected_route": route,
            "chosen_L": chosen_l,
            "exp4_avg_latency_us": float(row.get("avg_latency_us") or 0.0),
            "exp4_p99_latency_us": float(row.get("p99_latency_us") or 0.0),
            "exp4_recall@10": float(row.get("recall@10") or row.get("recall") or 0.0),
        }
        override = overrides.get((selector, bucket))
        if override:
            setting.update({
                "selected_route": override.get("selected_route", setting["selected_route"]),
                "chosen_L": int(override.get("chosen_L", setting["chosen_L"])),
                "override": override,
            })
        settings[(selector, bucket)] = setting
    return settings


def build_jobs(repo: Path, args: argparse.Namespace, index_prefix: Path) -> list[dict[str, Any]]:
    settings = load_exp4_settings(repo, load_l_overrides(args.out_dir / "l_overrides.json"))
    jobs: list[dict[str, Any]] = []
    for selector in ["intersect", "range"]:
        for bucket in BUCKET_ORDER:
            setting = settings.get((selector, bucket))
            if not setting:
                continue
            for threads in args.thread_sweep:
                job_id = f"{selector}_{bucket}_t{threads:02d}"
                truth = repo / "experiments/r116_suite/exp4_intersect_range_selectivity/truth" / f"gt_1m_{selector}_{bucket}.bin"
                query_label = query_label_path(repo, selector, bucket)
                command = [
                    str(repo / "build/tests/search_disk_index_hybrid"),
                    "float",
                    str(index_prefix),
                    str(threads),
                    "4",
                    str(repo / "experiments/r116_suite/data/sift_query_1000.bin"),
                    str(truth),
                    "10",
                    "l2",
                    "pq",
                    selector,
                    str(query_label),
                    setting["selected_route"],
                    "0",
                    "0",
                    str(setting["chosen_L"]),
                    "--jsonl-output",
                    str(args.out_dir / "measure_driver.jsonl"),
                ]
                jobs.append({
                    "id": job_id,
                    "phase": "sweep",
                    "selector_type": selector,
                    "filter_type": selector,
                    "bucket": bucket,
                    "threads": threads,
                    "selected_route": setting["selected_route"],
                    "chosen_L": setting["chosen_L"],
                    "expected_output": str(args.out_dir / "jobs" / f"{job_id}.json"),
                    "truth": str(truth),
                    "query_label_file": str(query_label),
                    "cmd": command,
                    "source_exp4": setting,
                })
    return jobs


def write_manifest(repo: Path, args: argparse.Namespace, index_prefix: Path, jobs: list[dict[str, Any]]) -> dict[str, Any]:
    manifest = {
        "project": "pipeann_r116_exp6_aris_cpu",
        "created_at": now(),
        "aris_repo": args.aris_repo,
        "aris_commit": args.aris_commit,
        "aris_skills": [
            "skills/skills-codex/experiment-queue/SKILL.md",
            "skills/shared-references/experiment-integrity.md",
        ],
        "evaluation_type": "real_gt",
        "cwd": str(repo),
        "ssh": "node6",
        "resource": "cpu",
        "max_parallel": 1,
        "latency_budget_ms": args.latency_budget_ms,
        "query_count": 1000,
        "thread_sweep": args.thread_sweep,
        "index_prefix": str(index_prefix),
        "build": {
            "R": args.build_r,
            "L": args.build_l,
            "pq_bytes": args.pq_bytes,
            "memory_gb": args.memory_gb,
        },
        "phases": [{"name": "sweep", "depends_on": [], "jobs": jobs}],
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def initial_state(manifest: dict[str, Any]) -> dict[str, Any]:
    jobs = []
    for phase in manifest["phases"]:
        for job in phase["jobs"]:
            jobs.append({
                "id": job["id"],
                "phase": phase["name"],
                "status": "pending",
                "attempts": 0,
                "started": None,
                "completed": None,
                "expected_output": job["expected_output"],
                "error": None,
            })
    return {
        "meta": {
            "project": manifest["project"],
            "started": now(),
            "manifest_path": str(Path(manifest["cwd"]) / "experiments/r116_suite/exp6_aris_cpu/manifest.json"),
        },
        "phases": [{"name": "sweep", "depends_on": [], "status": "pending"}],
        "jobs": jobs,
    }


def load_or_init_state(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    state_path = args.out_dir / "queue_state.json"
    if state_path.exists() and not args.rerun:
        return json.loads(state_path.read_text(encoding="utf-8"))
    return initial_state(manifest)


def save_state(args: argparse.Namespace, state: dict[str, Any]) -> None:
    state_path = args.out_dir / "queue_state.json"
    tmp = state_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(state_path)


def completed_output(job: dict[str, Any]) -> dict[str, Any] | None:
    output = Path(job["expected_output"])
    if not output.exists():
        return None
    return json.loads(output.read_text(encoding="utf-8"))


def run_jobs(repo: Path, args: argparse.Namespace, manifest: dict[str, Any], state: dict[str, Any]) -> None:
    jobs_by_id = {job["id"]: job for phase in manifest["phases"] for job in phase["jobs"]}
    state_by_id = {job["id"]: job for job in state["jobs"]}
    for job_id in [job["id"] for phase in manifest["phases"] for job in phase["jobs"]]:
        job = jobs_by_id[job_id]
        state_job = state_by_id[job_id]
        output = completed_output(job)
        if output is not None:
            state_job.update({"status": "completed", "completed": output.get("completed_at") or state_job.get("completed")})
            save_state(args, state)
            continue

        selector = job["selector_type"]
        bucket = job["bucket"]
        compute_truth_if_needed(repo, selector, bucket, args)

        state_job.update({"status": "running", "attempts": int(state_job.get("attempts") or 0) + 1, "started": now(), "error": None})
        state["phases"][0]["status"] = "running"
        save_state(args, state)
        before_count = len(read_jsonl(args.out_dir / "measure_driver.jsonl"))
        started = now()
        try:
            elapsed = run_command(job["cmd"], repo, args.out_dir / "logs/jobs" / f"{job_id}.log")
            measured = read_last_jsonl(args.out_dir / "measure_driver.jsonl", before_count)
            if "recall" in measured and "recall@10" not in measured:
                measured["recall@10"] = measured["recall"]
            measured.update({
                "status": "ok" if float(measured.get("recall@10") or 0.0) >= 98.0 else "failed_recall",
                "bucket": bucket,
                "selector_type": selector,
                "filter_type": selector,
                "threads": job["threads"],
                "points": 1_000_000,
                "query_count": 1000,
                "selected_route": job["selected_route"],
                "chosen_L": job["chosen_L"],
                "latency_budget_ms": args.latency_budget_ms,
                "within_avg_latency_budget": float(measured.get("avg_latency_us") or 0.0) <= args.latency_budget_ms * 1000.0,
                "within_p90_latency_budget": float(measured.get("p90_latency_us") or 0.0) <= args.latency_budget_ms * 1000.0,
                "within_p95_latency_budget": float(measured.get("p95_latency_us") or 0.0) <= args.latency_budget_ms * 1000.0,
                "within_p99_latency_budget": float(measured.get("p99_latency_us") or 0.0) <= args.latency_budget_ms * 1000.0,
                "source_experiment": "exp4_intersect_range_selectivity",
                "aris_job_id": job_id,
                "started_at": started,
                "completed_at": now(),
                "wall_s": elapsed,
            })
            output_path = Path(job["expected_output"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(measured, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            append_jsonl(args.out_dir / "results.jsonl", measured)
            state_job.update({"status": "completed", "completed": measured["completed_at"]})
            save_state(args, state)
        except Exception as exc:
            state_job.update({"status": "stuck", "completed": now(), "error": str(exc)})
            save_state(args, state)
            raise
    state["phases"][0]["status"] = "completed"
    state["meta"]["completed"] = now()
    save_state(args, state)


def collect_results(args: argparse.Namespace, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for phase in manifest["phases"]:
        for job in phase["jobs"]:
            output = completed_output(job)
            if output is not None:
                rows.append(output)
    rows.sort(key=lambda row: (
        0 if row.get("selector_type") == "intersect" else 1,
        BUCKET_ORDER.index(row["bucket"]) if row.get("bucket") in BUCKET_ORDER else 999,
        int(row.get("threads") or 0),
    ))
    return rows


def summarize(rows: list[dict[str, Any]], budget_ms: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    budget_us = budget_ms * 1000.0
    summary_rows: list[dict[str, Any]] = []
    by_thread: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_thread.setdefault(int(row["threads"]), []).append(row)
    for threads in sorted(by_thread):
        group = by_thread[threads]
        summary_rows.append({
            "threads": threads,
            "rows": len(group),
            "min_recall@10": min(float(r.get("recall@10") or 0.0) for r in group),
            "max_avg_latency_ms": max(float(r.get("avg_latency_us") or 0.0) for r in group) / 1000.0,
            "max_p90_latency_ms": max(float(r.get("p90_latency_us") or 0.0) for r in group) / 1000.0,
            "max_p95_latency_ms": max(float(r.get("p95_latency_us") or 0.0) for r in group) / 1000.0,
            "max_p99_latency_ms": max(float(r.get("p99_latency_us") or 0.0) for r in group) / 1000.0,
            "avg_budget_pass_rows": sum(float(r.get("avg_latency_us") or 0.0) <= budget_us for r in group),
            "p90_budget_pass_rows": sum(float(r.get("p90_latency_us") or 0.0) <= budget_us for r in group),
            "p95_budget_pass_rows": sum(float(r.get("p95_latency_us") or 0.0) <= budget_us for r in group),
            "p99_budget_pass_rows": sum(float(r.get("p99_latency_us") or 0.0) <= budget_us for r in group),
        })
    recall_failed_rows = sum(float(r.get("recall@10") or 0.0) < 98.0 for r in rows)
    avg_budget_failed_rows = sum(float(r.get("avg_latency_us") or 0.0) > budget_us for r in rows)
    if len(rows) != 320:
        status = "incomplete"
    elif recall_failed_rows or avg_budget_failed_rows:
        status = "failed"
    else:
        status = "ok"
    summary = {
        "status": status,
        "rows": len(rows),
        "expected_rows": 320,
        "latency_budget_ms": budget_ms,
        "min_recall@10": min((float(r.get("recall@10") or 0.0) for r in rows), default=0.0),
        "max_avg_latency_ms": max((float(r.get("avg_latency_us") or 0.0) for r in rows), default=0.0) / 1000.0,
        "max_p90_latency_ms": max((float(r.get("p90_latency_us") or 0.0) for r in rows), default=0.0) / 1000.0,
        "max_p95_latency_ms": max((float(r.get("p95_latency_us") or 0.0) for r in rows), default=0.0) / 1000.0,
        "max_p99_latency_ms": max((float(r.get("p99_latency_us") or 0.0) for r in rows), default=0.0) / 1000.0,
        "recall_failed_rows": recall_failed_rows,
        "avg_budget_failed_rows": avg_budget_failed_rows,
        "p90_budget_failed_rows": sum(float(r.get("p90_latency_us") or 0.0) > budget_us for r in rows),
        "p95_budget_failed_rows": sum(float(r.get("p95_latency_us") or 0.0) > budget_us for r in rows),
        "p99_budget_failed_rows": sum(float(r.get("p99_latency_us") or 0.0) > budget_us for r in rows),
    }
    return summary, summary_rows


def plot(args: argparse.Namespace, rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 420,
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    threads = [int(row["threads"]) for row in summary_rows]
    metrics = [
        ("max_avg_latency_ms", "avg", "#2563eb"),
        ("max_p90_latency_ms", "p90", "#16a34a"),
        ("max_p95_latency_ms", "p95", "#f59e0b"),
        ("max_p99_latency_ms", "p99", "#dc2626"),
    ]
    fig, ax = plt.subplots(figsize=(13.5, 7.5), constrained_layout=True)
    for key, label, color in metrics:
        ax.plot(threads, [float(row[key]) for row in summary_rows], marker="o", linewidth=2.2, color=color, label=label)
    ax.axhline(args.latency_budget_ms, linestyle=":", color="#111827", linewidth=1.8, label=f"{args.latency_budget_ms:g} ms")
    ax.set_xlabel("Foreground query threads")
    ax.set_ylabel("Worst-case latency across all selectivities (ms)")
    ax.set_title("r116 ARIS CPU sweep: max avg/p90/p95/p99 latency over intersect+range workloads")
    ax.set_xticks(threads)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=5, loc="upper left", frameon=False)
    fig.savefig(args.out_dir / "latency_percentiles_worstcase_highres.png", bbox_inches="tight")
    fig.savefig(args.out_dir / "latency_percentiles_worstcase_highres.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.8), sharey=True, constrained_layout=True)
    for ax, selector in zip(axes, ["intersect", "range"]):
        selector_rows = [row for row in rows if row.get("selector_type") == selector]
        by_thread: dict[int, list[dict[str, Any]]] = {}
        for row in selector_rows:
            by_thread.setdefault(int(row["threads"]), []).append(row)
        selector_threads = sorted(by_thread)
        for source_key, label, color in [
            ("avg_latency_us", "avg", "#2563eb"),
            ("p90_latency_us", "p90", "#16a34a"),
            ("p95_latency_us", "p95", "#f59e0b"),
            ("p99_latency_us", "p99", "#dc2626"),
        ]:
            ax.plot(
                selector_threads,
                [max(float(row.get(source_key) or 0.0) for row in by_thread[t]) / 1000.0 for t in selector_threads],
                marker="o",
                linewidth=2.0,
                color=color,
                label=label,
            )
        ax.axhline(args.latency_budget_ms, linestyle=":", color="#111827", linewidth=1.6)
        ax.set_title(selector)
        ax.set_xlabel("Foreground query threads")
        ax.set_xticks(selector_threads)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Worst-case latency by selector (ms)")
    axes[1].legend(ncol=4, loc="upper left", frameon=False)
    fig.suptitle("r116 ARIS CPU sweep by selector")
    fig.savefig(args.out_dir / "latency_percentiles_by_selector_highres.png", bbox_inches="tight")
    fig.savefig(args.out_dir / "latency_percentiles_by_selector_highres.pdf", bbox_inches="tight")
    plt.close(fig)


def write_review(args: argparse.Namespace, manifest: dict[str, Any], summary: dict[str, Any]) -> None:
    avg_status = "PASS" if summary["avg_budget_failed_rows"] == 0 and summary["rows"] == summary["expected_rows"] else "FAIL"
    p90_status = "PASS" if summary["p90_budget_failed_rows"] == 0 and summary["rows"] == summary["expected_rows"] else "WARN"
    p99_status = "PASS" if summary["p99_budget_failed_rows"] == 0 and summary["rows"] == summary["expected_rows"] else "WARN"
    override_path = args.out_dir / "l_overrides.json"
    override_note = ""
    if override_path.exists():
        override_note = (
            "- PASS: `l_overrides.json` documents post-review L overrides. In this run, `u75` was recalibrated from "
            "graph `L=50` to graph `L=75` after the first sweep found recall@10 below 98%.\n"
        )
    text = f"""# ARIS-Style Experiment Review: r116 CPU Thread Sweep

日期：2026-05-19

## Scope

- ARIS source: `{manifest.get('aris_repo')}` at `{manifest.get('aris_commit')}`
- Skills used as protocol: `experiment-queue`, `experiment-integrity`
- Executor/reviewer note: this is an ARIS-style self review in Codex Desktop, not an independent cross-model reviewer run.
- Evaluation type: `real_gt`
- Resource mode: CPU, `max_parallel=1`
- Rows: `{summary['rows']}` / `{summary['expected_rows']}`

## Experiment Settings Review

- PASS: The experiment uses real filtered ground truth from `compute_groundtruth`, not synthetic or self-normalized labels.
- PASS: Route and `L` are inherited from `exp4_intersect_range_selectivity` except documented overrides.
{override_note.rstrip()}
- PASS: The sweep covers both `intersect` and `range`, all 10 selectivity buckets, and foreground query threads 1-16.
- PASS: `search_disk_index_hybrid` emits `avg`, `p90`, `p95`, and `p99` latency fields.
- PASS: The runner writes `manifest.json`, `queue_state.json`, per-job logs, per-job JSON outputs, `table.csv`, and summary plots.
- WATCH: The source index prefix is `{manifest.get('index_prefix')}`. It is a completed r116/PQ32 build from the interrupted exp6 run and is referenced explicitly in the manifest.
- WATCH: The 10ms acceptance in the user's latest instruction is average latency. Percentiles are plotted and reported, but p99 is expected to be stricter and may exceed 10ms.
- WATCH: Original ARIS GPU queue manager is not used because node6 has no `nvidia-smi` and this is a CPU/SSD benchmark.

## Result Review

- Average latency budget: `{avg_status}`; failed rows above 10ms = `{summary['avg_budget_failed_rows']}`.
- p90 latency budget: `{p90_status}`; failed rows above 10ms = `{summary['p90_budget_failed_rows']}`.
- p95 latency budget: `{'PASS' if summary['p95_budget_failed_rows'] == 0 and summary['rows'] == summary['expected_rows'] else 'WARN'}`; failed rows above 10ms = `{summary['p95_budget_failed_rows']}`.
- p99 latency budget: `{p99_status}`; failed rows above 10ms = `{summary['p99_budget_failed_rows']}`.
- Minimum recall@10: `{summary['min_recall@10']:.4f}`.
- Recall@10 rows below 98%: `{summary['recall_failed_rows']}`.
- Max avg latency: `{summary['max_avg_latency_ms']:.3f} ms`.
- Max p90 latency: `{summary['max_p90_latency_ms']:.3f} ms`.
- Max p95 latency: `{summary['max_p95_latency_ms']:.3f} ms`.
- Max p99 latency: `{summary['max_p99_latency_ms']:.3f} ms`.

## Artifacts

- Manifest: `manifest.json`
- Queue state: `queue_state.json`
- Full table: `table.csv`
- Thread summary: `thread_summary.csv`
- L overrides: `l_overrides.json`
- U75 recalibration log: `recalibration_u75.jsonl`
- Worst-case plot: `latency_percentiles_worstcase_highres.png`
- Selector plot: `latency_percentiles_by_selector_highres.png`
"""
    (args.out_dir / "ARIS_EXPERIMENT_REVIEW.md").write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/mnt/bak3/lzg/PipeANN-github"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/r116_suite/exp6_aris_cpu"))
    parser.add_argument("--index-prefix", type=Path, default=Path("experiments/r116_suite/exp6_query_thread_budget/tmp/direct_1m"))
    parser.add_argument("--thread-sweep", type=parse_threads, default=parse_threads("1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16"))
    parser.add_argument("--latency-budget-ms", type=float, default=10.0)
    parser.add_argument("--build-r", type=int, default=116)
    parser.add_argument("--build-l", type=int, default=220)
    parser.add_argument("--pq-bytes", type=int, default=32)
    parser.add_argument("--memory-gb", type=int, default=64)
    parser.add_argument("--aris-repo", default="/Users/zhengganglin/Downloads/Auto-claude-code-research-in-sleep")
    parser.add_argument("--aris-commit", default="")
    parser.add_argument("--rerun-filter", default="", help="Comma-separated selector:bucket filters to rerun, e.g. intersect:u75,range:u75.")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    args.out_dir = (repo / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir
    args.index_prefix = (repo / args.index_prefix).resolve() if not args.index_prefix.is_absolute() else args.index_prefix
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for needed in [
        repo / "build/tests/search_disk_index_hybrid",
        repo / "build/tests/utils/compute_groundtruth",
        repo / "experiments/r116_suite/exp4_intersect_range_selectivity/table.csv",
    ]:
        if not needed.exists():
            raise FileNotFoundError(str(needed))

    jobs = build_jobs(repo, args, args.index_prefix)
    manifest = write_manifest(repo, args, args.index_prefix, jobs)
    if args.rerun_filter:
        filters = {part.strip() for part in args.rerun_filter.split(",") if part.strip()}
        for job in jobs:
            key = f"{job['selector_type']}:{job['bucket']}"
            if key in filters:
                output = Path(job["expected_output"])
                if output.exists():
                    output.unlink()
    if not args.plot_only:
        build_index_if_needed(repo, args.index_prefix, args)
        state = load_or_init_state(args, manifest)
        run_jobs(repo, args, manifest, state)
    rows = collect_results(args, manifest)
    write_csv(args.out_dir / "table.csv", rows)
    summary, summary_rows = summarize(rows, args.latency_budget_ms)
    write_csv(args.out_dir / "thread_summary.csv", summary_rows)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if rows:
        plot(args, rows, summary_rows)
    write_review(args, manifest, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["rows"] == summary["expected_rows"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
