#!/usr/bin/env python3
"""Measure foreground PipeANN search while low-core background maintenance runs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


class BackgroundStoppedError(RuntimeError):
    """Raised after the background process has already been stopped and logged."""


def kill_process_group(proc: subprocess.Popen[str], sig: int) -> None:
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        pass


def normalize(path: Path) -> Path:
    return path.expanduser().resolve()


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def rewrite_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def run_capture(cmd: list[str]) -> str:
    return subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout


def run_logged(
    cmd: list[str],
    log_path: Path,
    *,
    env: dict[str, str] | None = None,
    timeout_s: float | None = None,
) -> tuple[float, str]:
    started = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    stdout = ""
    try:
        stdout, _ = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        kill_process_group(proc, signal.SIGTERM)
        try:
            stdout, _ = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            kill_process_group(proc, signal.SIGKILL)
            stdout, _ = proc.communicate()
    elapsed = time.time() - started
    returncode = proc.returncode if proc.returncode is not None else -signal.SIGKILL
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        log.write("$ " + shlex.join(cmd) + "\n")
        log.write(stdout)
        if stdout and not stdout.endswith("\n"):
            log.write("\n")
        if timed_out:
            log.write(f"[timeout_s] {timeout_s}\n")
        log.write(f"[returncode] {returncode}\n")
        log.write(f"[elapsed_wall_s] {elapsed:.6f}\n\n")
    if timed_out:
        raise RuntimeError(f"command timed out after {timeout_s}s: {shlex.join(cmd)}; see {log_path}")
    if returncode != 0:
        raise RuntimeError(f"command failed ({returncode}): {shlex.join(cmd)}; see {log_path}")
    return elapsed, stdout


def replace_arg(cmd: list[str], name: str, value: str) -> None:
    try:
        idx = cmd.index(name)
    except ValueError:
        cmd.extend([name, value])
        return
    if idx + 1 >= len(cmd):
        raise RuntimeError(f"argument {name} has no value in raw_command")
    cmd[idx + 1] = value


def get_arg(cmd: list[str], name: str) -> str:
    try:
        idx = cmd.index(name)
    except ValueError as exc:
        raise RuntimeError(f"raw_command missing {name}") from exc
    if idx + 1 >= len(cmd):
        raise RuntimeError(f"argument {name} has no value in raw_command")
    return cmd[idx + 1]


def selected_payload(row: dict[str, Any]) -> dict[str, Any]:
    payload = row.get("selected")
    return payload if isinstance(payload, dict) else row


def thread_env(threads: int) -> dict[str, str]:
    env = os.environ.copy()
    for name in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TBB_NUM_THREADS",
    ]:
        env[name] = str(threads)
    env["OMP_PROC_BIND"] = "false"
    env["OMP_PLACES"] = "cores"
    return env


def make_foreground_cmd(raw_command: str, args: argparse.Namespace, output_jsonl: Path) -> list[str]:
    cmd = shlex.split(raw_command)
    cmd[0] = str(args.binary_root / "dynamic_update_suite_driver")
    replace_arg(cmd, "--jsonl-output", str(output_jsonl))
    replace_arg(cmd, "--cpu-cap", str(args.foreground_cpu_cap))
    if args.query_limit > 0:
        replace_arg(cmd, "--query-limit", str(args.query_limit))
    return ["taskset", "-c", args.foreground_cpu_range, *cmd]


def make_schedule(rows: list[dict[str, Any]], args: argparse.Namespace, schedule: str) -> list[tuple[int, dict[str, Any]]]:
    if schedule == "none":
        return []
    if len(rows) == 1 and args.foreground_row_index != 0:
        args.foreground_row_index = 0
    if args.foreground_row_index < 0 or args.foreground_row_index >= len(rows):
        raise RuntimeError("--foreground-row-index out of range")
    if schedule == "single":
        return [(args.foreground_row_index, rows[args.foreground_row_index])]
    if schedule == "all-selected-once":
        return list(enumerate(rows))
    raise RuntimeError(f"unknown foreground schedule: {schedule}")


def disk_layout_meta(prefix_text: str) -> dict[str, Any]:
    disk_path = Path(str(prefix_text) + "_disk.index")
    if not disk_path.exists() or disk_path.stat().st_size < 64:
        return {"actual_disk_layout_status": "missing", "actual_disk_index_path": str(disk_path)}
    with disk_path.open("rb") as f:
        header = f.read(96)
    nr = int.from_bytes(header[0:4], "little")
    if nr < 11 or len(header) < 96:
        return {
            "actual_disk_layout_status": "ok",
            "actual_disk_index_path": str(disk_path),
            "actual_layout_version": 1,
        }
    return {
        "actual_disk_layout_status": "ok",
        "actual_disk_index_path": str(disk_path),
        "actual_layout_version": int.from_bytes(header[64:72], "little"),
        "actual_layout_block_bytes": int.from_bytes(header[72:80], "little"),
        "actual_layout_nodes_per_block": int.from_bytes(header[80:88], "little"),
        "actual_layout_read_page_bytes": int.from_bytes(header[88:96], "little"),
    }


def same_prefix(left: str, right: str) -> bool:
    if not left or not right:
        return False
    return normalize(Path(left)) == normalize(Path(right))


def source_metadata(source_row: dict[str, Any], row_index: int, raw_command: str) -> dict[str, Any]:
    metadata = {"source_row_index": row_index}
    for key in [
        "case_id",
        "prefix",
        "cycle_idx",
        "variant",
        "selector_type",
        "bucket",
        "route",
        "search_l",
        "v1_source_prefix",
        "v3_source_prefix",
        "layout",
        "layout_version",
        "layout_variant",
        "physical_read_unit_bytes",
        "per_node_read_request_bytes",
    ]:
        if key in source_row:
            metadata[f"source_{key}"] = source_row[key]
            metadata.setdefault(key, source_row[key])
    cmd = shlex.split(raw_command)
    command_source_prefix = get_arg(cmd, "--source-prefix")
    expected_v3_prefix = str(source_row.get("v3_source_prefix") or "")
    actual_layout = disk_layout_meta(command_source_prefix)
    row_layout_ok = (
        int(source_row.get("layout_version", 0) or 0) == 3
        and str(source_row.get("layout") or "") == "supersector32k"
        and str(source_row.get("layout_variant") or "") == "page_aware_slots"
        and int(source_row.get("physical_read_unit_bytes", 0) or 0) == 4096
        and int(source_row.get("per_node_read_request_bytes", 0) or 0) == 4096
    )
    actual_layout_ok = (
        actual_layout.get("actual_disk_layout_status") == "ok"
        and int(actual_layout.get("actual_layout_version", 0) or 0) == 3
        and int(actual_layout.get("actual_layout_block_bytes", 0) or 0) == 32768
        and int(actual_layout.get("actual_layout_read_page_bytes", 0) or 0) == 4096
    )
    prefix_match = same_prefix(command_source_prefix, expected_v3_prefix)
    metadata.update(
        {
            "foreground_command_source_prefix": command_source_prefix,
            "source_prefix_matches_v3": prefix_match,
            "source_row_layout_v3_supersector32k": row_layout_ok,
            "source_actual_layout_v3_supersector32k": actual_layout_ok,
            "source_layout_metadata_valid": prefix_match and row_layout_ok and actual_layout_ok,
            **actual_layout,
        }
    )
    return metadata


def run_foreground_once(
    source_row: dict[str, Any],
    row_index: int,
    args: argparse.Namespace,
    out: Path,
    condition: str,
    repeat: int,
) -> dict[str, Any]:
    raw_command = str(source_row.get("raw_command") or "")
    if not raw_command:
        raise RuntimeError(f"selected foreground row {row_index} has no raw_command")
    tmp = out / "tmp" / f"{condition}_{repeat:04d}_row{row_index:04d}.jsonl"
    if tmp.exists():
        tmp.unlink()
    cmd = make_foreground_cmd(raw_command, args, tmp)
    sample_started = time.time()
    elapsed, _ = run_logged(
        cmd,
        out / "logs" / f"foreground_{condition}.log",
        env=thread_env(args.foreground_cpu_cap),
        timeout_s=args.foreground_timeout_s,
    )
    rows = read_jsonl(tmp)
    if len(rows) != 1:
        raise RuntimeError(f"expected one foreground row in {tmp}, got {len(rows)}")
    row = rows[0]
    row.update(
        {
            "condition": condition,
            "repeat": repeat,
            "foreground_started_wall_s": sample_started,
            "foreground_finished_wall_s": time.time(),
            "foreground_elapsed_wall_s": elapsed,
            "foreground_cpu_range": args.foreground_cpu_range,
            "background_cpu_range": args.background_cpu_range if condition == "during_background" else "",
            **source_metadata(source_row, row_index, raw_command),
        }
    )
    append_jsonl(out / "foreground_search_results.jsonl", row)
    return row


def stop_background(
    proc: subprocess.Popen[str],
    started: float,
    out: Path,
    args: argparse.Namespace,
    reason: str,
    repeat: int,
    state: dict[str, Any] | None = None,
    lock: threading.Lock | None = None,
) -> None:
    if proc.poll() is None:
        kill_process_group(proc, signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            kill_process_group(proc, signal.SIGKILL)
            proc.wait()
    elapsed = time.time() - started
    if state is not None and lock is not None:
        elapsed = background_elapsed(started, state, lock)
    elif reason == "timeout":
        elapsed = min(elapsed, args.background_timeout_s)
    append_jsonl(
        out / "background_maintenance_results.jsonl",
        {
            "background_kind": "full_build_pq_retrain",
            "background_elapsed_wall_s": elapsed,
            "background_cpu_range": args.background_cpu_range,
            "background_cpu_cap": args.background_cpu_cap,
            "overlap_samples": repeat,
            "required_overlap_samples": args.required_during_rows,
            "status": reason,
        },
    )


def background_timed_out(state: dict[str, Any], lock: threading.Lock) -> bool:
    with lock:
        return bool(state.get("timed_out"))


def background_elapsed(started: float, state: dict[str, Any], lock: threading.Lock) -> float:
    with lock:
        exit_time = state.get("exit_time")
        if exit_time is None:
            exit_time = time.time()
            state["exit_time"] = exit_time
    return float(exit_time) - started


def start_background_watchdog(
    proc: subprocess.Popen[str], started: float, timeout_s: float
) -> tuple[dict[str, Any], threading.Lock, threading.Thread]:
    state: dict[str, Any] = {"timed_out": False, "exit_time": None}
    lock = threading.Lock()

    def watch() -> None:
        deadline = started + timeout_s
        while True:
            remaining = deadline - time.time()
            if remaining <= 0.0:
                if proc.poll() is None:
                    with lock:
                        state["timed_out"] = True
                        state["exit_time"] = deadline
                    kill_process_group(proc, signal.SIGTERM)
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        kill_process_group(proc, signal.SIGKILL)
                        proc.wait()
                    with lock:
                        state["exit_time"] = min(float(state.get("exit_time") or time.time()), time.time())
                else:
                    with lock:
                        if state.get("exit_time") is None:
                            state["exit_time"] = time.time()
                return
            try:
                proc.wait(timeout=remaining)
                with lock:
                    if state.get("exit_time") is None:
                        state["exit_time"] = time.time()
                return
            except subprocess.TimeoutExpired:
                with lock:
                    state["timed_out"] = True
                    state["exit_time"] = deadline
                if proc.poll() is None:
                    kill_process_group(proc, signal.SIGTERM)
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        kill_process_group(proc, signal.SIGKILL)
                        proc.wait()
                return

    thread = threading.Thread(target=watch, name="background-timeout-watchdog", daemon=True)
    thread.start()
    return state, lock, thread


def build_background_cmd(raw_command: str, args: argparse.Namespace, out: Path) -> list[str]:
    fg_cmd = shlex.split(raw_command)
    data_bin = get_arg(fg_cmd, "--data-bin")
    label_file = get_arg(fg_cmd, "--base-label-file")
    dest_prefix = args.background_index_root / f"full_build_pq_retrain_{int(time.time())}"
    write_json(out / "background_prefix.json", {"prefix": str(dest_prefix)})
    return [
        "taskset",
        "-c",
        args.background_cpu_range,
        str(args.build_binary),
        "float",
        data_bin,
        str(dest_prefix),
        str(args.build_r),
        str(args.build_l),
        str(args.pq_bytes),
        str(args.build_memory_gb),
        str(args.background_cpu_cap),
        "l2",
        "pq",
        "spmat",
        label_file,
        "--label-storage",
        "sidecar",
    ]


def start_background(cmd: list[str], args: argparse.Namespace, out: Path) -> tuple[subprocess.Popen[str], float]:
    log_path = out / "logs" / "background_full_build.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("a")
    log.write("$ " + shlex.join(cmd) + "\n")
    log.flush()
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=thread_env(args.background_cpu_cap),
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    return proc, time.time()


def finish_background(
    proc: subprocess.Popen[str],
    started: float,
    out: Path,
    args: argparse.Namespace,
    *,
    overlap_samples: int,
    background_state: dict[str, Any],
    background_lock: threading.Lock,
) -> dict[str, Any]:
    code = proc.wait()
    elapsed = background_elapsed(started, background_state, background_lock)
    log_path = out / "logs" / "background_full_build.log"
    with log_path.open("a") as log:
        log.write(f"[returncode] {code}\n")
        log.write(f"[elapsed_wall_s] {elapsed:.6f}\n")
    text = log_path.read_text(errors="replace")
    def extract(pattern: str) -> float | None:
        match = re.search(pattern, text)
        return float(match.group(1)) if match else None
    row = {
        "background_kind": "full_build_pq_retrain",
        "background_elapsed_wall_s": elapsed,
        "pq_train_wall_s": extract(r"Pivots generated in ([0-9.]+)s"),
        "pq_recode_wall_s": extract(r"Compressed data written in: ([0-9.]+)s"),
        "background_cpu_range": args.background_cpu_range,
        "background_cpu_cap": args.background_cpu_cap,
        "overlap_samples": overlap_samples,
        "required_overlap_samples": args.required_during_rows,
        "status": "ok" if code == 0 else "failed",
    }
    append_jsonl(out / "background_maintenance_results.jsonl", row)
    if code != 0:
        raise RuntimeError(f"background full build failed ({code}); see {log_path}")
    return row


def summarize(out: Path, foreground_rows: list[dict[str, Any]], background_row: dict[str, Any], args: argparse.Namespace) -> None:
    baseline = [r for r in foreground_rows if r["condition"] == "baseline"]
    during = [r for r in foreground_rows if r["condition"] == "during_background"]
    during_unique_rows = {r.get("source_row_index") for r in during}
    during_4k_pairs = [
        (float(r.get("mean_read_size", 0.0)), float(r.get("mean_n_4k", 0.0)))
        for r in during
        if r.get("mean_read_size") not in (None, "") and r.get("mean_n_4k") not in (None, "")
    ]
    during_4k_ratio_pass = [
        n_4k > 0.0 and abs((read_size / n_4k) - 4096.0) <= 1e-6
        for read_size, n_4k in during_4k_pairs
    ]
    during_unit_4k_pass = [
        int(r.get("physical_read_unit_bytes", 0) or 0) == 4096
        and int(r.get("per_node_read_request_bytes", 0) or 0) == 4096
        for r in during
    ]
    during_layout_metadata_pass = [bool(r.get("source_layout_metadata_valid")) for r in during]
    summary = {
        "foreground_rows": len(foreground_rows),
        "baseline_rows": len(baseline),
        "during_background_rows": len(during),
        "during_unique_source_rows": len(during_unique_rows),
        "required_during_rows": args.required_during_rows,
        "foreground_cpu_range": args.foreground_cpu_range,
        "background_cpu_range": args.background_cpu_range,
        "background_cpu_cap": args.background_cpu_cap,
        "background_kind": background_row.get("background_kind"),
        "background_elapsed_wall_s": background_row.get("background_elapsed_wall_s"),
        "background_pq_train_wall_s": background_row.get("pq_train_wall_s"),
        "background_pq_recode_wall_s": background_row.get("pq_recode_wall_s"),
        "baseline_max_avg_latency_ms": max((float(r.get("avg_latency_us", 0.0)) / 1000.0 for r in baseline), default=0.0),
        "during_max_avg_latency_ms": max((float(r.get("avg_latency_us", 0.0)) / 1000.0 for r in during), default=0.0),
        "baseline_max_p95_latency_ms": max((float(r.get("p95_latency_us", 0.0)) / 1000.0 for r in baseline), default=0.0),
        "during_max_p95_latency_ms": max((float(r.get("p95_latency_us", 0.0)) / 1000.0 for r in during), default=0.0),
        "during_recall_pass": sum(1 for r in during if float(r.get("recall@10", 0.0)) >= 98.0),
        "during_avg_lt_10ms_pass": sum(1 for r in during if float(r.get("avg_latency_us", 1e18)) < 10000.0),
        "during_p95_lt_10ms_pass": sum(1 for r in during if float(r.get("p95_latency_us", 1e18)) < 10000.0),
        "during_4k_evidence_rows": len(during_4k_pairs),
        "during_4k_ratio_pass": sum(1 for passed in during_4k_ratio_pass if passed),
        "during_unit_4k_pass": sum(1 for passed in during_unit_4k_pass if passed),
        "during_layout_metadata_pass": sum(1 for passed in during_layout_metadata_pass if passed),
        "during_max_read_size_per_4k": max((read_size / n_4k for read_size, n_4k in during_4k_pairs if n_4k > 0.0), default=0.0),
        "foreground_schedule": args.foreground_schedule,
        "baseline_schedule": args.baseline_schedule,
    }
    summary["claim_status"] = (
        "PASS"
        if during
        and len(during) >= args.required_during_rows
        and summary["during_recall_pass"] == len(during)
        and summary["during_avg_lt_10ms_pass"] == len(during)
        and summary["during_p95_lt_10ms_pass"] == len(during)
        and len(during_4k_pairs) == len(during)
        and summary["during_4k_ratio_pass"] == len(during)
        and summary["during_unit_4k_pass"] == len(during)
        and summary["during_layout_metadata_pass"] == len(during)
        else "FAIL"
    )
    summary["claim_status_with_p95"] = summary["claim_status"]
    if len(during_4k_pairs) != len(during):
        summary["four_k_read_size_status"] = "NEEDS_EVIDENCE"
    else:
        summary["four_k_read_size_status"] = (
            "PASS"
            if summary["during_4k_ratio_pass"] == len(during)
            and summary["during_unit_4k_pass"] == len(during)
            and summary["during_layout_metadata_pass"] == len(during)
            else "FAIL"
        )
    write_json(out / "background_interference_summary.json", summary)
    write_csv(out / "foreground_search_results.csv", foreground_rows)
    write_csv(out / "background_interference_results.csv", foreground_rows)


def write_phase0(out: Path, args: argparse.Namespace) -> None:
    logs = out / "logs"
    for name, cmd in {
        "phase0_git_status.log": ["git", "status", "--short"],
        "phase0_git_head.log": ["git", "rev-parse", "HEAD"],
        "phase0_git_diff_stat.log": ["git", "diff", "--stat"],
        "phase0_uname.log": ["uname", "-a"],
        "phase0_lscpu.log": ["lscpu"],
    }.items():
        (logs / name).parent.mkdir(parents=True, exist_ok=True)
        (logs / name).write_text(run_capture(cmd))
    write_json(
        out / "evidence" / "runner_config.json",
        {
            "foreground_selected_jsonl": str(args.foreground_selected_jsonl),
            "foreground_row_index": args.foreground_row_index,
            "foreground_schedule": args.foreground_schedule,
            "baseline_schedule": args.baseline_schedule,
            "binary_root": str(args.binary_root),
            "build_binary": str(args.build_binary),
            "background_index_root": str(args.background_index_root),
            "foreground_cpu_range": args.foreground_cpu_range,
            "background_cpu_range": args.background_cpu_range,
            "baseline_repeats": args.baseline_repeats,
            "during_repeats": args.during_repeats,
            "query_limit": args.query_limit,
            "background_timeout_s": args.background_timeout_s,
            "foreground_timeout_s": args.foreground_timeout_s,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--foreground-selected-jsonl", type=Path, required=True)
    parser.add_argument("--foreground-row-index", type=int, default=2)
    parser.add_argument("--foreground-schedule", choices=["single", "all-selected-once"], default="single")
    parser.add_argument("--baseline-schedule", choices=["none", "single", "all-selected-once"], default="single")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--binary-root", type=Path, default=ROOT / "build_reviewed_20260601_explicitmat" / "tests")
    parser.add_argument("--build-binary", type=Path, default=ROOT / "build_reviewed_20260601_explicitmat" / "tests" / "build_disk_index")
    parser.add_argument("--background-index-root", type=Path, default=Path("/mnt/nvme1n1/PipeANN-maintenance-work/indexes"))
    parser.add_argument("--foreground-cpu-range", default="20-27")
    parser.add_argument("--foreground-cpu-cap", type=int, default=8)
    parser.add_argument("--background-cpu-range", default="16-19")
    parser.add_argument("--background-cpu-cap", type=int, default=4)
    parser.add_argument("--baseline-repeats", type=int, default=2)
    parser.add_argument(
        "--during-repeats",
        type=int,
        default=None,
        help="Single schedule: number of foreground samples. all-selected-once schedule: number of full selected-list passes.",
    )
    parser.add_argument("--query-limit", type=int, default=1000)
    parser.add_argument("--build-r", type=int, default=116)
    parser.add_argument("--build-l", type=int, default=220)
    parser.add_argument("--pq-bytes", type=int, default=16)
    parser.add_argument("--build-memory-gb", type=int, default=64)
    parser.add_argument("--background-timeout-s", type=float, default=1800.0)
    parser.add_argument("--foreground-timeout-s", type=float, default=300.0)
    args = parser.parse_args()
    args.foreground_selected_jsonl = normalize(args.foreground_selected_jsonl if args.foreground_selected_jsonl.is_absolute() else ROOT / args.foreground_selected_jsonl)
    args.out_dir = normalize(args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir)
    args.binary_root = normalize(args.binary_root if args.binary_root.is_absolute() else ROOT / args.binary_root)
    args.build_binary = normalize(args.build_binary if args.build_binary.is_absolute() else ROOT / args.build_binary)
    args.background_index_root = normalize(args.background_index_root)
    if is_relative_to(args.background_index_root, ROOT):
        raise RuntimeError("--background-index-root must be outside the repository")
    source_root = args.foreground_selected_jsonl.parent
    if source_root.name == "raw":
        source_root = source_root.parent
    if args.out_dir == source_root or is_relative_to(args.out_dir, source_root):
        raise RuntimeError("--out-dir must not be the selected source experiment or one of its children")
    if args.foreground_cpu_cap <= 0 or args.background_cpu_cap <= 0:
        raise RuntimeError("CPU caps must be positive")
    if args.during_repeats is None:
        args.during_repeats = 1 if args.foreground_schedule == "all-selected-once" else 6
    if args.baseline_repeats < 0 or args.during_repeats <= 0:
        raise RuntimeError("--baseline-repeats must be non-negative and --during-repeats must be positive")
    return args


def main() -> int:
    args = parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "tmp").mkdir(parents=True, exist_ok=True)
    args.background_index_root.mkdir(parents=True, exist_ok=True)
    rows = [selected_payload(row) for row in read_jsonl(args.foreground_selected_jsonl)]
    baseline_schedule = make_schedule(rows, args, args.baseline_schedule)
    during_schedule = make_schedule(rows, args, args.foreground_schedule)
    write_phase0(out, args)
    if not during_schedule:
        raise RuntimeError("foreground schedule produced no during-background rows")
    args.required_during_rows = len(during_schedule) * args.during_repeats
    background_seed_row = rows[args.foreground_row_index]
    background_raw_command = str(background_seed_row.get("raw_command") or "")
    if not background_raw_command:
        raise RuntimeError("selected background seed row has no raw_command")

    foreground_rows: list[dict[str, Any]] = []
    for repeat in range(args.baseline_repeats):
        for position, (row_index, source_row) in enumerate(baseline_schedule):
            sample_id = repeat if args.baseline_schedule == "single" else repeat * len(baseline_schedule) + position
            foreground_rows.append(run_foreground_once(source_row, row_index, args, out, "baseline", sample_id))

    background_cmd = build_background_cmd(background_raw_command, args, out)
    proc, started = start_background(background_cmd, args, out)
    background_state, background_lock, background_watchdog = start_background_watchdog(proc, started, args.background_timeout_s)
    repeat = 0
    try:
        while proc.poll() is None:
            if background_timed_out(background_state, background_lock):
                stop_background(proc, started, out, args, "timeout", repeat, background_state, background_lock)
                raise BackgroundStoppedError(f"background maintenance exceeded timeout {args.background_timeout_s}s")
            if repeat < args.required_during_rows or args.foreground_schedule == "single":
                row_index, source_row = during_schedule[repeat % len(during_schedule)]
                foreground_rows.append(run_foreground_once(source_row, row_index, args, out, "during_background", repeat))
                repeat += 1
                if repeat >= args.required_during_rows and args.foreground_schedule == "single" and proc.poll() is None:
                    time.sleep(10)
            else:
                time.sleep(10)
    except BackgroundStoppedError:
        raise
    except Exception:
        if background_timed_out(background_state, background_lock):
            stop_background(proc, started, out, args, "timeout", repeat, background_state, background_lock)
            raise BackgroundStoppedError(f"background maintenance exceeded timeout {args.background_timeout_s}s")
        if proc.poll() is not None and proc.returncode not in (0, None):
            finish_background(
                proc,
                started,
                out,
                args,
                overlap_samples=repeat,
                background_state=background_state,
                background_lock=background_lock,
            )
        stop_background(proc, started, out, args, "foreground_failed", repeat)
        raise
    background_watchdog.join(timeout=1.0)
    if background_timed_out(background_state, background_lock):
        stop_background(proc, started, out, args, "timeout", repeat, background_state, background_lock)
        raise BackgroundStoppedError(f"background maintenance exceeded timeout {args.background_timeout_s}s")
    background_done_at = started + background_elapsed(started, background_state, background_lock)
    overlap_rows = [
        row for row in foreground_rows
        if row.get("condition") == "during_background"
        and float(row.get("foreground_started_wall_s", 1e18)) < background_done_at
    ]
    if len(overlap_rows) != repeat:
        foreground_rows = [row for row in foreground_rows if row.get("condition") != "during_background"] + overlap_rows
        repeat = len(overlap_rows)
        rewrite_jsonl(out / "foreground_search_results.jsonl", foreground_rows)
    if proc.returncode not in (0, None):
        finish_background(
            proc,
            started,
            out,
            args,
            overlap_samples=repeat,
            background_state=background_state,
            background_lock=background_lock,
        )
    if repeat < args.required_during_rows:
        stop_background(proc, started, out, args, "insufficient_overlap", repeat, background_state, background_lock)
        raise RuntimeError("background maintenance ended before enough overlapping foreground samples were collected")
    background_row = finish_background(
        proc,
        started,
        out,
        args,
        overlap_samples=repeat,
        background_state=background_state,
        background_lock=background_lock,
    )
    summarize(out, foreground_rows, background_row, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
