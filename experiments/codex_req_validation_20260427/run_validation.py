#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pipeann_hybrid_experiment as phe  # noqa: E402


EXP_ROOT = Path(__file__).resolve().parent
SOURCE_EXP = REPO_ROOT / "experiments" / "sift1m_uniform_prefilter_pq_staged_compare_autoroute_tau_insert6_20260427"
SOURCE_MANIFEST = SOURCE_EXP / "stage_manifest.json"
COMMANDS_JSONL = EXP_ROOT / "commands.jsonl"
SUMMARY_JSON = EXP_ROOT / "summary.json"

SELECTIVITY_SPECS = (
    ("u1e-03", 0.001),
    ("u3e-03", 0.003),
    ("u1e-02", 0.01),
    ("u5e-02", 0.05),
    ("u1e-01", 0.1),
    ("u25", 0.25),
    ("u30", 0.3),
    ("u50", 0.5),
    ("u75", 0.75),
    ("u100", 1.0),
)
EQUALITY_BUCKETS = ("u1e-03", "u1e-02", "u1e-01", "u25", "u100")
QUERY_COUNT = 100
PROBE_QUERY_COUNT = 100
K = 10
THREADS = 1
BEAMWIDTH = 4
SEARCH_L = 100
TARGET_RECALL = 98.0
LATENCY_LIMIT_US = 10_000.0
MEMORY_LIMIT_KB = 30 * 1024
PQ_BITS = 8


def repo_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as reader:
        return json.load(reader)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as writer:
        json.dump(payload, writer, indent=2, sort_keys=True)
        writer.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as reader:
        for line in reader:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as writer:
        json.dump(record, writer, sort_keys=True)
        writer.write("\n")


def run_command(
    name: str,
    cmd: Sequence[str],
    *,
    cwd: Path = REPO_ROOT,
    env_overrides: dict[str, str] | None = None,
    log_path: Path | None = None,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    ensure_dir(EXP_ROOT)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    if env_overrides:
        env.update(env_overrides)

    started_at = time.time()
    append_jsonl(
        COMMANDS_JSONL,
        {
            "event": "start",
            "name": name,
            "cmd": list(cmd),
            "cwd": str(cwd),
            "env_overrides": env_overrides or {},
            "started_at": started_at,
        },
    )
    result = subprocess.run(
        list(cmd),
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    ended_at = time.time()
    if log_path is not None:
        ensure_dir(log_path.parent)
        with log_path.open("w", encoding="utf-8") as writer:
            writer.write("$ " + " ".join(cmd) + "\n")
            writer.write("\n[stdout]\n")
            writer.write(result.stdout)
            writer.write("\n[stderr]\n")
            writer.write(result.stderr)
    append_jsonl(
        COMMANDS_JSONL,
        {
            "event": "end",
            "name": name,
            "returncode": result.returncode,
            "duration_s": ended_at - started_at,
            "ended_at": ended_at,
            "log_path": None if log_path is None else str(log_path),
        },
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({name}, rc={result.returncode}); see {log_path or 'captured output'}"
        )
    return result


def ensure_no_competing_processes() -> None:
    result = subprocess.run(
        ["ps", "-eo", "pid,ppid,stat,etime,pcpu,pmem,comm,args"],
        text=True,
        capture_output=True,
        check=True,
    )
    needles = (
        "run_sift1m",
        "pipeann_hybrid_experiment.py",
        "search_disk_index_hybrid",
        "dynamic_prefilter_stage_driver",
    )
    current_pid = os.getpid()
    offenders: list[str] = []
    for line in result.stdout.splitlines():
        if str(current_pid) in line or "run_validation.py" in line:
            continue
        if any(needle in line for needle in needles):
            offenders.append(line)
    if offenders:
        raise RuntimeError("competing experiment processes are running:\n" + "\n".join(offenders))


def stage_by_name(manifest: dict[str, Any], stage_name: str) -> dict[str, Any]:
    for stage in manifest["stages"]:
        if stage["name"] == stage_name:
            return stage
    raise KeyError(stage_name)


def ensure_stage_1m_base(manifest: dict[str, Any]) -> None:
    stage_1m = stage_by_name(manifest, "stage_1m")
    stage_base = Path(stage_1m["base_bin"])
    if stage_base.exists():
        return
    full_base = Path(manifest["base_bin"])
    ensure_dir(stage_base.parent)
    try:
        os.symlink(full_base, stage_base)
    except OSError:
        shutil.copy2(full_base, stage_base)


def workload_counts(summary_json: Path) -> tuple[int | None, int | None]:
    payload = read_json(summary_json)
    compare_query_count = payload.get("queries_per_selectivity")
    if compare_query_count is None:
        compare_query_count = payload.get("queries_per_bucket")
    probe_query_count = payload.get("probe_queries_per_selectivity")
    return (
        None if compare_query_count is None else int(compare_query_count),
        None if probe_query_count is None else int(probe_query_count),
    )


def ensure_stage_workloads(manifest: dict[str, Any], stage: dict[str, Any]) -> None:
    workload_dir = Path(stage["workload_dir"])
    summary_json = workload_dir / "uniform_exact_selectivity_summary.json"
    if summary_json.exists() and workload_counts(summary_json) == (QUERY_COUNT, PROBE_QUERY_COUNT):
        return
    if workload_dir.exists():
        shutil.rmtree(workload_dir)
    cmd = [
        str(REPO_ROOT / ".venv" / "bin" / "python"),
        "scripts/pipeann_hybrid_experiment.py",
        "generate-uniform-exact-selectivity-workloads",
        "--base-bin",
        stage["base_bin"],
        "--query-bin",
        manifest["query_bin"],
        "--index-type",
        "float",
        "--selector-type",
        "intersect",
        "--out-dir",
        str(workload_dir),
        "--queries-per-bucket",
        str(QUERY_COUNT),
        "--probe-queries-per-bucket",
        str(PROBE_QUERY_COUNT),
    ]
    for bucket_name, selectivity in SELECTIVITY_SPECS:
        cmd.extend(["--selectivity-spec", f"{bucket_name}:{selectivity}"])
    run_command(
        f"generate_workloads_{stage['name']}",
        cmd,
        log_path=EXP_ROOT / "logs" / f"generate_workloads_{stage['name']}.log",
        timeout=7200,
    )


def existing_runtime_prefix(stage: dict[str, Any]) -> Path:
    return Path(stage["runtime_dir"]) / f"sift1m_uniform_prefilter_compare_{stage['short_name']}_pq{PQ_BITS}"


def source_prefix(stage: dict[str, Any]) -> Path:
    return Path(stage["source_prefixes"][str(PQ_BITS)])


def run_dynamic_transition(
    previous_stage: dict[str, Any],
    next_stage: dict[str, Any],
    *,
    source: Path,
) -> dict[str, Any]:
    dest = source_prefix(next_stage)
    transition_dir = EXP_ROOT / "dynamic" / f"{previous_stage['short_name']}_to_{next_stage['short_name']}"
    summary_path = transition_dir / "insert_batch_summary.json"
    probe_path = transition_dir / "during_insert_probe.jsonl"
    if Path(f"{dest}_disk.index").exists() and summary_path.exists() and probe_path.exists():
        return {
            "transition": f"{previous_stage['short_name']}->{next_stage['short_name']}",
            "dest_prefix": str(dest),
            "insert_summary": str(summary_path),
            "probe_jsonl": str(probe_path),
            "skipped": True,
        }

    ensure_dir(dest.parent)
    ensure_dir(transition_dir)
    for path in dest.parent.glob(dest.name + "*"):
        if path.is_file() or path.is_symlink():
            path.unlink()
    if probe_path.exists():
        probe_path.unlink()

    cmd = [
        str(REPO_ROOT / "build" / "tests" / "dynamic_prefilter_stage_driver"),
        "--source-prefix",
        str(source),
        "--dest-prefix",
        str(dest),
        "--data-bin",
        next_stage["base_bin"],
        "--label-file",
        str(Path(next_stage["workload_dir"]) / "base.uniform_exact_selectivity.spmat"),
        "--query-bin",
        str(Path(next_stage["workload_dir"]) / "probe_queries.bin"),
        "--workload-dir",
        next_stage["workload_dir"],
        "--probe-jsonl",
        str(probe_path),
        "--insert-summary-json",
        str(summary_path),
        "--selector-type",
        "intersect",
        "--metric",
        "l2",
        "--insert-start",
        str(previous_stage["npoints"]),
        "--insert-count",
        str(next_stage["npoints"] - previous_stage["npoints"]),
        "--insert-threads",
        "8",
        "--search-threads",
        str(THREADS),
        "--materialize-threads",
        "4",
        "--beamwidth",
        str(BEAMWIDTH),
        "--k",
        str(K),
        "--search-l",
        str(SEARCH_L),
        "--mem-l",
        "0",
        "--recalibration-sample-limit",
        "100",
        "--recalibration-timeout-s",
        "900",
    ]
    for bucket_name, selectivity in SELECTIVITY_SPECS:
        cmd.extend(["--bucket-spec", f"{bucket_name}:{selectivity}"])

    run_command(
        f"dynamic_{previous_stage['short_name']}_to_{next_stage['short_name']}",
        cmd,
        log_path=transition_dir / "dynamic_transition.log",
        timeout=7200,
    )
    return {
        "transition": f"{previous_stage['short_name']}->{next_stage['short_name']}",
        "dest_prefix": str(dest),
        "insert_summary": str(summary_path),
        "probe_jsonl": str(probe_path),
        "skipped": False,
    }


def prepare_runtime_prefix(stage: dict[str, Any]) -> Path:
    runtime_prefix = EXP_ROOT / "runtime" / f"sift1m_validation_{stage['short_name']}_pq{PQ_BITS}"
    if not (Path(f"{runtime_prefix}_disk.index").exists() and Path(f"{runtime_prefix}_labels.densebit").exists()):
        cmd = [
            str(REPO_ROOT / ".venv" / "bin" / "python"),
            "scripts/pipeann_hybrid_experiment.py",
            "prepare-index-prefix-for-labels",
            "--source-prefix",
            str(source_prefix(stage)),
            "--dest-prefix",
            str(runtime_prefix),
            "--label-file",
            str(Path(stage["workload_dir"]) / "base.uniform_exact_selectivity.spmat"),
            "--summary-json",
            str(EXP_ROOT / "runtime" / "prepare_runtime_stage_1m.json"),
        ]
        run_command(
            "prepare_runtime_stage_1m",
            cmd,
            log_path=EXP_ROOT / "logs" / "prepare_runtime_stage_1m.log",
            timeout=3600,
        )
    bootstrap_runtime_metadata(stage, runtime_prefix, "intersect")
    return runtime_prefix


def selector_mask(selector_type: str) -> int:
    if selector_type == "intersect":
        return 1
    if selector_type == "subset":
        return 2
    if selector_type == "range":
        return 4
    raise ValueError(f"unsupported selector_type: {selector_type}")


def bootstrap_runtime_metadata(stage: dict[str, Any], runtime_prefix: Path, selector_type: str) -> None:
    meta_path = Path(f"{runtime_prefix}_hybrid.meta")
    wanted_mask = selector_mask(selector_type)
    if meta_path.exists() and (hybrid_metadata_selector_mask(meta_path) & wanted_mask) == wanted_mask:
        return
    if meta_path.exists():
        meta_path.unlink()
    cmd = [
        str(REPO_ROOT / "build" / "tests" / "dynamic_prefilter_stage_driver"),
        "--source-prefix",
        str(runtime_prefix),
        "--dest-prefix",
        str(runtime_prefix),
        "--label-file",
        str(Path(stage["workload_dir"]) / "base.uniform_exact_selectivity.spmat"),
        "--query-bin",
        str(Path(stage["workload_dir"]) / "probe_queries.bin"),
        "--workload-dir",
        stage["workload_dir"],
        "--insert-start",
        "0",
        "--insert-count",
        "0",
        "--insert-threads",
        "1",
        "--search-threads",
        str(THREADS),
        "--materialize-threads",
        "1",
        "--beamwidth",
        str(BEAMWIDTH),
        "--k",
        str(K),
        "--search-l",
        str(SEARCH_L),
        "--mem-l",
        "0",
        "--selector-type",
        selector_type,
        "--metric",
        "l2",
        "--recalibration-sample-limit",
        "100",
        "--recalibration-timeout-s",
        "900",
    ]
    for bucket_name, selectivity in SELECTIVITY_SPECS:
        cmd.extend(["--bucket-spec", f"{bucket_name}:{selectivity}"])
    run_command(
        f"bootstrap_runtime_stage_1m_{selector_type}",
        cmd,
        log_path=EXP_ROOT / "runtime" / f"bootstrap_runtime_stage_1m_{selector_type}.log",
        timeout=1800,
    )


def hybrid_metadata_selector_mask(meta_path: Path) -> int:
    try:
        raw = meta_path.read_bytes()[:32]
        if len(raw) < 32:
            return 0
        return int(struct.unpack_from("<QIIQQ", raw, 0)[4])
    except OSError:
        return 0


def calibrate_equality(stage: dict[str, Any], index_prefix: Path) -> Path:
    out_dir = EXP_ROOT / "equality_calibration"
    output_json = out_dir / "prefilter_rerank_calibration.json"
    if output_json.exists():
        payload = read_json(output_json)
        done = {record.get("bucket_name") for record in payload.get("results", [])}
        if all(bucket in done for bucket in EQUALITY_BUCKETS):
            return output_json

    cmd = [
        str(REPO_ROOT / ".venv" / "bin" / "python"),
        "scripts/pipeann_hybrid_experiment.py",
        "calibrate-rerank",
        "--summary-json",
        str(Path(stage["workload_dir"]) / "uniform_exact_selectivity_summary.json"),
        "--index-prefix",
        str(index_prefix),
        "--out-dir",
        str(out_dir),
        "--output-json",
        str(output_json),
        "--threads",
        str(THREADS),
        "--beamwidth",
        str(BEAMWIDTH),
        "--k",
        str(K),
        "--similarity",
        "l2",
        "--nbr-type",
        "pq",
        "--mem-l",
        "0",
        "--search-l",
        str(SEARCH_L),
        "--target-recall",
        str(int(TARGET_RECALL)),
        "--calibration-queries",
        str(QUERY_COUNT),
        "--max-selectivity",
        "1.0",
        "--block-candidates",
        "16384",
        "--timeout",
        "7200",
    ]
    for bucket_name in EQUALITY_BUCKETS:
        cmd.extend(["--bucket", bucket_name])
    run_command(
        "calibrate_equality_stage_1m",
        cmd,
        log_path=EXP_ROOT / "logs" / "calibrate_equality_stage_1m.log",
        timeout=7200,
        env_overrides={"PIPEANN_PQ_MMAP": "1"},
    )
    return output_json


def write_constant_spmat(path: Path, nrows: int, nlabels: int, row_labels: Sequence[int]) -> None:
    labels = np.asarray(list(row_labels), dtype=np.int32)
    indptr = np.arange(nrows + 1, dtype=np.int64) * labels.size
    indices = np.tile(labels, nrows).astype(np.int32)
    data = np.ones(indices.shape[0], dtype=np.float32)
    ensure_dir(path.parent)
    with path.open("wb") as writer:
        writer.write(phe.SPMAT_HEADER.pack(nrows, nlabels, int(indices.shape[0])))
        indptr.tofile(writer)
        indices.tofile(writer)
        data.tofile(writer)


def create_range_workload(stage: dict[str, Any], index_prefix: Path) -> dict[str, Any]:
    range_dir = EXP_ROOT / "range"
    qbin = range_dir / "queries.bin"
    qspmat = range_dir / "queries.spmat"
    truthset = range_dir / "truthset.bin"
    metadata_path = range_dir / "metadata.json"
    if qbin.exists() and qspmat.exists() and truthset.exists() and metadata_path.exists():
        return read_json(metadata_path)

    summary = read_json(Path(stage["workload_dir"]) / "uniform_exact_selectivity_summary.json")
    sidecar = phe.DenseBitsetSidecar.load(Path(f"{index_prefix}_labels.densebit"))
    tags_by_id = phe.load_tags_by_id(index_prefix, sidecar.npoints)
    _, _, query_rows = phe.load_bin_matrix(Path(summary["query_bin"]), "float")
    _, _, base_rows = phe.load_bin_matrix(Path(summary["base_bin"]), "float")

    row_ids = list(range(min(QUERY_COUNT, query_rows.shape[0])))
    phe.write_bin_subset(qbin, query_rows, row_ids)
    write_constant_spmat(qspmat, len(row_ids), sidecar.nlabels, [0, 2])

    candidate_index_ids = sidecar.materialize_candidates("range", [0, 2])
    candidate_tags = np.asarray(tags_by_id[candidate_index_ids], dtype=np.uint32)
    exact_topk_ids = phe.compute_exact_topk_ids(
        np.asarray(query_rows[row_ids], dtype=np.float32),
        base_rows,
        candidate_tags,
        k=K,
        similarity="l2",
        block_candidates=16384,
    )
    phe.write_truthset_ids(truthset, exact_topk_ids)
    metadata = {
        "selector_type": "range",
        "range": [0, 2],
        "query_count": len(row_ids),
        "candidate_count": int(candidate_tags.shape[0]),
        "selectivity": float(candidate_tags.shape[0]) / float(sidecar.npoints),
        "query_bin": str(qbin),
        "query_labels": str(qspmat),
        "truthset": str(truthset),
    }
    write_json(metadata_path, metadata)
    return metadata


def extract_search_record(path: Path) -> dict[str, Any]:
    records = [record for record in read_jsonl(path) if record.get("format") == "pipeann.hybrid.search.v1"]
    if len(records) != 1:
        raise RuntimeError(f"expected one search record in {path}, found {len(records)}")
    return records[0]


def run_hybrid_search(
    name: str,
    index_prefix: Path,
    query_bin: Path,
    truthset: Path | str,
    selector_type: str,
    query_labels: Path,
    route: str,
    rerank_l: int,
    jsonl_path: Path,
    log_path: Path,
) -> dict[str, Any]:
    if jsonl_path.exists():
        jsonl_path.unlink()
    cmd = [
        str(REPO_ROOT / "build" / "tests" / "search_disk_index_hybrid"),
        "float",
        str(index_prefix),
        str(THREADS),
        str(BEAMWIDTH),
        str(query_bin),
        str(truthset),
        str(K),
        "l2",
        "pq",
        selector_type,
        str(query_labels),
        route,
        "0",
        "0",
        str(SEARCH_L),
        "--jsonl-output",
        str(jsonl_path),
    ]
    run_command(
        name,
        cmd,
        log_path=log_path,
        timeout=7200,
        env_overrides={"PIPEANN_PQ_MMAP": "1", "PIPEANN_PREFILTER_RERANK_L": str(rerank_l)},
    )
    record = extract_search_record(jsonl_path)
    record["prefilter_rerank_l"] = int(rerank_l)
    return record


def evaluate_range(stage: dict[str, Any], index_prefix: Path) -> dict[str, Any]:
    bootstrap_runtime_metadata(stage, index_prefix, "range")
    metadata = create_range_workload(stage, index_prefix)
    range_dir = EXP_ROOT / "range"
    result_json = range_dir / "result.json"
    if result_json.exists():
        existing = read_json(result_json)
        best = existing.get("best", {})
        if (
            existing.get("pass") is True
            and int(best.get("fallback_count", 0)) == 0
            and int(best.get("prefilter_count", 0)) > 0
        ):
            return existing
        result_json.unlink()

    candidate_count = int(metadata["candidate_count"])
    evaluations: list[dict[str, Any]] = []

    def evaluate(rerank_l: int) -> dict[str, Any]:
        rerank_l = max(K, min(candidate_count, int(rerank_l)))
        for existing in evaluations:
            if int(existing["prefilter_rerank_l"]) == rerank_l:
                return existing
        record = run_hybrid_search(
            f"range_rerank_{rerank_l}",
            index_prefix,
            Path(metadata["query_bin"]),
            Path(metadata["truthset"]),
            "range",
            Path(metadata["query_labels"]),
            "prefilter",
            rerank_l,
            range_dir / f"rerank_{rerank_l}.jsonl",
            range_dir / f"rerank_{rerank_l}.log",
        )
        evaluations.append(record)
        return record

    full_record = evaluate(candidate_count)
    if float(full_record["recall"]) >= TARGET_RECALL and float(full_record["avg_latency_us"]) <= LATENCY_LIMIT_US:
        best = full_record
    else:
        low = K
        high = candidate_count
        best: dict[str, Any] | None = None
        while low <= high:
            mid = (low + high) // 2
            record = evaluate(mid)
            if float(record["recall"]) >= TARGET_RECALL:
                best = record
                high = mid - 1
            else:
                low = mid + 1
        if best is None:
            best = full_record
    result = {
        "metadata": metadata,
        "evaluations": sorted(evaluations, key=lambda item: int(item["prefilter_rerank_l"])),
        "best": best,
        "pass": float(best.get("recall") or 0.0) >= TARGET_RECALL
        and float(best.get("avg_latency_us") or float("inf")) <= LATENCY_LIMIT_US,
    }
    write_json(result_json, result)
    return result


def write_truthset_first_row(source: Path, dest: Path) -> None:
    with source.open("rb") as reader:
        header = reader.read(phe.BIN_HEADER.size)
        npts, dim = phe.BIN_HEADER.unpack(header)
        if npts <= 0:
            raise ValueError(f"empty truthset: {source}")
        row = np.fromfile(reader, dtype=np.uint32, count=dim)
    ensure_dir(dest.parent)
    with dest.open("wb") as writer:
        writer.write(phe.BIN_HEADER.pack(1, dim))
        row.tofile(writer)


def make_single_query_case(name: str, query_bin: Path, query_labels: Path, truthset: Path) -> tuple[Path, Path, Path]:
    case_dir = EXP_ROOT / "memory" / name
    single_qbin = case_dir / "query.bin"
    single_qspmat = case_dir / "query.spmat"
    single_truth = case_dir / "truthset.bin"
    if single_qbin.exists() and single_qspmat.exists() and single_truth.exists():
        return single_qbin, single_qspmat, single_truth
    _, _, rows = phe.load_bin_matrix(query_bin, "float")
    phe.write_bin_subset(single_qbin, rows, [0])
    matrix = phe.SpmatMatrix.load(query_labels)
    phe.write_spmat_subset(single_qspmat, matrix, [0])
    write_truthset_first_row(truthset, single_truth)
    return single_qbin, single_qspmat, single_truth


def measure_memory_case(
    case_name: str,
    index_prefix: Path,
    selector_type: str,
    query_bin: Path,
    query_labels: Path,
    truthset: Path,
    rerank_l: int,
) -> dict[str, Any]:
    out_dir = EXP_ROOT / "memory" / case_name
    result_json = out_dir / "memory.json"
    if result_json.exists():
        return read_json(result_json)
    jsonl_path = out_dir / "search.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()
    cmd = [
        "/usr/bin/time",
        "-v",
        str(REPO_ROOT / "build" / "tests" / "search_disk_index_hybrid"),
        "float",
        str(index_prefix),
        str(THREADS),
        str(BEAMWIDTH),
        str(query_bin),
        str(truthset),
        str(K),
        "l2",
        "pq",
        selector_type,
        str(query_labels),
        "prefilter",
        "0",
        "0",
        str(SEARCH_L),
        "--jsonl-output",
        str(jsonl_path),
    ]
    result = run_command(
        f"memory_{case_name}",
        cmd,
        log_path=out_dir / "time.log",
        timeout=3600,
        env_overrides={"PIPEANN_PQ_MMAP": "1", "PIPEANN_PREFILTER_RERANK_L": str(rerank_l)},
    )
    match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", result.stderr)
    if not match:
        raise RuntimeError(f"failed to parse MaxRSS for {case_name}")
    search_record = extract_search_record(jsonl_path)
    payload = {
        "case": case_name,
        "selector_type": selector_type,
        "prefilter_rerank_l": int(rerank_l),
        "max_rss_kb": int(match.group(1)),
        "search_record": search_record,
    }
    write_json(result_json, payload)
    return payload


def measure_memory(stage: dict[str, Any], index_prefix: Path, equality_calibration: Path,
                   range_result: dict[str, Any]) -> dict[str, Any]:
    output_json = EXP_ROOT / "memory" / "summary.json"
    if output_json.exists():
        existing = read_json(output_json)
        if "failing_cases" in existing:
            return existing
        output_json.unlink()
    calibration = read_json(equality_calibration)
    overrides = {str(k): int(v) for k, v in calibration["overrides"].items()}

    cases: list[dict[str, Any]] = []
    for bucket in ("u1e-03", "u100"):
        bootstrap_runtime_metadata(stage, index_prefix, "intersect")
        bucket_dir = equality_calibration.parent / bucket
        qbin, qspmat, truth = make_single_query_case(
            f"equality_{bucket}",
            bucket_dir / "queries.bin",
            bucket_dir / "queries.spmat",
            bucket_dir / "truthset.bin",
        )
        cases.append(
            measure_memory_case(
                f"equality_{bucket}",
                index_prefix,
                "intersect",
                qbin,
                qspmat,
                truth,
                overrides[bucket],
            )
        )

    range_best = range_result["best"]
    range_metadata = range_result["metadata"]
    bootstrap_runtime_metadata(stage, index_prefix, "range")
    qbin, qspmat, truth = make_single_query_case(
        "range_0_2",
        Path(range_metadata["query_bin"]),
        Path(range_metadata["query_labels"]),
        Path(range_metadata["truthset"]),
    )
    cases.append(
        measure_memory_case(
            "range_0_2",
            index_prefix,
            "range",
            qbin,
            qspmat,
            truth,
            int(range_best["prefilter_rerank_l"]),
        )
    )

    min_case = min(cases, key=lambda item: int(item["max_rss_kb"]))
    failing_cases = [
        item["case"]
        for item in cases
        if int(item["max_rss_kb"]) > MEMORY_LIMIT_KB
    ]
    payload = {
        "cases": cases,
        "min_single_query_max_rss_kb": int(min_case["max_rss_kb"]),
        "min_case": min_case["case"],
        "limit_kb": MEMORY_LIMIT_KB,
        "failing_cases": failing_cases,
        "pass": not failing_cases,
    }
    write_json(output_json, payload)
    return payload


def compute_bloat(stage: dict[str, Any], index_prefix: Path) -> dict[str, Any]:
    suffixes = (
        "_disk.index",
        "_pq_compressed.bin",
        "_pq_pivots.bin",
        "_labels.densebit",
        "_disk.index.tags",
        "_hybrid.meta",
    )
    files: dict[str, int] = {}
    for suffix in suffixes:
        path = Path(f"{index_prefix}{suffix}")
        files[suffix] = path.stat().st_size if path.exists() else 0
    raw_vector_bytes = int(stage["npoints"]) * 128 * 4
    total_index_bytes = sum(files.values())
    payload = {
        "npoints": int(stage["npoints"]),
        "dim": 128,
        "raw_vector_bytes": raw_vector_bytes,
        "files": files,
        "total_index_bytes": total_index_bytes,
        "total_to_raw_ratio": total_index_bytes / raw_vector_bytes,
        "extra_over_raw_ratio": (total_index_bytes - raw_vector_bytes) / raw_vector_bytes,
        "pass": (total_index_bytes - raw_vector_bytes) / raw_vector_bytes <= 1.0,
    }
    write_json(EXP_ROOT / "bloat" / "summary.json", payload)
    return payload


def load_dynamic_results(dynamic_runs: list[dict[str, Any]]) -> dict[str, Any]:
    transitions: list[dict[str, Any]] = []
    for run in dynamic_runs:
        insert_summary_path = Path(run["insert_summary"])
        probe_jsonl_path = Path(run["probe_jsonl"])
        files_present = insert_summary_path.exists() and probe_jsonl_path.exists()
        summary = read_json(insert_summary_path) if insert_summary_path.exists() else {}
        probes = read_jsonl(probe_jsonl_path) if probe_jsonl_path.exists() else []
        probe_validations: list[dict[str, Any]] = []
        for probe in probes:
            recall = probe.get("recall")
            avg_latency_us = float(probe.get("avg_latency_us", float("inf")))
            latency_pass = avg_latency_us <= LATENCY_LIMIT_US
            recall_pass = recall is not None and float(recall) >= TARGET_RECALL
            probe_validations.append(
                {
                    "bucket_name": probe.get("bucket_name"),
                    "avg_latency_us": avg_latency_us,
                    "latency_pass": latency_pass,
                    "recall": recall,
                    "recall_pass": recall_pass,
                    "pass": latency_pass and recall_pass,
                }
            )
        transition_pass = (
            files_present
            and bool(probes)
            and bool(summary.get("probe_started_near_insert_begin"))
            and all(item["pass"] for item in probe_validations)
        )
        transitions.append(
            {
                **run,
                "summary": summary,
                "probes": probes,
                "probe_validations": probe_validations,
                "pass": transition_pass,
            }
        )
    payload = {
        "transitions": transitions,
        "pass": all(item["pass"] for item in transitions),
    }
    write_json(EXP_ROOT / "dynamic" / "summary.json", payload)
    return payload


def collect_equality_results(index_prefix: Path, calibration_path: Path) -> dict[str, Any]:
    payload = read_json(calibration_path)
    results: list[dict[str, Any]] = []
    for record in payload.get("results", []):
        bucket = record["bucket_name"]
        rerank_l = int(record["prefilter_rerank_l"])
        bucket_dir = calibration_path.parent / bucket
        prefilter_record = extract_search_record(bucket_dir / f"rerank_{rerank_l}.jsonl")
        auto_record = run_hybrid_search(
            f"equality_auto_{bucket}",
            index_prefix,
            bucket_dir / "queries.bin",
            bucket_dir / "truthset.bin",
            "intersect",
            bucket_dir / "queries.spmat",
            "auto",
            rerank_l,
            bucket_dir / "auto.jsonl",
            bucket_dir / "auto.log",
        )
        results.append(
            {
                **record,
                "achieved_avg_latency_us": float(auto_record.get("avg_latency_us", float("inf"))),
                "achieved_qps": float(auto_record.get("qps", 0.0)),
                "achieved_recall": float(auto_record.get("recall", 0.0)),
                "prefilter_calibration_record": prefilter_record,
                "search_record": auto_record,
            }
        )
    output = {
        "results": results,
        "overrides": payload.get("overrides", {}),
        "pass": all(
            float(record.get("achieved_recall") or 0.0) >= TARGET_RECALL
            and float(record.get("achieved_avg_latency_us") or float("inf")) <= LATENCY_LIMIT_US
            for record in results
        ),
    }
    write_json(EXP_ROOT / "equality_calibration" / "summary.json", output)
    return output


def main() -> int:
    ensure_dir(EXP_ROOT)
    ensure_no_competing_processes()

    manifest = read_json(SOURCE_MANIFEST)
    ensure_stage_1m_base(manifest)
    stage_500k = stage_by_name(manifest, "stage_500k")
    stage_750k = stage_by_name(manifest, "stage_750k")
    stage_1m = stage_by_name(manifest, "stage_1m")

    ensure_stage_workloads(manifest, stage_750k)
    ensure_stage_workloads(manifest, stage_1m)

    source_500k = existing_runtime_prefix(stage_500k)
    if not Path(f"{source_500k}_disk.index").exists():
        raise FileNotFoundError(f"missing stage_500k runtime prefix: {source_500k}")

    dynamic_runs = [
        run_dynamic_transition(stage_500k, stage_750k, source=source_500k),
        run_dynamic_transition(stage_750k, stage_1m, source=source_prefix(stage_750k)),
    ]
    dynamic_summary = load_dynamic_results(dynamic_runs)

    runtime_prefix = prepare_runtime_prefix(stage_1m)
    equality_calibration = calibrate_equality(stage_1m, runtime_prefix)
    equality_summary = collect_equality_results(runtime_prefix, equality_calibration)
    range_summary = evaluate_range(stage_1m, runtime_prefix)
    memory_summary = measure_memory(stage_1m, runtime_prefix, equality_calibration, range_summary)
    bloat_summary = compute_bloat(stage_1m, runtime_prefix)

    summary = {
        "format": "pipeann.codex_req_validation.v1",
        "experiment_root": str(EXP_ROOT),
        "hardware": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "platform": platform.platform(),
        },
        "dataset": {
            "name": "SIFT1M",
            "stage": "stage_1m",
            "npoints": int(stage_1m["npoints"]),
            "dim": int(manifest["dim"]),
            "pq_bits": PQ_BITS,
        },
        "acceptance": {
            "target_recall_at_10": TARGET_RECALL,
            "single_thread_avg_latency_limit_us": LATENCY_LIMIT_US,
            "single_query_memory_limit_kb": MEMORY_LIMIT_KB,
            "extra_bloat_limit_ratio": 1.0,
        },
        "equality": equality_summary,
        "range": range_summary,
        "dynamic": dynamic_summary,
        "memory": memory_summary,
        "bloat": bloat_summary,
    }
    summary["overall_pass"] = bool(
        equality_summary["pass"]
        and range_summary["pass"]
        and dynamic_summary["pass"]
        and memory_summary["pass"]
        and bloat_summary["pass"]
    )
    write_json(SUMMARY_JSON, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["overall_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
