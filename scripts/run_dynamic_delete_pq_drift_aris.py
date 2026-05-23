#!/usr/bin/env python3
"""ARIS runner for dynamic delete/merge/PQ-drift experiments.

This script is intentionally orchestration-only. It records provenance, applies
a CPU core cap, and calls PipeANN binaries/driver modes that emit raw JSONL.
Run it only after the corresponding driver changes have been reviewed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import re


BUCKETS = [
    "u1e-03",
    "u3e-03",
    "u1e-02",
    "u5e-02",
    "u1e-01",
    "u25",
    "u30",
    "u50",
    "u75",
    "u100",
]

SELECTIVITY = {
    "u1e-03": 0.001,
    "u3e-03": 0.003,
    "u1e-02": 0.01,
    "u5e-02": 0.05,
    "u1e-01": 0.1,
    "u25": 0.25,
    "u30": 0.30,
    "u50": 0.50,
    "u75": 0.75,
    "u100": 1.00,
}

DEFAULT_L_SWEEP = [50, 75, 100, 150, 200, 250, 300]

DRIVER_CONTRACT: dict[str, Any] = {
    "contract_version": 1,
    "duplicate_arg_policy": "Driver must reject duplicate scalar CLI args; the runner avoids emitting duplicates.",
    "delete_id_semantics": {
        "phase1_phase2_delete_id_file": "One uint32 tag/id per line, interpreted as current live external tag when tags are enabled.",
        "phase3_delete_scope": "`current-live-tags` means sample uniformly without replacement from all tags live at the start of that cycle.",
    },
    "live_corpus_contract": {
        "live_data_bin": "PipeANN save_bin float matrix. Row i is the exact vector for live_tag_file row i.",
        "live_base_label_file": "spmat with nrow equal to live_data_bin npoints. Row i contains labels for live_tag_file row i.",
        "live_tag_file": "PipeANN save_bin uint32/TagT vector with nrow equal to live_data_bin npoints.",
        "gt_scope": "All post-cycle GT must be computed against these live files, not original base files.",
    },
    "required_modes": {
        "measure-delete-only": {
            "purpose": "Load an existing disk index, lazy-delete a supplied tag/id set, do not call final_merge.",
            "required_args": ["--source-prefix", "--jsonl-output", "--delete-id-file", "--delete-count"],
        },
        "measure-delete-then-merge": {
            "purpose": "Lazy-delete a supplied tag/id set and then materialize through final_merge into --dest-prefix.",
            "required_args": ["--source-prefix", "--dest-prefix", "--jsonl-output", "--delete-id-file", "--delete-count"],
        },
        "delete-batch": {
            "purpose": "Phase3 composition step: lazy-delete supplied tags and final_merge to an intermediate prefix.",
            "required_args": ["--source-prefix", "--dest-prefix", "--jsonl-output", "--delete-id-file", "--delete-count"],
        },
        "insert-only": {
            "purpose": "Phase3 composition step: insert replacement vectors using the supplied tag file and final_merge to the next-cycle prefix.",
            "required_args": ["--source-prefix", "--dest-prefix", "--jsonl-output", "--data-bin", "--insert-tag-file", "--insert-count"],
        },
        "cycle-delete-insert": {
            "purpose": "Runner-composed cycle: delete-batch+merge to an intermediate index, then insert-only+merge equal-count replacement vectors into the deleted tag set and emit live-corpus files for GT.",
            "implementation_status": "implemented_as_runner_composition_of_delete-batch_and_insert-only",
            "required_args": [
                "--source-prefix",
                "--dest-prefix",
                "--jsonl-output",
                "--delete-id-file",
                "--insert-tag-file",
                "--data-bin",
                "--insert-count",
            ],
        },
        "pq-drift": {
            "purpose": "Compare direct-build PQ with zero-data incremental PQ using seed-trained pivots and no full-corpus retrain, plus optional retrain cost proxies from build logs.",
            "implementation_status": "implemented_smoke_for_direct_vs_zero_insert_seed_pq_no_retrain",
            "required_args": ["--jsonl-output", "--data-bin", "--base-label-file", "--query-bin"],
        },
        "zero-insert-only": {
            "purpose": "Driver mode used by Phase4: insert from an empty flat index, materialize once threshold is crossed, optionally using seed-trained PQ pivots.",
            "required_args": [
                "--source-prefix",
                "--jsonl-output",
                "--data-bin",
                "--insert-count",
                "--flat-threshold",
                "--pq-bytes",
                "--flat-pq-pivots",
            ],
        },
        "measure-dynamic-search": {
            "purpose": "Run search for a fixed route/L/query/GT/selector and append full route, recall, latency, and RSS stats.",
            "required_args": [
                "--source-prefix",
                "--jsonl-output",
                "--query-bin",
                "--truthset-bin",
                "--query-label-file",
                "--selector-type",
                "--route",
                "--search-l",
            ],
        },
    },
    "required_json_fields": [
        "mode",
        "phase",
        "status",
        "cpu_cap",
        "cpu_cap_enforced",
        "threads",
        "source_prefix",
        "dest_prefix",
        "delete_count",
        "deleted_tag_hash",
        "delete_scope",
        "insert_count",
        "insert_segment",
        "live_point_count",
        "live_data_bin",
        "live_base_label_file",
        "live_tag_file",
        "live_gt_scope",
        "route",
        "actual_route",
        "search_l",
        "recall@10",
        "avg_latency_us",
        "p95_latency_us",
        "candidate_count",
        "prefilter_count",
        "graph_count",
        "fallback_count",
        "tau_m",
        "threshold_version",
        "delete_wall_s",
        "merge_wall_s",
        "insert_wall_s",
        "wall_s",
        "max_rss_kb",
        "pq_bytes",
        "pq_codebook_hash",
        "pq_code_hash",
        "pq_retrained",
        "pq_train_core_count",
        "pq_train_wall_s",
        "pq_recode_wall_s",
        "pq_training_points",
        "requested_points",
        "code_point_count",
        "code_chunks",
        "point_count_consistent",
        "label_storage_mode",
        "disk_format_version",
        "main_index_label_size",
        "raw_command",
    ],
    "mode_required_json_fields": {
        "measure-delete-only": [
            "mode",
            "status",
            "cpu_cap",
            "cpu_cap_enforced",
            "cpu_affinity_allowed_cpus",
            "source_prefix",
            "delete_count",
            "deleted_tag_hash",
            "delete_scope",
            "live_point_count",
            "delete_wall_s",
            "raw_command",
        ],
        "measure-delete-then-merge": [
            "mode",
            "status",
            "cpu_cap",
            "cpu_cap_enforced",
            "cpu_affinity_allowed_cpus",
            "source_prefix",
            "dest_prefix",
            "delete_count",
            "deleted_tag_hash",
            "delete_scope",
            "live_point_count",
            "delete_wall_s",
            "merge_wall_s",
            "main_index_label_size",
            "label_storage_mode",
            "raw_command",
        ],
        "cycle-delete-insert": [
            "delete_step",
            "insert_step",
            "mode",
            "status",
            "cpu_cap",
            "cpu_cap_enforced",
            "cpu_affinity_allowed_cpus",
            "source_prefix",
            "dest_prefix",
            "delete_count",
            "deleted_tag_hash",
            "delete_scope",
            "insert_count",
            "insert_segment",
            "live_point_count",
            "live_data_bin",
            "live_base_label_file",
            "live_tag_file",
            "live_gt_scope",
            "raw_command",
        ],
        "delete-batch": [
            "mode",
            "status",
            "cpu_cap",
            "cpu_cap_enforced",
            "cpu_affinity_allowed_cpus",
            "source_prefix",
            "dest_prefix",
            "delete_count",
            "deleted_tag_hash",
            "delete_scope",
            "delete_elapsed_s",
            "merge_elapsed_s",
            "raw_command",
        ],
        "insert-only": [
            "mode",
            "status",
            "cpu_cap",
            "cpu_cap_enforced",
            "cpu_affinity_allowed_cpus",
            "source_prefix",
            "dest_prefix",
            "insert_count",
            "insert_scope",
            "inserted_tag_hash",
            "insert_elapsed_s",
            "merge_elapsed_s",
            "live_point_count",
            "raw_command",
        ],
        "pq-drift": [
            "mode",
            "status",
            "cpu_cap",
            "cpu_cap_enforced",
            "requested_points",
            "insert_count",
            "live_point_count",
            "code_point_count",
            "code_chunks",
            "point_count_consistent",
            "pq_bytes",
            "pq_codebook_hash",
            "pq_code_hash",
            "pq_retrained",
            "pq_train_core_count",
            "pq_train_wall_s",
            "pq_recode_wall_s",
            "pq_training_points",
            "pq_training_corpus_points",
            "seed_points",
            "flat_threshold",
            "variant",
            "final_index_prefix",
            "raw_command",
        ],
        "zero-insert-only": [
            "mode",
            "status",
            "cpu_cap",
            "cpu_cap_enforced",
            "source_prefix",
            "final_index_prefix",
            "insert_count",
            "insert_wall_s",
            "merge_wall_s",
            "live_point_count",
            "pq_bytes",
            "flat_threshold",
            "flat_pq_pivots",
            "main_index_label_size",
            "label_sidecar_loadable",
            "raw_command",
        ],
        "measure-dynamic-search": [
            "mode",
            "status",
            "cpu_cap",
            "cpu_cap_enforced",
            "source_prefix",
            "route",
            "actual_route",
            "search_l",
            "recall@10",
            "avg_latency_us",
            "p95_latency_us",
            "candidate_count",
            "prefilter_count",
            "graph_count",
            "raw_command",
        ],
    },
    "field_schema": {
        "cpu_cap": "int cores requested by runner",
        "cpu_cap_enforced": "bool true only if taskset/numactl or equivalent affinity was applied",
        "cpu_affinity_allowed_cpus": "string or array from sched_getaffinity/proc status in child process",
        "delete_wall_s": "float seconds spent inside lazy-delete loop only",
        "merge_wall_s": "float seconds spent inside final_merge/merge_deletes only",
        "insert_wall_s": "float seconds spent inserting vectors only",
        "avg_latency_us": "float microseconds per query",
        "p95_latency_us": "float microseconds per query",
        "candidate_count": "float or int mean candidates for this selector/query batch",
        "main_index_label_size": "int bytes of label payload embedded in main node record; must be 0 for sidecar-only claim",
    },
}


@dataclass(frozen=True)
class Paths:
    repo: Path
    out: Path
    raw: Path
    logs: Path
    evidence: Path
    data: Path
    labels: Path
    truth: Path
    indexes: Path


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def mkdirs(paths: Paths) -> None:
    for path in [paths.out, paths.raw, paths.logs, paths.evidence, paths.data, paths.labels, paths.truth, paths.indexes]:
        path.mkdir(parents=True, exist_ok=True)


def as_repo_path(repo: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo / path


def sha256_file(path: Path, limit_bytes: int | None = None) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        remaining = limit_bytes
        while True:
            if remaining is not None and remaining <= 0:
                break
            chunk_size = 1024 * 1024 if remaining is None else min(1024 * 1024, remaining)
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    digest = h.hexdigest()
    if limit_bytes is not None:
        return f"sha256_first_{limit_bytes}_bytes:{digest}"
    return f"sha256:{digest}"


def file_record(path: Path, role: str) -> dict[str, Any]:
    if not path.exists():
        return {"role": role, "path": str(path), "exists": False}
    stat = path.stat()
    # Large SIFT files are expensive to hash fully; record full hashes for
    # metadata-sized files and prefix hashes for large vector binaries.
    limit = None if stat.st_size <= 512 * 1024 * 1024 else 512 * 1024 * 1024
    return {
        "role": role,
        "path": str(path),
        "exists": True,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "hash": sha256_file(path, limit),
    }


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def read_bin_header(path: Path) -> tuple[int, int]:
    with path.open("rb") as reader:
        raw = reader.read(8)
    if len(raw) != 8:
        raise ValueError(f"failed to read PipeANN bin header from {path}")
    return struct.unpack("ii", raw)


def read_pq_code_header(path: Path) -> tuple[int, int]:
    return read_bin_header(path)


def write_identity_tag_bin(path: Path, npoints: int) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as writer:
        writer.write(struct.pack("ii", npoints, 1))
        for start in range(0, npoints, 1_000_000):
            end = min(npoints, start + 1_000_000)
            payload = bytearray()
            for value in range(start, end):
                payload += struct.pack("I", value)
            writer.write(payload)


def copy_bin_segment_with_wrap(source: Path, dest: Path, start: int, count: int, npoints: int, dim: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    vector_bytes = dim * 4
    chunk_vectors = 32_768
    with source.open("rb") as src, dest.open("wb") as dst:
        dst.write(struct.pack("ii", count, dim))
        remaining = count
        cursor = start % npoints
        while remaining > 0:
            contiguous = min(remaining, npoints - cursor, chunk_vectors)
            src.seek(8 + cursor * vector_bytes)
            payload = src.read(contiguous * vector_bytes)
            if len(payload) != contiguous * vector_bytes:
                raise ValueError(f"unexpected EOF while copying vector segment from {source}")
            dst.write(payload)
            remaining -= contiguous
            cursor = (cursor + contiguous) % npoints


def copy_fvecs_segment(source: Path, dest: Path, start: int, count: int) -> tuple[int, int]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src:
        dim_raw = src.read(4)
        if len(dim_raw) != 4:
            raise ValueError(f"failed to read fvecs dimension from {source}")
        dim = struct.unpack("i", dim_raw)[0]
        record_bytes = 4 + dim * 4
        file_size = source.stat().st_size
        if file_size % record_bytes != 0:
            raise ValueError(f"fvecs file size is not a multiple of record size: {source}")
        npoints = file_size // record_bytes
        if start + count > npoints:
            raise ValueError(f"requested fvecs segment exceeds {source}: start={start}, count={count}, n={npoints}")
        with dest.open("wb") as dst:
            dst.write(struct.pack("ii", count, dim))
            src.seek(start * record_bytes)
            for _ in range(count):
                local_dim = struct.unpack("i", src.read(4))[0]
                if local_dim != dim:
                    raise ValueError(f"inconsistent fvecs dimension in {source}")
                payload = src.read(dim * 4)
                if len(payload) != dim * 4:
                    raise ValueError(f"unexpected EOF while reading {source}")
                dst.write(payload)
    return int(npoints), int(dim)


def materialize_insert_segment(paths: Paths, args: argparse.Namespace, cycle_idx: int, count: int) -> Path:
    dest = paths.data / f"cycle_{cycle_idx:02d}_insert_vectors.bin"
    if dest.exists():
        return dest
    sift100m = as_repo_path(paths.repo, args.sift100m_bin)
    if sift100m.exists():
        start = cycle_idx * args.npoints
        if sift100m.suffix == ".fvecs":
            copy_fvecs_segment(sift100m, dest, start, count)
        else:
            total, dim = read_bin_header(sift100m)
            if start + count > total:
                raise ValueError(f"SIFT segment exceeds {sift100m}: start={start}, count={count}, n={total}")
            copy_bin_segment_with_wrap(sift100m, dest, start, count, total, dim)
        return dest
    if not args.allow_sift1m_segment_fallback:
        raise FileNotFoundError(
            f"SIFT100M source not found at {sift100m}; pass --allow-sift1m-segment-fallback for smoke-only reuse of base_bin"
        )
    base_bin = as_repo_path(paths.repo, args.base_bin)
    total, dim = read_bin_header(base_bin)
    start = (cycle_idx * count) % total
    copy_bin_segment_with_wrap(base_bin, dest, start, count, total, dim)
    append_jsonl(paths.raw / "phase3_warnings.jsonl", {
        "cycle": cycle_idx,
        "warning": "SIFT100M missing; used wrapped SIFT1M base vectors for smoke-only replacement segment",
        "segment_start": start,
        "segment_count": count,
    })
    return dest


def load_segment_replacements(segment_bin: Path, tags: list[int]) -> dict[int, bytes]:
    count, dim = read_bin_header(segment_bin)
    if count != len(tags):
        raise ValueError(f"segment count {count} does not match tag count {len(tags)} for {segment_bin}")
    vector_bytes = dim * 4
    with segment_bin.open("rb") as reader:
        reader.seek(8)
        payload = reader.read(count * vector_bytes)
    if len(payload) != count * vector_bytes:
        raise ValueError(f"failed to read complete segment payload from {segment_bin}")
    return {
        int(tag): payload[row * vector_bytes:(row + 1) * vector_bytes]
        for row, tag in enumerate(tags)
    }


def materialize_live_data(base_bin: Path, replacements: dict[int, bytes], dest: Path, npoints: int) -> None:
    base_npoints, dim = read_bin_header(base_bin)
    if base_npoints < npoints:
        raise ValueError(f"base bin has {base_npoints} points, expected at least {npoints}")
    vector_bytes = dim * 4
    for tag, payload in replacements.items():
        if tag < 0 or tag >= npoints:
            raise ValueError(f"replacement tag {tag} outside live corpus range 0..{npoints - 1}")
        if len(payload) != vector_bytes:
            raise ValueError(f"replacement vector for tag {tag} has {len(payload)} bytes, expected {vector_bytes}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with base_bin.open("rb") as src, dest.open("wb") as dst:
        dst.write(struct.pack("ii", npoints, dim))
        src.seek(8)
        for tag in range(npoints):
            original = src.read(vector_bytes)
            if len(original) != vector_bytes:
                raise ValueError(f"unexpected EOF in {base_bin}")
            dst.write(replacements.get(tag, original))


def write_spmat_prefix(source: Path, dest: Path, nrows: int) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader:
        header = reader.read(24)
        if len(header) != 24:
            raise ValueError(f"failed to read spmat header from {source}")
        nrow, ncol, nnz = struct.unpack("qqq", header)
        if nrows > nrow:
            raise ValueError(f"requested {nrows} rows from {source}, but only {nrow} rows exist")
        indptr_raw = reader.read((nrow + 1) * 8)
        if len(indptr_raw) != (nrow + 1) * 8:
            raise ValueError(f"failed to read spmat indptr from {source}")
        indptr = list(struct.unpack(f"{nrow + 1}q", indptr_raw))
        prefix_nnz = indptr[nrows]
        indices_raw = reader.read(nnz * 4)
        data_raw = reader.read(nnz * 4)
        if len(indices_raw) != nnz * 4 or len(data_raw) != nnz * 4:
            raise ValueError(f"failed to read spmat payload from {source}")
    prefix_indptr = [value for value in indptr[:nrows + 1]]
    with dest.open("wb") as writer:
        writer.write(struct.pack("qqq", nrows, ncol, prefix_nnz))
        writer.write(struct.pack(f"{nrows + 1}q", *prefix_indptr))
        writer.write(indices_raw[:prefix_nnz * 4])
        writer.write(data_raw[:prefix_nnz * 4])


def extract_log_seconds(log_path: Path, pattern: str) -> float | None:
    if not log_path.exists():
        return None
    match = re.search(pattern, log_path.read_text(encoding="utf-8", errors="ignore"))
    if not match:
        return None
    return float(match.group(1))


def extract_log_int(log_path: Path, pattern: str) -> int | None:
    if not log_path.exists():
        return None
    match = re.search(pattern, log_path.read_text(encoding="utf-8", errors="ignore"))
    if not match:
        return None
    return int(match.group(1))


def query_label_path(paths: Paths, args: argparse.Namespace, selector_type: str, bucket: str) -> Path:
    label_dir = as_repo_path(paths.repo, args.query_label_dir)
    candidates = []
    if selector_type == "range":
        candidates.append(label_dir / f"query_1000_range_{bucket}.spmat")
    candidates.append(label_dir / f"query_1000_{bucket}.spmat")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"missing query label for selector={selector_type}, bucket={bucket}: {candidates}")


def update_claim_status(paths: Paths, claim_id: str, status: str, evidence: list[str], note: str = "") -> None:
    registry_path = paths.out / "claim_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    for claim in registry.get("claims", []):
        if claim.get("id") == claim_id:
            claim["status"] = status
            claim["evidence"] = evidence
            if note:
                claim["note"] = note
            break
    write_json(registry_path, registry)


def cpu_prefix(cpu_cap: int) -> list[str]:
    if cpu_cap <= 0:
        return []
    cpu_range = f"0-{cpu_cap - 1}"
    if shutil.which("taskset"):
        return ["taskset", "-c", cpu_range]
    if shutil.which("numactl"):
        return ["numactl", f"--physcpubind={cpu_range}"]
    raise RuntimeError("CPU cap requested but neither taskset nor numactl is available")


def capped_env(cpu_cap: int, extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    if cpu_cap > 0:
        for key in [
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "BLIS_NUM_THREADS",
            "TBB_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ]:
            env[key] = str(cpu_cap)
        env["OMP_PROC_BIND"] = "false"
        env["OMP_PLACES"] = "cores"
    if extra:
        env.update(extra)
    return env


def run_command(
    cmd: list[str],
    *,
    cwd: Path,
    log_path: Path,
    cpu_cap: int = 0,
    env_extra: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    full_cmd = cpu_prefix(cpu_cap) + cmd
    env = capped_env(cpu_cap, env_extra)
    started = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(full_cmd) + "\n")
        log.write("cwd=" + str(cwd) + "\n")
        log.write("env_subset=" + json.dumps({k: env.get(k) for k in sorted(env) if k.startswith(("OMP_", "MKL_", "OPENBLAS_", "PIPEANN_", "NUMEXPR_", "BLIS_", "TBB_", "VECLIB_"))}, sort_keys=True) + "\n\n")
        proc = subprocess.run(full_cmd, cwd=cwd, env=env, text=True, stdout=log, stderr=subprocess.STDOUT)
        elapsed = time.time() - started
        log.write(f"\nreturncode={proc.returncode}\nelapsed_wall_s={elapsed:.6f}\n")
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed rc={proc.returncode}; see {log_path}")
    return proc


def write_claim_registry(paths: Paths) -> None:
    claims = [
        {
            "id": "C1_DELETE_IS_MARK_ONLY",
            "claim": "Current delete path marks tombstones/live state and does not immediately rewrite the main disk index.",
            "status": "PENDING",
            "evidence": [],
        },
        {
            "id": "C2_DELETE_60PCT_SUB_MS",
            "claim": "Deleting 60% of vectors has mean latency <0.9 ms/vector under the CPU cap.",
            "status": "PENDING",
            "evidence": [],
        },
        {
            "id": "C3_MERGE_MATERIALIZE_TIME",
            "claim": "Materializing marked deletes through merge has measured wall time under CPU_CAP=16.",
            "status": "PENDING",
            "evidence": [],
        },
        {
            "id": "C4_CYCLES_RECALL_RETUNED",
            "claim": "After repeated 60% delete + equal-count insert cycles, retuned route/L can reach recall@10 >= 98%.",
            "status": "PENDING",
            "evidence": [],
        },
        {
            "id": "C5_PQ_DRIFT",
            "claim": "PQ drift from zero-data insertion is quantified against direct-build PQ code, including retrain cost.",
            "status": "PENDING",
            "evidence": [],
        },
        {
            "id": "C6_LABEL_SIDECAR",
            "claim": "The v2 main disk index stores only raw vectors plus adjacency; labels live in sidecar files.",
            "status": "PENDING",
            "evidence": [],
        },
    ]
    write_json(paths.out / "claim_registry.json", {"created_utc": now_stamp(), "claims": claims})


def write_driver_contract(paths: Paths) -> None:
    write_json(paths.evidence / "driver_contract.json", DRIVER_CONTRACT)
    required = "\n".join(
        f"- `{mode}`: {spec['purpose']}"
        + (f"\n  Implementation status: {spec['implementation_status']}" if "implementation_status" in spec else "")
        + "\n  Required args: "
        + ", ".join(f"`{arg}`" for arg in spec["required_args"])
        for mode, spec in DRIVER_CONTRACT["required_modes"].items()
    )
    fields = "\n".join(f"- `{field}`" for field in DRIVER_CONTRACT["required_json_fields"])
    mode_fields = "\n".join(
        f"### `{mode}`\n" + "\n".join(f"- `{field}`" for field in fields_for_mode)
        for mode, fields_for_mode in DRIVER_CONTRACT["mode_required_json_fields"].items()
    )
    field_schema = "\n".join(f"- `{field}`: {meaning}" for field, meaning in DRIVER_CONTRACT["field_schema"].items())
    write_text(
        paths.evidence / "driver_contract.md",
        "# Dynamic Delete/PQ Drift Driver Contract\n\n"
        "The experiment runner assumes these driver modes and fields. If the C++ driver cannot emit them, "
        "the corresponding claim must be marked UNSUPPORTED rather than inferred.\n\n"
        f"Duplicate arg policy: {DRIVER_CONTRACT['duplicate_arg_policy']}\n\n"
        "ID/tag semantics:\n"
        f"- Phase1/2 delete file: {DRIVER_CONTRACT['delete_id_semantics']['phase1_phase2_delete_id_file']}\n"
        f"- Phase3 delete scope: {DRIVER_CONTRACT['delete_id_semantics']['phase3_delete_scope']}\n\n"
        "Live corpus contract:\n"
        f"- live_data_bin: {DRIVER_CONTRACT['live_corpus_contract']['live_data_bin']}\n"
        f"- live_base_label_file: {DRIVER_CONTRACT['live_corpus_contract']['live_base_label_file']}\n"
        f"- live_tag_file: {DRIVER_CONTRACT['live_corpus_contract']['live_tag_file']}\n"
        f"- GT scope: {DRIVER_CONTRACT['live_corpus_contract']['gt_scope']}\n\n"
        "## Required Modes\n\n"
        f"{required}\n\n"
        "## Global JSON Fields\n\n"
        f"{fields}\n\n"
        "## Mode-Specific Required JSON Fields\n\n"
        f"{mode_fields}\n\n"
        "## Field Schema and Units\n\n"
        f"{field_schema}\n",
    )


def write_system_inventory(paths: Paths) -> None:
    tools = {name: shutil.which(name) for name in ["taskset", "numactl", "lscpu", "nproc", "git"]}
    write_json(paths.evidence / "tool_inventory.json", tools)
    commands = [
        (["uname", "-a"], "uname.log"),
        (["lscpu"], "lscpu.log"),
        (["nproc", "--all"], "nproc.log"),
        (["bash", "-lc", "cat /proc/self/status | sed -n '/Cpus_allowed/p;/Mems_allowed/p'"], "cpuset.log"),
        (["git", "diff", "--stat"], "git_diff_stat.log"),
        (["git", "diff", "--submodule"], "git_diff.patch"),
        (["git", "submodule", "status", "--recursive"], "git_submodule_status.log"),
    ]
    for cmd, name in commands:
        run_command(cmd, cwd=paths.repo, log_path=paths.logs / f"phase0_{name}", check=False)


def phase0_inventory(paths: Paths, args: argparse.Namespace) -> None:
    write_driver_contract(paths)
    write_system_inventory(paths)
    binaries = [
        paths.repo / "build/tests/dynamic_update_suite_driver",
        paths.repo / "build/tests/build_disk_index",
        paths.repo / "build/tests/search_disk_index_hybrid",
        paths.repo / "build/tests/utils/compute_groundtruth",
        paths.repo / "build/tests/calibrate_hybrid_threshold",
    ]
    records: list[dict[str, Any]] = []
    for binary in binaries:
        records.append(file_record(binary, "binary"))
    for path, role in [
        (as_repo_path(paths.repo, args.base_bin), "base_bin"),
        (as_repo_path(paths.repo, args.query_bin), "query_bin"),
        (as_repo_path(paths.repo, args.base_labels), "base_labels"),
        (as_repo_path(paths.repo, args.sift100m_bin), "sift100m_bin"),
    ]:
        records.append(file_record(path, role))
    query_label_dir = as_repo_path(paths.repo, args.query_label_dir)
    for bucket in BUCKETS:
        records.append(file_record(query_label_dir / f"query_1000_{bucket}.spmat", f"query_label_intersect_{bucket}"))
        records.append(file_record(query_label_dir / f"query_1000_range_{bucket}.spmat", f"query_label_range_{bucket}"))

    git_log = paths.logs / "phase0_git.log"
    run_command(["git", "rev-parse", "HEAD"], cwd=paths.repo, log_path=git_log, check=False)
    run_command(["git", "status", "--short"], cwd=paths.repo, log_path=paths.logs / "phase0_git_status.log", check=False)
    run_command([str(paths.repo / "build/tests/dynamic_update_suite_driver"), "--help"],
                cwd=paths.repo, log_path=paths.logs / "phase0_driver_help.log", check=False)
    write_json(paths.evidence / "phase0_inventory.json", {
        "cpu_cap": args.cpu_cap,
        "records": records,
        "hash_note": "Files larger than 512MiB use a clearly marked prefix hash for fast provenance; full hashes should be added for final paper-grade claims if runtime permits.",
        "aris_docs": [
            "/Users/zhengganglin/Downloads/Auto-claude-code-research-in-sleep/docs/ARIS_INTRO.md",
            "/Users/zhengganglin/Downloads/Auto-claude-code-research-in-sleep/docs/TRAE_ARIS_RUNBOOK_CN.md",
        ],
    })


def build_or_reuse_index(paths: Paths, args: argparse.Namespace, prefix: Path, base_bin: Path, base_labels: Path) -> None:
    disk_index = Path(str(prefix) + "_disk.index")
    if disk_index.exists():
        append_jsonl(paths.raw / "index_builds.jsonl", {
            "event": "reuse_index",
            "prefix": str(prefix),
            "disk_index": file_record(disk_index, "disk_index"),
        })
        return
    cmd = [
        str(paths.repo / "build/tests/build_disk_index"),
        "float",
        str(base_bin),
        str(prefix),
        str(args.build_r),
        str(args.build_l),
        str(args.pq_bytes),
        str(args.memory_gb),
        str(args.cpu_cap),
        args.metric,
        args.nbr_type,
        "spmat",
        str(base_labels),
        "--calibration-selector-type",
        "intersect",
        "--calibration-threads",
        "1",
        "--calibration-beamwidth",
        str(args.beamwidth),
        "--calibration-k",
        str(args.k),
        "--calibration-mem-l",
        "0",
        "--calibration-l-search",
        "100",
        "--label-storage",
        "sidecar",
    ]
    run_command(cmd, cwd=paths.repo, log_path=paths.logs / f"build_{prefix.name}.log", cpu_cap=args.cpu_cap)
    append_jsonl(paths.raw / "index_builds.jsonl", {
        "event": "build_index",
        "prefix": str(prefix),
        "disk_index": file_record(disk_index, "disk_index"),
    })


def make_delete_ids(path: Path, npoints: int, fraction: float, seed: int) -> int:
    count = int(math.floor(npoints * fraction))
    rng = random.Random(seed)
    ids = list(range(npoints))
    rng.shuffle(ids)
    chosen = sorted(ids[:count])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for value in chosen:
            f.write(f"{value}\n")
    return count


def driver_base_cmd(paths: Paths, args: argparse.Namespace, mode: str, source: Path, jsonl: Path,
                    route: str | None = None, search_l: int | None = None) -> list[str]:
    cmd = [
        str(paths.repo / "build/tests/dynamic_update_suite_driver"),
        "--mode",
        mode,
        "--source-prefix",
        str(source),
        "--jsonl-output",
        str(jsonl),
        "--insert-threads",
        str(min(args.cpu_cap, args.insert_threads)),
        "--search-threads",
        "1",
        "--merge-threads",
        str(min(args.cpu_cap, args.merge_threads)),
        "--build-l",
        str(args.build_l),
        "--build-r",
        str(args.build_r),
        "--beamwidth",
        str(args.beamwidth),
        "--k",
        str(args.k),
        "--cpu-cap",
        str(args.cpu_cap),
        "--metric",
        args.metric,
    ]
    if route is not None:
        cmd += ["--route", route]
    if search_l is not None:
        cmd += ["--search-l", str(search_l)]
    return cmd


def phase1_delete(paths: Paths, args: argparse.Namespace) -> None:
    base_bin = as_repo_path(paths.repo, args.base_bin)
    base_labels = as_repo_path(paths.repo, args.base_labels)
    prefix = paths.indexes / "sift1m_r116_pq16_direct"
    build_or_reuse_index(paths, args, prefix, base_bin, base_labels)
    delete_ids = paths.data / "delete_ids_60pct_seeded.txt"
    delete_count = make_delete_ids(delete_ids, args.npoints, 0.60, args.seed)
    jsonl = paths.raw / "phase1_delete_only.jsonl"
    before = len(read_jsonl(jsonl))
    cmd = driver_base_cmd(paths, args, "measure-delete-only", prefix, jsonl)
    cmd += [
        "--delete-id-file",
        str(delete_ids),
        "--delete-count",
        str(delete_count),
        "--data-bin",
        str(base_bin),
        "--base-label-file",
        str(base_labels),
    ]
    run_command(cmd, cwd=paths.repo, log_path=paths.logs / "phase1_delete_only.log", cpu_cap=args.cpu_cap)
    latest_driver_row(jsonl, before, DRIVER_CONTRACT["mode_required_json_fields"]["measure-delete-only"])
    append_jsonl(paths.raw / "phase1_inputs.jsonl", {
        "delete_id_file": file_record(delete_ids, "delete_id_file"),
        "delete_count": delete_count,
        "npoints": args.npoints,
        "fraction": 0.60,
        "delete_scope": "delete_id_file_tags",
    })
    update_claim_status(
        paths,
        "C2_DELETE_60PCT_SUB_MS",
        "EVIDENCE_SMOKE",
        ["raw/phase1_delete_only.jsonl", "raw/phase1_inputs.jsonl", "logs/phase1_delete_only.log"],
        "Smoke evidence; final-grade status depends on independent ARIS review.",
    )


def phase2_merge(paths: Paths, args: argparse.Namespace) -> None:
    base_bin = as_repo_path(paths.repo, args.base_bin)
    base_labels = as_repo_path(paths.repo, args.base_labels)
    source = paths.indexes / "sift1m_r116_pq16_direct"
    build_or_reuse_index(paths, args, source, base_bin, base_labels)
    delete_ids = paths.data / "delete_ids_60pct_seeded.txt"
    delete_count = make_delete_ids(delete_ids, args.npoints, 0.60, args.seed)
    dest = paths.indexes / "sift1m_r116_pq16_after_60pct_delete_merge"
    jsonl = paths.raw / "phase2_delete_then_merge.jsonl"
    before = len(read_jsonl(jsonl))
    cmd = driver_base_cmd(paths, args, "measure-delete-then-merge", source, jsonl)
    cmd += [
        "--dest-prefix",
        str(dest),
        "--delete-id-file",
        str(delete_ids),
        "--delete-count",
        str(delete_count),
        "--data-bin",
        str(base_bin),
        "--base-label-file",
        str(base_labels),
    ]
    run_command(cmd, cwd=paths.repo, log_path=paths.logs / "phase2_delete_then_merge.log", cpu_cap=args.cpu_cap)
    latest_driver_row(jsonl, before, DRIVER_CONTRACT["mode_required_json_fields"]["measure-delete-then-merge"])
    append_jsonl(paths.raw / "phase2_outputs.jsonl", {
        "dest_prefix": str(dest),
        "disk_index": file_record(Path(str(dest) + "_disk.index"), "merged_disk_index"),
        "tags": file_record(Path(str(dest) + "_disk.index.tags"), "merged_tags"),
        "labels_densebit": file_record(Path(str(dest) + "_labels.densebit"), "merged_labels_densebit"),
        "delete_id_file": file_record(delete_ids, "delete_id_file"),
        "delete_scope": "delete_id_file_tags",
        "artifact_retention": "large merged artifacts remain in the remote experiment directory; this JSONL records size and hash for frozen provenance",
    })
    update_claim_status(
        paths,
        "C3_MERGE_MATERIALIZE_TIME",
        "EVIDENCE_SMOKE",
        ["raw/phase2_delete_then_merge.jsonl", "raw/phase2_outputs.jsonl", "logs/phase2_delete_then_merge.log"],
        "Smoke evidence; CPU affinity must be checked by independent ARIS review.",
    )
    update_claim_status(
        paths,
        "C6_LABEL_SIDECAR",
        "EVIDENCE_SMOKE",
        ["raw/phase2_delete_then_merge.jsonl", "raw/phase2_outputs.jsonl", "logs/phase2_delete_then_merge.log"],
        "Sidecar evidence requires main_index_label_size=0 and label_sidecar_loadable=true in driver output.",
    )


def compute_truth(paths: Paths, args: argparse.Namespace, base_bin: Path, query_bin: Path, base_labels: Path,
                  query_labels: Path, selector_type: str, out: Path, tag_file: Path | None = None) -> None:
    if out.exists():
        return
    cmd = [
        str(paths.repo / "build/tests/utils/compute_groundtruth"),
        "float",
        args.metric,
        str(base_bin),
        str(query_bin),
        str(args.k),
        str(out),
        str(tag_file) if tag_file is not None else "null",
        "spmat",
        selector_type,
        str(base_labels),
        str(query_labels),
    ]
    run_command(cmd, cwd=paths.repo, log_path=paths.logs / f"gt_{out.stem}.log", cpu_cap=args.cpu_cap)


def calibrate_bucket(paths: Paths, args: argparse.Namespace, prefix: Path, base_bin: Path, base_labels: Path,
                     query_bin: Path, query_labels: Path, truth: Path, selector_type: str, bucket: str,
                     cycle: str) -> dict[str, Any]:
    jsonl = paths.raw / f"calibration_{cycle}_{selector_type}_{bucket}.jsonl"
    candidates: list[dict[str, Any]] = []
    for route in ["prefilter", "graph"]:
        for search_l in DEFAULT_L_SWEEP:
            before = len(read_jsonl(jsonl))
            cmd = driver_base_cmd(paths, args, "measure-dynamic-search", prefix, jsonl,
                                  route=route, search_l=search_l)
            cmd += [
                "--data-bin",
                str(base_bin),
                "--base-label-file",
                str(base_labels),
                "--query-bin",
                str(query_bin),
                "--truthset-bin",
                str(truth),
                "--query-label-file",
                str(query_labels),
                "--selector-type",
                selector_type,
                "--query-limit",
                str(args.query_count),
            ]
            run_command(cmd, cwd=paths.repo,
                        log_path=paths.logs / f"calib_{cycle}_{selector_type}_{bucket}_{route}_L{search_l}.log",
                        cpu_cap=args.cpu_cap,
                        env_extra={"PIPEANN_PQ_MMAP": "0", "PIPEANN_PQ_MMAP_DROP_CACHE": "0"})
            rows = read_jsonl(jsonl)
            if len(rows) <= before:
                raise RuntimeError(f"calibration command did not append to {jsonl}")
            row = rows[-1]
            latest_driver_row(jsonl, before, DRIVER_CONTRACT["mode_required_json_fields"]["measure-dynamic-search"])
            row.update({"cycle": cycle, "selector_type": selector_type, "bucket": bucket,
                        "configured_route": route, "configured_L": search_l,
                        "selection_policy": "post_hoc_retuned_fastest_feasible_recall_ge_98"})
            candidates.append(row)
    passing = [r for r in candidates if float(r.get("recall@10", r.get("recall", 0.0))) >= 98.0]
    if not passing:
        selected = min(candidates, key=lambda r: float(r.get("avg_latency_us", float("inf"))))
        selected["selection_status"] = "failed_recall"
        selected["supports_recall_claim"] = False
    else:
        selected = min(passing, key=lambda r: float(r.get("avg_latency_us", float("inf"))))
        selected["selection_status"] = "pass"
        selected["supports_recall_claim"] = True
    selected["candidate_count_total"] = len(candidates)
    selected["candidate_count_passing_recall"] = len(passing)
    append_jsonl(paths.raw / "selected_route_l.jsonl", selected)
    return selected


def latest_driver_row(jsonl: Path, before_len: int, required_fields: Iterable[str]) -> dict[str, Any]:
    rows = read_jsonl(jsonl)
    if len(rows) <= before_len:
        raise RuntimeError(f"driver did not append to {jsonl}")
    row = rows[-1]
    missing = [
        field
        for field in required_fields
        if field not in row or row[field] is None or (isinstance(row[field], str) and row[field] == "")
    ]
    if missing:
        raise RuntimeError(f"driver row missing required fields {missing}; see {jsonl}")
    return row


def phase3_cycles(paths: Paths, args: argparse.Namespace) -> None:
    source = paths.indexes / "sift1m_r116_pq16_direct"
    base_bin = as_repo_path(paths.repo, args.base_bin)
    base_labels = as_repo_path(paths.repo, args.base_labels)
    query_bin = as_repo_path(paths.repo, args.query_bin)
    build_or_reuse_index(paths, args, source, base_bin, base_labels)
    cycles = args.main_cycles
    current_prefix = source
    replacements: dict[int, bytes] = {}
    live_tag_file = paths.data / "live_identity_tags.bin"
    write_identity_tag_bin(live_tag_file, args.npoints)
    phase3_buckets = [bucket for bucket in args.phase3_buckets.split(",") if bucket]
    for cycle_idx in range(1, cycles + 1):
        delete_ids = paths.data / f"cycle_{cycle_idx:02d}_delete_ids_60pct_seeded.txt"
        delete_count = make_delete_ids(delete_ids, args.npoints, 0.60, args.seed + cycle_idx)
        after_delete = paths.indexes / f"cycle_{cycle_idx:02d}_after_delete_merge"
        delete_jsonl = paths.raw / "phase3_delete_steps.jsonl"
        delete_before = len(read_jsonl(delete_jsonl))
        delete_cmd = driver_base_cmd(paths, args, "delete-batch", current_prefix, delete_jsonl)
        delete_cmd += [
            "--dest-prefix",
            str(after_delete),
            "--delete-id-file",
            str(delete_ids),
            "--delete-count",
            str(delete_count),
            "--data-bin",
            str(base_bin),
            "--base-label-file",
            str(base_labels),
        ]
        run_command(delete_cmd, cwd=paths.repo, log_path=paths.logs / f"phase3_cycle_{cycle_idx:02d}_delete_merge.log",
                    cpu_cap=args.cpu_cap, env_extra={"PIPEANN_PQ_MMAP": "0", "PIPEANN_PQ_MMAP_DROP_CACHE": "0"})
        delete_row = latest_driver_row(delete_jsonl, delete_before, [
            "mode", "status", "source_prefix", "dest_prefix", "delete_count", "deleted_tag_hash",
            "delete_scope", "delete_elapsed_s", "merge_elapsed_s", "raw_command",
        ])

        insert_segment = materialize_insert_segment(paths, args, cycle_idx, delete_count)
        dest = paths.indexes / f"cycle_{cycle_idx:02d}_after_delete_insert"
        jsonl = paths.raw / "phase3_cycles.jsonl"
        before = len(read_jsonl(jsonl))
        cmd = driver_base_cmd(paths, args, "insert-only", after_delete, jsonl)
        cmd += [
            "--dest-prefix",
            str(dest),
            "--data-bin",
            str(insert_segment),
            "--insert-start",
            "0",
            "--insert-count",
            str(delete_count),
            "--insert-tag-file",
            str(delete_ids),
            "--base-label-file",
            str(base_labels),
        ]
        run_command(cmd, cwd=paths.repo, log_path=paths.logs / f"phase3_cycle_{cycle_idx:02d}_insert_merge.log",
                    cpu_cap=args.cpu_cap, env_extra={"PIPEANN_PQ_MMAP": "0", "PIPEANN_PQ_MMAP_DROP_CACHE": "0"})
        cycle_row = latest_driver_row(jsonl, before, [
            "mode", "status", "source_prefix", "dest_prefix", "insert_count", "insert_scope",
            "inserted_tag_hash", "insert_elapsed_s", "merge_elapsed_s", "live_point_count", "raw_command",
        ])
        current_prefix = dest
        deleted_tags = [int(line) for line in delete_ids.read_text(encoding="utf-8").splitlines() if line.strip()]
        replacements.update(load_segment_replacements(insert_segment, deleted_tags))
        live_data_bin = paths.data / f"cycle_{cycle_idx:02d}_live_data_by_tag.bin"
        materialize_live_data(base_bin, replacements, live_data_bin, args.npoints)
        live_base_labels = base_labels
        append_jsonl(paths.raw / "phase3_live_corpus_inventory.jsonl", {
            "cycle": cycle_idx,
            "live_data_bin": file_record(live_data_bin, f"cycle_{cycle_idx:02d}_live_data_bin"),
            "live_base_label_file": file_record(live_base_labels, f"cycle_{cycle_idx:02d}_live_base_label_file"),
            "live_tag_file": file_record(live_tag_file, f"cycle_{cycle_idx:02d}_live_tag_file"),
            "delete_step": delete_row,
            "insert_step": cycle_row,
            "deleted_tag_hash": delete_row.get("deleted_tag_hash"),
            "inserted_tag_hash": cycle_row.get("inserted_tag_hash"),
            "delete_scope": "delete_id_file_tags",
            "insert_scope": "insert_tag_file_tags_reusing_deleted_tags_after_delete_merge",
            "insert_count": cycle_row.get("insert_count"),
            "live_gt_scope": "post_cycle_live_corpus",
            "replacement_policy": "deleted tags are reused only after delete-batch final_merge removes old nodes",
        })
        for selector_type in args.selector_types.split(","):
            for bucket in phase3_buckets:
                qlabel = query_label_path(paths, args, selector_type, bucket)
                truth = paths.truth / f"cycle_{cycle_idx:02d}_{selector_type}_{bucket}.bin"
                compute_truth(paths, args, live_data_bin, query_bin, live_base_labels, qlabel, selector_type, truth,
                              tag_file=live_tag_file)
                calibrate_bucket(paths, args, current_prefix, live_data_bin, live_base_labels, query_bin, qlabel,
                                 truth, selector_type, bucket, f"cycle{cycle_idx:02d}")
    update_claim_status(
        paths,
        "C4_CYCLES_RECALL_RETUNED",
        "EVIDENCE_SMOKE",
        ["raw/phase3_delete_steps.jsonl", "raw/phase3_cycles.jsonl", "raw/phase3_live_corpus_inventory.jsonl",
         "raw/selected_route_l.jsonl"],
        "Evidence is smoke/main depending on --main-cycles and whether SIFT100M or SIFT1M fallback was used.",
    )


def phase4_pq_drift(paths: Paths, args: argparse.Namespace) -> None:
    points = args.phase4_points
    seed_points = args.phase4_seed_points if args.phase4_seed_points is not None else args.phase4_threshold
    flat_threshold = args.phase4_flat_threshold if args.phase4_flat_threshold is not None else points - 1
    base_bin = as_repo_path(paths.repo, args.base_bin)
    base_labels = as_repo_path(paths.repo, args.base_labels)
    query_bin = as_repo_path(paths.repo, args.query_bin)
    total, dim = read_bin_header(base_bin)
    if points > total:
        raise ValueError(f"--phase4-points {points} exceeds base corpus size {total}")
    if seed_points <= 0 or seed_points >= points:
        raise ValueError("phase4 seed points must be positive and smaller than --phase4-points")
    if flat_threshold <= 0 or flat_threshold >= points:
        raise ValueError("phase4 flat threshold must be positive and smaller than --phase4-points")

    phase4_data = paths.data / f"phase4_final_{points}.bin"
    copy_bin_segment_with_wrap(base_bin, phase4_data, 0, points, total, dim)
    phase4_labels = paths.labels / f"phase4_base_{points}.spmat"
    write_spmat_prefix(base_labels, phase4_labels, points)
    phase4_tags = paths.data / f"phase4_identity_tags_{points}.bin"
    write_identity_tag_bin(phase4_tags, points)

    seed_data = paths.data / f"phase4_seed_{seed_points}.bin"
    copy_bin_segment_with_wrap(base_bin, seed_data, 0, seed_points, total, dim)
    seed_labels = paths.labels / f"phase4_seed_{seed_points}.spmat"
    write_spmat_prefix(base_labels, seed_labels, seed_points)
    seed_prefix = paths.indexes / f"phase4_seed_pq{args.pq_bytes}_{seed_points}"
    build_or_reuse_index(paths, args, seed_prefix, seed_data, seed_labels)
    seed_build_log = paths.logs / f"build_{seed_prefix.name}.log"
    seed_pivots = Path(str(seed_prefix) + "_pq_pivots.bin")
    append_jsonl(paths.raw / "phase4_corpus_inventory.jsonl", {
        "requested_points": points,
        "seed_points": seed_points,
        "flat_threshold": flat_threshold,
        "phase4_data": file_record(phase4_data, "phase4_final_data"),
        "phase4_labels": file_record(phase4_labels, "phase4_final_labels"),
        "phase4_tags": file_record(phase4_tags, "phase4_identity_tags"),
        "seed_data": file_record(seed_data, "phase4_seed_data"),
        "seed_labels": file_record(seed_labels, "phase4_seed_labels"),
        "seed_pivots": file_record(seed_pivots, "phase4_seed_pq_pivots"),
        "zero_insert_path": "flat_until_final_materialization",
        "gt_scope": "phase4_final_data because zero live count is required to match requested_points",
    })

    direct_prefix = paths.indexes / f"phase4_direct_pq{args.pq_bytes}_{points}"
    build_or_reuse_index(paths, args, direct_prefix, phase4_data, phase4_labels)
    direct_build_log = paths.logs / f"build_{direct_prefix.name}.log"

    zero_prefix = paths.indexes / (
        f"phase4_zero_insert_pq{args.pq_bytes}_seed{seed_points}_flat{flat_threshold}_{points}"
    )
    zero_jsonl = paths.raw / "phase4_zero_insert.jsonl"
    zero_before = len(read_jsonl(zero_jsonl))
    zero_cmd = driver_base_cmd(paths, args, "zero-insert-only", zero_prefix, zero_jsonl)
    zero_cmd += [
        "--data-bin",
        str(phase4_data),
        "--insert-start",
        "0",
        "--insert-count",
        str(points),
        "--flat-threshold",
        str(flat_threshold),
        "--pq-bytes",
        str(args.pq_bytes),
        "--flat-pq-pivots",
        str(seed_pivots),
        "--base-label-file",
        str(phase4_labels),
    ]
    run_command(zero_cmd, cwd=paths.repo, log_path=paths.logs / "phase4_zero_insert.log", cpu_cap=args.cpu_cap,
                env_extra={"PIPEANN_PQ_MMAP": "0", "PIPEANN_PQ_MMAP_DROP_CACHE": "0"})
    zero_row = latest_driver_row(zero_jsonl, zero_before, [
        "mode", "status", "cpu_cap", "cpu_cap_enforced", "source_prefix", "final_index_prefix",
        "insert_count", "insert_wall_s", "merge_wall_s", "live_point_count", "pq_bytes", "flat_threshold",
        "flat_pq_pivots", "main_index_label_size", "label_sidecar_loadable", "raw_command",
    ])
    zero_final_prefix = Path(zero_row["final_index_prefix"])
    if not zero_final_prefix.is_absolute():
        zero_final_prefix = paths.repo / zero_final_prefix

    drift_jsonl = paths.raw / "phase4_pq_drift.jsonl"
    for variant, prefix, retrained, train_log_path, recode_log_path, training_points, seed_pivots_path in [
        ("direct_build", direct_prefix, True, direct_build_log, direct_build_log, points, None),
        ("zero_insert_seed_pq_no_retrain", zero_final_prefix, False, seed_build_log, paths.logs / "phase4_zero_insert.log",
         seed_points, seed_pivots),
    ]:
        prefix_path = Path(prefix)
        pq_pivots = Path(str(prefix_path) + "_pq_pivots.bin")
        pq_codes = Path(str(prefix_path) + "_pq_compressed.bin")
        code_point_count, code_chunks = read_pq_code_header(pq_codes)
        live_point_count = points if retrained else int(zero_row["live_point_count"])
        train_s = extract_log_seconds(train_log_path, r"Pivots generated in ([0-9.]+)s")
        recode_s = extract_log_seconds(recode_log_path, r"Compressed data written in: ([0-9.]+)s")
        train_sample_points = extract_log_int(train_log_path, r"Generating PQ pivots with training data of size: ([0-9]+)")
        append_jsonl(drift_jsonl, {
            "mode": "pq-drift",
            "status": "ok",
            "variant": variant,
            "cpu_cap": args.cpu_cap,
            "cpu_cap_enforced": True,
            "requested_points": points,
            "points": live_point_count,
            "insert_count": None if retrained else int(zero_row["insert_count"]),
            "live_point_count": live_point_count,
            "code_point_count": code_point_count,
            "code_chunks": code_chunks,
            "point_count_consistent": live_point_count == code_point_count,
            "dim": dim,
            "pq_bytes": args.pq_bytes,
            "pq_retrained": retrained,
            "pq_train_core_count": args.cpu_cap,
            "pq_train_wall_s": train_s,
            "pq_recode_wall_s": recode_s,
            "pq_training_points": train_sample_points,
            "pq_training_corpus_points": training_points,
            "pq_training_scope": "full_final_corpus_sample" if retrained else "initial_seed_prefix",
            "seed_points": seed_points if not retrained else None,
            "flat_threshold": flat_threshold if not retrained else None,
            "zero_insert_path": "flat_until_final_materialization" if not retrained else None,
            "final_index_prefix": str(prefix_path),
            "pq_codebook_hash": file_record(pq_pivots, f"{variant}_pq_pivots"),
            "pq_code_hash": file_record(pq_codes, f"{variant}_pq_codes"),
            "seed_pq_pivots_hash": file_record(seed_pivots_path, f"{variant}_seed_pq_pivots")
            if seed_pivots_path is not None else None,
            "raw_command": "runner_composed_phase4",
        })
    inconsistent = [
        row for row in read_jsonl(drift_jsonl)
        if row.get("variant") in {"direct_build", "zero_insert_seed_pq_no_retrain"}
        and row.get("point_count_consistent") is not True
    ]
    if inconsistent:
        raise RuntimeError(f"phase4 PQ drift point-count mismatch: {inconsistent}")
    if int(zero_row["live_point_count"]) != points:
        raise RuntimeError(
            f"zero-insert final live_point_count={zero_row['live_point_count']} does not match requested points={points}"
        )

    phase4_buckets = [bucket for bucket in args.phase4_buckets.split(",") if bucket]
    for selector_type in args.selector_types.split(","):
        for bucket in phase4_buckets:
            qlabel = query_label_path(paths, args, selector_type, bucket)
            truth = paths.truth / f"phase4_{selector_type}_{bucket}.bin"
            compute_truth(paths, args, phase4_data, query_bin, phase4_labels, qlabel, selector_type, truth,
                          tag_file=phase4_tags)
            for variant, prefix in [
                ("direct_build", direct_prefix),
                ("zero_insert_seed_pq_no_retrain", zero_final_prefix),
            ]:
                selected = calibrate_bucket(paths, args, Path(prefix), phase4_data, phase4_labels, query_bin, qlabel,
                                            truth, selector_type, bucket, f"phase4_{variant}")
                append_jsonl(paths.raw / "phase4_selected_route_l.jsonl", {
                    "variant": variant,
                    "selector_type": selector_type,
                    "bucket": bucket,
                    "selected": selected,
                })

    update_claim_status(
        paths,
        "C5_PQ_DRIFT",
        "EVIDENCE_SMOKE",
        ["raw/phase4_pq_drift.jsonl", "raw/phase4_zero_insert.jsonl", "raw/phase4_selected_route_l.jsonl",
         "raw/selected_route_l.jsonl", "raw/phase4_corpus_inventory.jsonl"],
        "Smoke compares direct-build PQ16 with zero-insert PQ16 using seed-trained pivots and no full-corpus PQ retrain on --phase4-points; zero-insert stays flat until final materialization by default. Full core sweep/triggered retrain remains unsupported unless separately run.",
    )


def summarize(paths: Paths) -> None:
    summary: dict[str, Any] = {"created_utc": now_stamp(), "files": {}}
    for rel in [
        "claim_registry.json",
        "evidence/phase0_inventory.json",
        "raw/phase1_delete_only.jsonl",
        "raw/phase2_delete_then_merge.jsonl",
        "raw/phase3_cycles.jsonl",
        "raw/phase4_pq_drift.jsonl",
        "raw/phase4_zero_insert.jsonl",
        "raw/phase4_selected_route_l.jsonl",
        "raw/phase4_corpus_inventory.jsonl",
        "raw/selected_route_l.jsonl",
    ]:
        path = paths.out / rel
        summary["files"][rel] = file_record(path, rel)
    selected_rows = read_jsonl(paths.raw / "selected_route_l.jsonl")
    if selected_rows:
        total = len(selected_rows)
        pass_count = sum(1 for row in selected_rows if row.get("supports_recall_claim") is True)
        failed = total - pass_count
        by_cycle: dict[str, dict[str, int]] = {}
        for row in selected_rows:
            cycle = str(row.get("cycle", "unknown"))
            bucket = by_cycle.setdefault(cycle, {"total": 0, "pass": 0, "failed": 0})
            bucket["total"] += 1
            if row.get("supports_recall_claim") is True:
                bucket["pass"] += 1
            else:
                bucket["failed"] += 1
        summary["retuned_calibration"] = {
            "selection_policy": "post_hoc_retuned_fastest_feasible_recall_ge_98",
            "total_selected_points": total,
            "supports_recall_claim_count": pass_count,
            "failed_recall_count": failed,
            "by_cycle": by_cycle,
        }
    write_json(paths.out / "summary.json", summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/mnt/bak3/lzg/PipeANN-github"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--phase", choices=["phase0", "phase1", "phase2", "phase3", "phase4", "all"],
                        default="phase0")
    parser.add_argument("--cpu-cap", type=int, default=16)
    parser.add_argument("--build-r", type=int, default=116)
    parser.add_argument("--build-l", type=int, default=220)
    parser.add_argument("--pq-bytes", type=int, default=16)
    parser.add_argument("--memory-gb", type=int, default=64)
    parser.add_argument("--beamwidth", type=int, default=4)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--metric", default="l2")
    parser.add_argument("--nbr-type", default="pq")
    parser.add_argument("--npoints", type=int, default=1_000_000)
    parser.add_argument("--query-count", type=int, default=1000)
    parser.add_argument("--base-bin", type=Path, default=Path("data/sift1m/sift_base.bin"))
    parser.add_argument("--query-bin", type=Path, default=Path("experiments/r116_suite/data/sift_query_1000.bin"))
    parser.add_argument("--base-labels", type=Path, default=Path("experiments/r116_suite/labels/base_1m.spmat"))
    parser.add_argument("--query-label-dir", type=Path, default=Path("experiments/r116_suite/labels"))
    parser.add_argument("--sift100m-bin", type=Path, default=Path("data/sift100m/sift100m_base.fvecs"))
    parser.add_argument("--seed", type=int, default=1162026)
    parser.add_argument("--main-cycles", type=int, default=5)
    parser.add_argument("--phase3-buckets", default=",".join(BUCKETS))
    parser.add_argument("--allow-sift1m-segment-fallback", action="store_true")
    parser.add_argument("--phase4-points", type=int, default=100_000)
    parser.add_argument("--phase4-seed-points", type=int, default=None)
    parser.add_argument("--phase4-flat-threshold", type=int, default=None)
    parser.add_argument("--phase4-threshold", type=int, default=10_000)
    parser.add_argument("--phase4-buckets", default="u1e-03,u50")
    parser.add_argument("--insert-threads", type=int, default=16)
    parser.add_argument("--merge-threads", type=int, default=16)
    parser.add_argument("--selector-types", default="intersect,range")
    parser.add_argument("--pq-core-sweep", type=lambda s: [int(x) for x in s.split(",") if x],
                        default=[1, 4, 8, 16])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = args.repo / "experiments" / f"dynamic_delete_pq_drift_aris_{now_stamp()}"
    paths = Paths(
        repo=args.repo,
        out=out_dir,
        raw=out_dir / "raw",
        logs=out_dir / "logs",
        evidence=out_dir / "evidence",
        data=out_dir / "data",
        labels=out_dir / "labels",
        truth=out_dir / "truth",
        indexes=out_dir / "indexes",
    )
    mkdirs(paths)
    if not (paths.out / "claim_registry.json").exists():
        write_claim_registry(paths)

    phase0_inventory(paths, args)
    if args.phase == "all":
        raise NotImplementedError("phase=all is disabled until phase3 and phase4 driver modes are implemented")
    if args.phase in {"phase1"}:
        phase1_delete(paths, args)
    if args.phase in {"phase2"}:
        phase2_merge(paths, args)
    if args.phase in {"phase3"}:
        phase3_cycles(paths, args)
    if args.phase in {"phase4"}:
        phase4_pq_drift(paths, args)
    summarize(paths)
    print(paths.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
