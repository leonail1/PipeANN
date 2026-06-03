#!/usr/bin/env python3
"""ARIS runner for early-PQ PQ-only sidecar maintenance.

The runner validates a deliberately narrow maintenance strategy: retrain and
recode only the PQ sidecar files while leaving the disk graph, layout metadata,
tag/id map, labels, and live-record packing untouched.  Delete/insert steps use
the existing incremental merge path only to create a durable serving prefix;
full graph/layout rebuild remains a separate comparison point.
"""

from __future__ import annotations

import argparse
import csv
import errno
import hashlib
import json
import mmap
import os
import shutil
import struct
import sys
import time
from array import array
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_dynamic_delete_pq_drift_aris as aris  # noqa: E402
import run_early_pq_triggered_maintenance_aris as triggered  # noqa: E402
import run_pq_drift_1m_aris as pq1m  # noqa: E402


DEFAULT_SENTINEL_BUCKETS = ["u50", "u75", "u100"]
DEFAULT_L_SWEEP = [50, 75, 100, 150, 200, 250, 300, 400, 450, 470, 500]
NON_PQ_SUFFIXES = [
    "_disk.index",
    "_disk.index.tags",
    "_mem.index",
    "_mem.index.tags",
    "_hybrid.meta",
    "_labels.densebit",
    "_partition.bin.aligned",
    "_tau_calibration.json",
    "_tau_calibration.points.jsonl",
]
PQ_SUFFIXES = {"_pq_pivots.bin", "_pq_compressed.bin"}


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return triggered.read_jsonl(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in keys})


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def fnum(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value in (None, ""):
        return default
    return float(value)


def avg_ms(row: dict[str, Any]) -> float:
    return triggered.avg_ms(row)


def p95_ms(row: dict[str, Any]) -> float:
    return triggered.p95_ms(row)


def recall(row: dict[str, Any]) -> float:
    return triggered.recall(row)


def rel(paths: aris.Paths, path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(paths.repo))
    except ValueError:
        return str(p)


def build_paths(repo: Path, out_dir: Path | None) -> aris.Paths:
    out = out_dir if out_dir is not None else repo / "experiments" / f"v100_pq_only_maintenance_{now_stamp()}"
    if not out.is_absolute():
        out = repo / out
    if out.exists():
        raise RuntimeError(f"--out-dir already exists: {out}")
    paths = aris.Paths(
        repo=repo,
        out=out,
        raw=out / "raw",
        logs=out / "logs",
        evidence=out / "evidence",
        data=out / "data",
        labels=out / "labels",
        truth=out / "truth",
        indexes=out / "indexes",
    )
    aris.mkdirs(paths)
    return paths


def write_claim_registry(paths: aris.Paths) -> None:
    write_json(paths.out / "pq_only_claim_registry.json", {
        "created_utc": now_stamp(),
        "claims": [
            {
                "id": "P1_PQ_ONLY_DESIGN_SCOPE",
                "status": "PENDING",
                "claim": "PQ-only maintenance changes only *_pq_pivots.bin and *_pq_compressed.bin.",
                "evidence": [],
            },
            {
                "id": "P2_PQ_ONLY_SMOKE_LOADS",
                "status": "PENDING",
                "claim": "A PQ-only serving prefix loads, has unchanged disk graph content, changed PQ hashes, and searches successfully.",
                "evidence": [],
            },
            {
                "id": "P3_PQ_ONLY_5CYCLE_GATE",
                "status": "PENDING",
                "claim": "PQ-only maintenance is tested on at least five 60% delete/insert cycles with recall and latency gates.",
                "evidence": [],
            },
            {
                "id": "P4_NON_PQ_DEGRADATION_DIAGNOSIS",
                "status": "PENDING",
                "claim": "If PQ-only is insufficient, non-PQ causes are diagnosed from recall, latency, IO, graph, tombstone, and space metrics.",
                "evidence": [],
            },
            {
                "id": "P5_BACKGROUND_INTERFERENCE",
                "status": "PENDING",
                "claim": "Low-core PQ-only background maintenance is profiled separately from foreground search.",
                "evidence": [],
            },
        ],
    })


def update_claim(paths: aris.Paths, claim_id: str, status: str, evidence: list[str], note: str) -> None:
    path = paths.out / "pq_only_claim_registry.json"
    registry = json.loads(path.read_text(encoding="utf-8"))
    for claim in registry["claims"]:
        if claim["id"] == claim_id:
            claim.update({"status": status, "evidence": evidence, "note": note})
            break
    write_json(path, registry)


def write_design_doc(paths: aris.Paths, args: argparse.Namespace) -> None:
    (paths.out / "pq_only_rebuild_design.md").write_text(
        "# PQ-only Rebuild Design\n\n"
        "Scope: retrain PQ pivots and recode PQ compressed vectors only. The serving prefix hardlinks or copies "
        "the source non-PQ files, then writes fresh `*_pq_pivots.bin` and `*_pq_compressed.bin` for the destination "
        "prefix. It does not rebuild the disk graph, compact live records, rewrite layout metadata, clean extra "
        "tombstones beyond the durable incremental merge used to materialize delete/insert operations, or change "
        "tag/id maps.\n\n"
        "Critical ordering rule: PipeANN disk ids are not raw vector row ids. The runner reads "
        "`*_disk.index.tags`, materializes a temporary data bin in disk-id/tag order, and uses that temporary bin "
        "only as the PQ train/recode input. The temporary file is not part of the serving prefix; it exists so PQ "
        "code row `i` corresponds to disk node `i` while preserving all non-PQ serving files byte-for-byte.\n\n"
        "Safety invariants:\n"
        "- `pq_bytes` / `n_chunks` stays fixed.\n"
        "- PQ code rows must equal the live vector file row count.\n"
        "- Source and destination disk files must be same-inode hardlinks when possible, or byte-equivalent copies.\n"
        "- PQ sidecars are written under a fresh destination prefix, so search never observes half-written source files.\n\n"
        "Experiment caveat: the chain uses existing incremental delete/insert merge to create a durable prefix. "
        "That is not a full graph rebuild, but it means a separate no-merge in-memory experiment is needed for a "
        "pure tombstone-accumulation claim.\n\n"
        f"Configured gates: recall >= {args.recall_floor}, avg latency < {args.latency_ms} ms, "
        f"p95 latency < {args.p95_latency_ms or args.latency_ms} ms.\n",
        encoding="utf-8",
    )


def hardlink_or_copy(src: Path, dst: Path) -> dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        method = "hardlink"
    except OSError as exc:
        if exc.errno not in {errno.EXDEV, errno.EPERM, errno.EACCES}:
            raise
        shutil.copy2(src, dst)
        method = "copy"
    src_stat = src.stat()
    dst_stat = dst.stat()
    return {
        "source": str(src),
        "dest": str(dst),
        "method": method,
        "size_bytes": dst_stat.st_size,
        "same_inode": src_stat.st_ino == dst_stat.st_ino and src_stat.st_dev == dst_stat.st_dev,
    }


def discover_non_pq_suffixes(source_prefix: Path) -> list[str]:
    suffixes = set(NON_PQ_SUFFIXES)
    prefix_text = str(source_prefix)
    for path in source_prefix.parent.glob(source_prefix.name + "_*"):
        if not path.is_file():
            continue
        path_text = str(path)
        if not path_text.startswith(prefix_text):
            continue
        suffix = path_text[len(prefix_text):]
        if suffix in PQ_SUFFIXES or suffix.startswith("_shadow_"):
            continue
        suffixes.add(suffix)
    return sorted(suffixes)


def stage_non_pq_prefix(source_prefix: Path, dest_prefix: Path) -> list[dict[str, Any]]:
    staged: list[dict[str, Any]] = []
    for suffix in discover_non_pq_suffixes(source_prefix):
        src = Path(str(source_prefix) + suffix)
        if src.exists():
            staged.append(hardlink_or_copy(src, Path(str(dest_prefix) + suffix)))
    if not Path(str(dest_prefix) + "_disk.index").exists():
        raise RuntimeError(f"PQ-only prefix missing staged disk index: {dest_prefix}_disk.index")
    return staged


def fingerprint_staged_non_pq(staged: list[dict[str, Any]], role: str) -> list[dict[str, Any]]:
    fingerprints: list[dict[str, Any]] = []
    for item in staged:
        source = Path(str(item["source"]))
        dest = Path(str(item["dest"]))
        suffix = ""
        for candidate in NON_PQ_SUFFIXES:
            if str(dest).endswith(candidate):
                suffix = candidate
                break
        fingerprints.append({
            "suffix": suffix,
            "method": item.get("method"),
            "source": full_file_record(source, f"{role}_{suffix}_source"),
            "dest": full_file_record(dest, f"{role}_{suffix}_dest"),
        })
    return fingerprints


def staged_non_pq_preserved(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> bool:
    if len(before) != len(after):
        return False
    for before_item, after_item in zip(before, after):
        if before_item.get("suffix") != after_item.get("suffix"):
            return False
        before_source_hash = before_item.get("source", {}).get("hash")
        before_dest_hash = before_item.get("dest", {}).get("hash")
        after_source_hash = after_item.get("source", {}).get("hash")
        after_dest_hash = after_item.get("dest", {}).get("hash")
        if before_source_hash != before_dest_hash:
            return False
        if after_source_hash != after_dest_hash:
            return False
        if before_source_hash != after_source_hash:
            return False
    return True


def latest_or_fail(jsonl: Path, before: int, fields: list[str]) -> dict[str, Any]:
    return aris.latest_driver_row(jsonl, before, fields)


def file_hash(path: Path, role: str) -> dict[str, Any]:
    return aris.file_record(path, role)


def load_u32_bin(path: Path) -> tuple[int, int, array]:
    npoints, dim = aris.read_bin_header(path)
    values = array("I")
    if values.itemsize != 4:
        raise RuntimeError("array('I') is not 32-bit on this platform")
    with path.open("rb") as reader:
        reader.seek(8)
        values.fromfile(reader, npoints * dim)
    if sys.byteorder != "little":
        values.byteswap()
    return npoints, dim, values


def materialize_disk_order_data(live_data: Path, disk_tags: Path, dest: Path) -> dict[str, Any]:
    data_points, dim = aris.read_bin_header(live_data)
    tag_points, tag_dim, tags = load_u32_bin(disk_tags)
    if tag_dim != 1:
        raise ValueError(f"disk tag file must have dim=1: {disk_tags} dim={tag_dim}")
    if tag_points != data_points:
        raise ValueError(f"disk tag count {tag_points} != live data points {data_points}")

    seen = bytearray(data_points)
    min_tag = data_points
    max_tag = 0
    for tag in tags:
        if tag >= data_points:
            raise ValueError(f"disk tag {tag} outside live data range 0..{data_points - 1}")
        if seen[tag]:
            raise ValueError(f"duplicate disk tag {tag} in {disk_tags}")
        seen[tag] = 1
        if tag < min_tag:
            min_tag = tag
        if tag > max_tag:
            max_tag = tag

    vector_bytes = dim * 4
    dest.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with live_data.open("rb") as src, mmap.mmap(src.fileno(), 0, access=mmap.ACCESS_READ) as mapped, dest.open("wb") as dst:
        dst.write(struct.pack("ii", data_points, dim))
        for tag in tags:
            offset = 8 + int(tag) * vector_bytes
            dst.write(mapped[offset:offset + vector_bytes])

    return {
        "source_live_data": str(live_data),
        "disk_tags": str(disk_tags),
        "ordered_data": str(dest),
        "data_points": data_points,
        "data_dim": dim,
        "tag_points": tag_points,
        "tag_dim": tag_dim,
        "tag_min": int(min_tag),
        "tag_max": int(max_tag),
        "tag_unique": True,
        "wall_s": time.time() - started,
        "ordered_data_size_bytes": dest.stat().st_size,
        "input_order": "disk_index_tag_order",
    }


def full_file_record(path: Path, role: str) -> dict[str, Any]:
    if not path.exists():
        return {"role": role, "path": str(path), "exists": False}
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            hasher.update(block)
    stat = path.stat()
    return {
        "role": role,
        "path": str(path),
        "exists": True,
        "hash": "sha256:" + hasher.hexdigest(),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "inode": stat.st_ino,
        "device": stat.st_dev,
        "hash_scope": "full_file",
    }


def run_pq_only_rebuild(paths: aris.Paths, args: argparse.Namespace, source_prefix: Path, dest_prefix: Path,
                        live_data: Path, cycle_idx: int, evidence_tag: str) -> dict[str, Any]:
    staged = stage_non_pq_prefix(source_prefix, dest_prefix)
    non_pq_before = fingerprint_staged_non_pq(staged, f"{evidence_tag}_before")
    ordered_data = paths.data / "pq_only_recode_order" / f"{dest_prefix.name}_{evidence_tag}_disk_order_data.bin"
    order_record = materialize_disk_order_data(live_data, Path(str(dest_prefix) + "_disk.index.tags"), ordered_data)
    out_jsonl = paths.raw / "pq_only_sidecar_rebuild.jsonl"
    before = len(read_jsonl(out_jsonl))
    log_path = paths.logs / f"pq_only_{evidence_tag}.log"
    cmd = aris.driver_base_cmd(paths, args, "rebuild-pq-sidecar", source_prefix, out_jsonl)
    cmd += ["--dest-prefix", str(dest_prefix), "--data-bin", str(ordered_data), "--pq-bytes", str(args.pq_bytes)]
    started = time.time()
    aris.run_command(cmd, cwd=paths.repo, log_path=log_path, cpu_cap=args.maintenance_cpu_cap,
                     env_extra={"PIPEANN_PQ_MMAP": "0", "PIPEANN_PQ_MMAP_DROP_CACHE": "0"})
    row = latest_or_fail(out_jsonl, before, [
        "mode", "status", "phase", "maintenance_kind", "source_prefix", "dest_prefix",
        "data_points", "data_dim", "code_point_count", "code_chunks", "pq_sidecar_wall_s",
        "source_disk_index_points", "dest_disk_index_points",
        "raw_command",
    ])
    if int(row.get("code_chunks") or 0) != args.pq_bytes:
        raise RuntimeError(f"PQ-only rebuild returned code_chunks={row.get('code_chunks')} but expected {args.pq_bytes}")
    source_disk = Path(str(source_prefix) + "_disk.index")
    dest_disk = Path(str(dest_prefix) + "_disk.index")
    source_pivots = Path(str(source_prefix) + "_pq_pivots.bin")
    dest_pivots = Path(str(dest_prefix) + "_pq_pivots.bin")
    source_codes = Path(str(source_prefix) + "_pq_compressed.bin")
    dest_codes = Path(str(dest_prefix) + "_pq_compressed.bin")
    source_disk_stat = source_disk.stat()
    dest_disk_stat = dest_disk.stat()
    non_pq_after = fingerprint_staged_non_pq(staged, f"{evidence_tag}_after")
    non_pq_preserved = staged_non_pq_preserved(non_pq_before, non_pq_after)
    record = {
        **row,
        "cycle_idx": cycle_idx,
        "evidence_tag": evidence_tag,
        "source_prefix": str(source_prefix),
        "dest_prefix": str(dest_prefix),
        "live_data": str(live_data),
        "pq_recode_input_order": "disk_index_tag_order",
        "pq_recode_ordered_data": order_record,
        "non_pq_files_staged": staged,
        "non_pq_fingerprint_before": non_pq_before,
        "non_pq_fingerprint_after": non_pq_after,
        "non_pq_preserved": non_pq_preserved,
        "non_pq_stage_methods": sorted({item["method"] for item in staged}),
        "disk_index_same_inode": source_disk_stat.st_ino == dest_disk_stat.st_ino
        and source_disk_stat.st_dev == dest_disk_stat.st_dev,
        "disk_index_size_bytes": dest_disk_stat.st_size,
        "source_pq_pivots": file_hash(source_pivots, f"{evidence_tag}_source_pq_pivots"),
        "dest_pq_pivots": file_hash(dest_pivots, f"{evidence_tag}_dest_pq_pivots"),
        "source_pq_codes": file_hash(source_codes, f"{evidence_tag}_source_pq_codes"),
        "dest_pq_codes": file_hash(dest_codes, f"{evidence_tag}_dest_pq_codes"),
        "pq_train_wall_s": aris.extract_log_seconds(log_path, r"Pivots generated in ([0-9.]+)s"),
        "pq_recode_wall_s": aris.extract_log_seconds(log_path, r"Compressed data written in: ([0-9.]+)s"),
        "pq_training_points": aris.extract_log_int(log_path, r"Generating PQ pivots with training data of size: ([0-9]+)"),
        "pq_only_elapsed_wall_s": time.time() - started,
        "disk_graph_rebuild_performed": False,
        "tombstone_cleanup_performed": False,
        "live_record_compact_performed": False,
        "layout_rewrite_performed": False,
        "tag_id_map_rewrite_performed": False,
        "log_path": str(log_path),
    }
    record["pq_pivots_changed"] = record["source_pq_pivots"].get("hash") != record["dest_pq_pivots"].get("hash")
    record["pq_codes_changed"] = record["source_pq_codes"].get("hash") != record["dest_pq_codes"].get("hash")
    if not record["non_pq_preserved"]:
        raise RuntimeError(f"PQ-only rebuild changed non-PQ staged files for {evidence_tag}")
    append_jsonl(paths.out / "pq_only_sidecar_rebuild.jsonl", record)
    return record


def gate_reasons(row: dict[str, Any], args: argparse.Namespace) -> list[str]:
    reasons = []
    if recall(row) < args.recall_floor:
        reasons.append(f"recall<{args.recall_floor:g}")
    if avg_ms(row) >= args.latency_ms:
        reasons.append(f"avg>={args.latency_ms:g}ms")
    p95_limit = args.p95_latency_ms or args.latency_ms
    if p95_ms(row) >= p95_limit:
        reasons.append(f"p95>={p95_limit:g}ms")
    return reasons


def append_outputs(paths: aris.Paths) -> None:
    for stem in [
        "pq_only_smoke_results",
        "pq_only_dynamic_update_results",
        "pq_only_vs_full_rebuild_compare",
        "pq_only_background_interference",
        "pq_only_sidecar_rebuild",
    ]:
        write_csv(paths.out / f"{stem}.csv", read_jsonl(paths.out / f"{stem}.jsonl"))


def summarize_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    fail_rows = []
    p95_limit = args.p95_latency_ms or args.latency_ms
    for row in rows:
        reasons = []
        if recall(row) < args.recall_floor:
            reasons.append("recall")
        if avg_ms(row) >= args.latency_ms:
            reasons.append("avg_latency")
        if p95_ms(row) >= p95_limit:
            reasons.append("p95_latency")
        if reasons:
            fail_rows.append({"case_id": row.get("case_id"), "reasons": reasons})
    return {
        "rows": len(rows),
        "fail_count": len(fail_rows),
        "fail_rows": fail_rows,
        "min_recall": min((recall(row) for row in rows), default=0.0),
        "max_avg_latency_ms": max((avg_ms(row) for row in rows), default=0.0),
        "max_p95_latency_ms": max((p95_ms(row) for row in rows), default=0.0),
        "recall_pass_count": sum(1 for row in rows if recall(row) >= args.recall_floor),
        "avg_lt_10ms_count": sum(1 for row in rows if avg_ms(row) < args.latency_ms),
        "p95_lt_limit_count": sum(1 for row in rows if p95_ms(row) < p95_limit),
    }


def write_diagnosis(paths: aris.Paths, rows: list[dict[str, Any]], maintenance_rows: list[dict[str, Any]],
                    args: argparse.Namespace) -> None:
    summary = summarize_rows(rows, args)
    text = [
        "# Non-PQ Degradation Diagnosis\n",
        f"- PQ-only generated rows: `{summary['rows']}`; failures: `{summary['fail_count']}`.\n",
        f"- Max avg latency: `{summary['max_avg_latency_ms']:.3f} ms`; max p95 latency: `{summary['max_p95_latency_ms']:.3f} ms`; min recall: `{summary['min_recall']:.2f}`.\n",
        "- The PQ-only chain intentionally does not full-rebuild graph/layout. It does use incremental delete/insert merge to make each cycle durable; pure in-memory tombstone accumulation remains a separate experiment.\n",
        "- If PQ-only rows fail while full triggered rebuild rows pass, the residual causes are graph topology, incremental-layout/packing, or merge/compact policy rather than PQ drift alone.\n",
    ]
    if summary["fail_rows"]:
        text.append("\n## Failure Rows\n")
        for item in summary["fail_rows"][:20]:
            text.append(f"- `{item['case_id']}`: {item['reasons']}\n")
    if maintenance_rows:
        max_train = max(fnum(row, "pq_train_wall_s") for row in maintenance_rows)
        max_recode = max(fnum(row, "pq_recode_wall_s") for row in maintenance_rows)
        text.append(f"\n## PQ-only Costs\n- Max train `{max_train:.3f}s`; max recode `{max_recode:.3f}s`.\n")
    (paths.out / "non_pq_degradation_diagnosis.md").write_text("".join(text), encoding="utf-8")


def write_space_note(paths: aris.Paths, args: argparse.Namespace) -> None:
    (paths.out / "pq_only_index_space_layout_note.md").write_text(
        "# PQ-only Space/Layout Note\n\n"
        "PQ-only maintenance writes fresh PQ sidecars under a destination prefix and hardlinks/copies non-PQ files from "
        "the source prefix. Therefore it does not prove a new node packing or layout claim by itself. Disk graph/layout "
        "space evidence must be audited independently. The expected direct size delta is limited to replacing "
        "`*_pq_pivots.bin` and `*_pq_compressed.bin` with the same `pq_bytes`/row count.\n\n"
        f"Configured PQ bytes: `{args.pq_bytes}`.\n",
        encoding="utf-8",
    )


def load_full_rebuild_rows(repo: Path, full_artifact_dir: Path) -> list[dict[str, Any]]:
    path = full_artifact_dir if full_artifact_dir.is_absolute() else repo / full_artifact_dir
    return read_jsonl(path / "optimized_dynamic_update_results.jsonl")


def compare_with_full(paths: aris.Paths, pq_rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    full_rows = load_full_rebuild_rows(paths.repo, args.full_rebuild_artifact_dir)
    full_by_case = {str(row.get("case_id")): row for row in full_rows}
    compare_rows = []
    for row in pq_rows:
        case = str(row.get("case_id"))
        full = full_by_case.get(case.replace("pq_only_chain", "triggered_retrain_chain"))
        compare_rows.append({
            "case_id": case,
            "cycle_idx": row.get("cycle_idx"),
            "selector_type": row.get("selector_type"),
            "bucket": row.get("bucket"),
            "pq_only_recall": recall(row),
            "pq_only_avg_latency_ms": avg_ms(row),
            "pq_only_p95_latency_ms": p95_ms(row),
            "pq_only_status": "fail" if gate_reasons(row, args) else "pass",
            "full_rebuild_case_id": full.get("case_id") if full else "",
            "full_rebuild_recall": recall(full) if full else "",
            "full_rebuild_avg_latency_ms": avg_ms(full) if full else "",
            "full_rebuild_p95_latency_ms": p95_ms(full) if full else "",
        })
    for item in compare_rows:
        append_jsonl(paths.out / "pq_only_vs_full_rebuild_compare.jsonl", item)


def run_smoke(paths: aris.Paths, args: argparse.Namespace) -> None:
    baseline = triggered.normalize_repo_path(paths.repo, args.baseline_dir)
    common = triggered.load_common_from_baseline(paths.repo, baseline)
    summary = json.loads((baseline / "early_pq_train_10k_summary.json").read_text(encoding="utf-8"))
    source_prefix = triggered.normalize_repo_path(paths.repo, summary["zero_final_prefix"])
    live_data = triggered.normalize_repo_path(paths.repo, common["data0"])
    dest_prefix = paths.indexes / "smoke_pq_only"
    rebuild = run_pq_only_rebuild(paths, args, source_prefix, dest_prefix, live_data, 0, "smoke")
    selected = triggered.calibrate_one(paths, args, dest_prefix, live_data, common["labels"], common["query"],
                                       common["tags"], "range", "u75", "smoke_pq_only")
    selected.update({
        "phase": "smoke",
        "variant": "pq_only_sidecar",
        "strategy_variant": "pq_only_sidecar",
        "case_id": "smoke_pq_only_range_u75",
        "maintenance_prefix": str(dest_prefix),
        "rebuild_record": rebuild,
        "avg_latency_ms": avg_ms(selected),
        "p95_latency_ms": p95_ms(selected),
        "recall": recall(selected),
    })
    append_jsonl(paths.out / "pq_only_smoke_results.jsonl", selected)
    update_claim(paths, "P1_PQ_ONLY_DESIGN_SCOPE", "PASS",
                 ["pq_only_rebuild_design.md", "pq_only_sidecar_rebuild.jsonl"],
                 "PQ-only runner stages non-PQ files and only regenerates PQ sidecars.")
    update_claim(paths, "P2_PQ_ONLY_SMOKE_LOADS", "PASS" if not gate_reasons(selected, args) else "FAIL",
                 ["pq_only_smoke_results.jsonl", "pq_only_sidecar_rebuild.jsonl"],
                 f"Smoke selected row: recall={recall(selected):.2f}, avg={avg_ms(selected):.3f}ms, p95={p95_ms(selected):.3f}ms.")
    append_outputs(paths)
    write_json(paths.out / "summary.json", {
        "created_utc": now_stamp(),
        "phase": "smoke",
        "smoke_summary": summarize_rows([selected], args),
        "rebuild": rebuild,
    })
    reasons = gate_reasons(selected, args)
    if reasons:
        raise RuntimeError(f"PQ-only smoke failed {selected['case_id']}: {reasons}")


def run_chain(paths: aris.Paths, args: argparse.Namespace) -> None:
    baseline = triggered.normalize_repo_path(paths.repo, args.baseline_dir)
    common = triggered.load_common_from_baseline(paths.repo, baseline)
    summary = json.loads((baseline / "early_pq_train_10k_summary.json").read_text(encoding="utf-8"))
    current_prefix = triggered.normalize_repo_path(paths.repo, summary["zero_final_prefix"])
    selectors = [item for item in args.selector_types.split(",") if item]
    buckets = [item for item in args.sentinel_buckets.split(",") if item]
    replacements: dict[int, bytes] = {}
    selected_rows: list[dict[str, Any]] = []
    maintenance_rows: list[dict[str, Any]] = []

    for cycle_idx in range(1, args.chain_cycles + 1):
        cycle_started = time.time()
        delete_ids = paths.data / f"chain_cycle{cycle_idx:02d}_delete_ids_60pct.txt"
        delete_count = aris.make_delete_ids(delete_ids, args.npoints, args.delete_fraction, args.seed + cycle_idx)
        after_delete = paths.indexes / f"chain_cycle{cycle_idx:02d}_after_delete_merge"
        delete_jsonl = paths.raw / "chain_delete_steps.jsonl"
        before = len(read_jsonl(delete_jsonl))
        delete_cmd = aris.driver_base_cmd(paths, args, "delete-batch", current_prefix, delete_jsonl)
        delete_cmd += [
            "--dest-prefix", str(after_delete),
            "--delete-id-file", str(delete_ids),
            "--delete-count", str(delete_count),
            "--data-bin", str(common["data0"]),
            "--base-label-file", str(common["labels"]),
        ]
        aris.run_command(delete_cmd, cwd=paths.repo, log_path=paths.logs / f"chain_cycle{cycle_idx:02d}_delete_merge.log",
                         cpu_cap=args.cpu_cap, env_extra={"PIPEANN_PQ_MMAP": "0", "PIPEANN_PQ_MMAP_DROP_CACHE": "0"})
        delete_row = latest_or_fail(delete_jsonl, before, ["mode", "status", "delete_elapsed_s", "merge_elapsed_s",
                                                           "delete_count", "live_point_count", "raw_command"])

        insert_segment = pq1m.segment_bin(paths, common["source"], cycle_idx * args.npoints, delete_count,
                                          f"chain_cycle{cycle_idx:02d}_insert_vectors.bin")
        after_insert = paths.indexes / f"chain_cycle{cycle_idx:02d}_after_insert_incremental_merge"
        insert_jsonl = paths.raw / "chain_insert_steps.jsonl"
        before = len(read_jsonl(insert_jsonl))
        insert_cmd = aris.driver_base_cmd(paths, args, "insert-only", after_delete, insert_jsonl)
        insert_cmd += [
            "--dest-prefix", str(after_insert),
            "--data-bin", str(insert_segment),
            "--insert-start", "0",
            "--insert-count", str(delete_count),
            "--insert-tag-file", str(delete_ids),
            "--base-label-file", str(common["labels"]),
        ]
        aris.run_command(insert_cmd, cwd=paths.repo, log_path=paths.logs / f"chain_cycle{cycle_idx:02d}_insert_merge.log",
                         cpu_cap=args.cpu_cap, env_extra={"PIPEANN_PQ_MMAP": "0", "PIPEANN_PQ_MMAP_DROP_CACHE": "0"})
        insert_row = latest_or_fail(insert_jsonl, before, ["mode", "status", "insert_elapsed_s", "merge_elapsed_s",
                                                           "insert_count", "live_point_count", "raw_command"])

        deleted_tags = [int(line) for line in delete_ids.read_text(encoding="utf-8").splitlines() if line.strip()]
        replacements.update(aris.load_segment_replacements(insert_segment, deleted_tags))
        live_data = paths.data / f"chain_cycle{cycle_idx:02d}_live_data_by_tag.bin"
        aris.materialize_live_data(common["data0"], replacements, live_data, args.npoints)

        pq_prefix = paths.indexes / f"chain_cycle{cycle_idx:02d}_pq_only"
        rebuild = run_pq_only_rebuild(paths, args, after_insert, pq_prefix, live_data, cycle_idx,
                                      f"chain_cycle{cycle_idx:02d}")
        current_prefix = pq_prefix
        maintenance = {
            "cycle_idx": cycle_idx,
            "maintenance_kind": "pq_only_sidecar_recode",
            "source_prefix": str(after_insert),
            "maintenance_prefix": str(pq_prefix),
            "delete_count": delete_count,
            "delete_ms_per_vector": fnum(delete_row, "delete_elapsed_s") * 1000.0 / max(delete_count, 1),
            "delete_elapsed_s": delete_row.get("delete_elapsed_s"),
            "delete_merge_s": delete_row.get("merge_elapsed_s"),
            "insert_count": insert_row.get("insert_count"),
            "insert_elapsed_s": insert_row.get("insert_elapsed_s"),
            "insert_merge_s": insert_row.get("merge_elapsed_s"),
            "pq_train_wall_s": rebuild.get("pq_train_wall_s"),
            "pq_recode_wall_s": rebuild.get("pq_recode_wall_s"),
            "pq_only_elapsed_wall_s": rebuild.get("pq_only_elapsed_wall_s"),
            "disk_index_same_inode": rebuild.get("disk_index_same_inode"),
            "pq_pivots_changed": rebuild.get("pq_pivots_changed"),
            "pq_codes_changed": rebuild.get("pq_codes_changed"),
            "incremental_delete_insert_merge_applied": True,
            "full_graph_rebuild": False,
            "cycle_elapsed_before_search_s": time.time() - cycle_started,
        }
        maintenance_rows.append(maintenance)
        append_jsonl(paths.out / "pq_only_background_interference.jsonl", {
            "cycle_idx": cycle_idx,
            "condition": "not_overlapped_chain_timing_only",
            "maintenance_kind": "pq_only_sidecar_recode",
            "pq_train_wall_s": rebuild.get("pq_train_wall_s"),
            "pq_recode_wall_s": rebuild.get("pq_recode_wall_s"),
            "pq_only_elapsed_wall_s": rebuild.get("pq_only_elapsed_wall_s"),
            "maintenance_cpu_cap": args.maintenance_cpu_cap,
        })

        for selector in selectors:
            for bucket in buckets:
                cycle_name = f"chain_cycle{cycle_idx:02d}_pq_only"
                selected = triggered.calibrate_one(paths, args, pq_prefix, live_data, common["labels"], common["query"],
                                                   common["tags"], selector, bucket, cycle_name)
                selected.update({
                    "phase": "chain",
                    "variant": "pq_only_chain",
                    "strategy_variant": "pq_only_sidecar_recode",
                    "cycle_idx": cycle_idx,
                    "selector_type": selector,
                    "bucket": bucket,
                    "case_id": triggered.case_id(cycle_idx, "pq_only_chain", selector, bucket),
                    "serving_source": "incremental_merge_plus_pq_only_sidecar",
                    "maintenance_prefix": str(pq_prefix),
                    "avg_latency_ms": avg_ms(selected),
                    "p95_latency_ms": p95_ms(selected),
                    "recall": recall(selected),
                    "incremental_delete_insert_merge_applied": True,
                    "full_graph_rebuild": False,
                    "pq_only_rebuild": rebuild,
                })
                append_jsonl(paths.out / "pq_only_dynamic_update_results.jsonl", selected)
                selected_rows.append(selected)
                reasons = gate_reasons(selected, args)
                if reasons:
                    write_json(paths.out / "fast_fail_status.json", {
                        "created_utc": now_stamp(),
                        "case_id": selected["case_id"],
                        "reasons": reasons,
                        "row": selected,
                    })
                    append_outputs(paths)
                    compare_with_full(paths, selected_rows, args)
                    write_diagnosis(paths, selected_rows, maintenance_rows, args)
                    raise RuntimeError(f"PQ-only chain fast-fail at {selected['case_id']}: {reasons}")

        write_json(paths.out / "summary.json", {
            "created_utc": now_stamp(),
            "phase": "chain",
            "completed_cycles": cycle_idx,
            "expected_cycles": args.chain_cycles,
            "expected_selected_rows": args.chain_cycles * len(selectors) * len(buckets),
            "observed_selected_rows": len(selected_rows),
            "strategy_summary": summarize_rows(selected_rows, args),
            "maintenance_rows": len(maintenance_rows),
        })
        append_outputs(paths)

    compare_with_full(paths, selected_rows, args)
    write_diagnosis(paths, selected_rows, maintenance_rows, args)
    append_outputs(paths)
    summary_rows = summarize_rows(selected_rows, args)
    status = "PASS" if len(selected_rows) == args.chain_cycles * len(selectors) * len(buckets) and summary_rows["fail_count"] == 0 else "FAIL"
    update_claim(paths, "P3_PQ_ONLY_5CYCLE_GATE", status,
                 ["pq_only_dynamic_update_results.jsonl", "pq_only_sidecar_rebuild.jsonl", "summary.json"],
                 f"PQ-only chain summary: {summary_rows}")
    update_claim(paths, "P4_NON_PQ_DEGRADATION_DIAGNOSIS", "PASS",
                 ["non_pq_degradation_diagnosis.md", "pq_only_vs_full_rebuild_compare.jsonl"],
                 "Non-PQ diagnosis generated from PQ-only vs full rebuild comparison.")
    update_claim(paths, "P5_BACKGROUND_INTERFERENCE", "NEEDS_OVERLAP",
                 ["pq_only_background_interference.jsonl"],
                 "Chain records PQ-only timing; foreground-overlap background run still required.")


def write_final_review(paths: aris.Paths, args: argparse.Namespace) -> None:
    rows = read_jsonl(paths.out / "pq_only_dynamic_update_results.jsonl")
    smoke = read_jsonl(paths.out / "pq_only_smoke_results.jsonl")
    summary = summarize_rows(rows or smoke, args)
    (paths.out / "aris_pq_only_final_review.md").write_text(
        "# PQ-only ARIS Review\n\n"
        f"- Rows reviewed: `{summary['rows']}`.\n"
        f"- Failures: `{summary['fail_count']}`.\n"
        f"- Min recall: `{summary['min_recall']:.2f}`.\n"
        f"- Max avg latency: `{summary['max_avg_latency_ms']:.3f} ms`.\n"
        f"- Max p95 latency: `{summary['max_p95_latency_ms']:.3f} ms`.\n"
        "- Guardrail: PQ-only claims do not include graph rebuild, tombstone accumulation without merge, or layout/space repack evidence.\n",
        encoding="utf-8",
    )
    (paths.out / "ppt_ready_pq_only_summary.md").write_text(
        "# PQ-only PPT Summary\n\n"
        f"- PQ-only rows: `{summary['rows']}`, failures `{summary['fail_count']}`.\n"
        f"- Min recall `{summary['min_recall']:.2f}`, max avg `{summary['max_avg_latency_ms']:.3f} ms`, max p95 `{summary['max_p95_latency_ms']:.3f} ms`.\n"
        "- Scope: only PQ pivots/codes are regenerated; full rebuild remains separate evidence.\n",
        encoding="utf-8",
    )


def parse_l_sweep(value: str) -> list[int]:
    parsed = sorted({int(item) for item in value.split(",") if item.strip()})
    if not parsed:
        raise argparse.ArgumentTypeError("empty L sweep")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/mnt/nvme1n1/PipeANN-github"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("experiments/v100_early_pq_10k_full_20260602T031600Z"))
    parser.add_argument("--full-rebuild-artifact-dir", type=Path,
                        default=Path("experiments/v100_early_pq_triggered_artifacts_final_lf_20260603T043152Z"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--phase", choices=["smoke", "chain"], default="smoke")
    parser.add_argument("--binary-root", type=Path, default=Path("build_reviewed_20260602_querycache/tests"))
    parser.add_argument("--cpu-start", type=int, default=0)
    parser.add_argument("--cpu-cap", type=int, default=16)
    parser.add_argument("--maintenance-cpu-cap", type=int, default=16)
    parser.add_argument("--build-r", type=int, default=116)
    parser.add_argument("--build-l", type=int, default=220)
    parser.add_argument("--pq-bytes", type=int, default=16)
    parser.add_argument("--memory-gb", type=int, default=64)
    parser.add_argument("--insert-threads", type=int, default=16)
    parser.add_argument("--merge-threads", type=int, default=16)
    parser.add_argument("--search-threads", type=int, default=1)
    parser.add_argument("--beamwidth", type=int, default=4)
    parser.add_argument("--query-beamwidth", type=int, default=4)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--metric", default="l2")
    parser.add_argument("--nbr-type", default="pq")
    parser.add_argument("--npoints", type=int, default=1_000_000)
    parser.add_argument("--bigann-bin", type=Path, default=Path("data/bigann/sift_base_6m_float.bin"))
    parser.add_argument("--query-bin", type=Path, default=Path("experiments/r116_suite_pq16_aris_20260520_072453/data/sift_query_1000.bin"))
    parser.add_argument("--query-label-dir", type=Path, default=Path("experiments/r116_suite_pq16_aris_20260520_072453/labels"))
    parser.add_argument("--base-labels", type=Path, default=Path("experiments/v100_early_pq_10k_full_20260602T031600Z/labels/base_labels_1000000.spmat"))
    parser.add_argument("--seed", type=int, default=1162026)
    parser.add_argument("--delete-fraction", type=float, default=0.60)
    parser.add_argument("--sentinel-buckets", default=",".join(DEFAULT_SENTINEL_BUCKETS))
    parser.add_argument("--selector-types", default="intersect,range")
    parser.add_argument("--chain-cycles", type=int, default=5)
    parser.add_argument("--query-count", type=int, default=1000)
    parser.add_argument("--latency-ms", type=float, default=10.0)
    parser.add_argument("--p95-latency-ms", type=float, default=0.0)
    parser.add_argument("--recall-floor", type=float, default=98.0)
    parser.add_argument("--l-sweep", type=parse_l_sweep, default=DEFAULT_L_SWEEP)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.repo = args.repo.resolve()
    if args.binary_root and not args.binary_root.is_absolute():
        args.binary_root = args.repo / args.binary_root
    if args.bigann_bin and not args.bigann_bin.is_absolute():
        args.bigann_bin = args.repo / args.bigann_bin
    if args.query_bin and not args.query_bin.is_absolute():
        args.query_bin = args.repo / args.query_bin
    if args.query_label_dir and not args.query_label_dir.is_absolute():
        args.query_label_dir = args.repo / args.query_label_dir
    if args.base_labels and not args.base_labels.is_absolute():
        args.base_labels = args.repo / args.base_labels
    if args.p95_latency_ms <= 0:
        args.p95_latency_ms = args.latency_ms
    aris.CPU_START = args.cpu_start
    aris.DEFAULT_L_SWEEP[:] = args.l_sweep

    paths = build_paths(args.repo, args.out_dir)
    write_claim_registry(paths)
    write_design_doc(paths, args)
    write_space_note(paths, args)
    update_claim(paths, "P1_PQ_ONLY_DESIGN_SCOPE", "READY", ["pq_only_rebuild_design.md"],
                 "PQ-only design is written; execution evidence pending.")

    if args.phase == "smoke":
        run_smoke(paths, args)
    elif args.phase == "chain":
        run_chain(paths, args)
    else:
        raise ValueError(args.phase)

    write_final_review(paths, args)
    append_outputs(paths)
    print(paths.out.relative_to(args.repo))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
