#!/usr/bin/env python3
"""Replay selected PipeANN searches on Supersector32K packed serving snapshots.

This runner is intentionally replay-based: it consumes JSONL rows that already
contain a dynamic_update_suite_driver raw_command, repacks each v1 serving
prefix into the v3 page-aware Supersector32K layout outside the repository, and
then reruns the same search with only source-prefix/jsonl-output adjusted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SECTOR_LEN = 4096
SUPERSECTOR_BYTES = 8 * SECTOR_LEN


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
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


def resolve_repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def cmd_text(cmd: list[str]) -> str:
    return shlex.join(str(part) for part in cmd)


def run_logged(cmd: list[str], log_path: Path, *, cwd: Path = ROOT) -> tuple[float, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - started
    with log_path.open("a") as log:
        log.write("$ " + cmd_text(cmd) + "\n")
        log.write(proc.stdout)
        if proc.stdout and not proc.stdout.endswith("\n"):
            log.write("\n")
        log.write(f"[returncode] {proc.returncode}\n")
        log.write(f"[elapsed_wall_s] {elapsed:.6f}\n\n")
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {cmd_text(cmd)}; see {log_path}")
    return elapsed, proc.stdout


def run_capture(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return proc.stdout


def sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_path(path: Path) -> Path:
    return path.expanduser().resolve()


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def source_fingerprint(src_prefix: Path) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for suffix in [
        "_disk.index",
        "_pq_compressed.bin",
        "_pq_pivots.bin",
        "_labels.densebit",
        "_hybrid.meta",
        "_disk.index.tags",
        "_mem.index",
        "_mem.index.tags",
    ]:
        path = Path(str(src_prefix) + suffix)
        if not path.exists():
            continue
        stat = path.stat()
        files[suffix] = {
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": sha256_file(path) if suffix == "_disk.index" else "",
        }
    return {"src_prefix": str(src_prefix), "files": files}


def fingerprint_token(fingerprint: dict[str, Any]) -> str:
    stable = json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(stable).hexdigest()[:12]


def write_phase0(out: Path, args: argparse.Namespace, selected_path: Path, selected_rows: list[dict[str, Any]]) -> None:
    logs = out / "logs"
    evidence = out / "evidence"
    for name, cmd in {
        "phase0_git_status.log": ["git", "status", "--short"],
        "phase0_git_diff_stat.log": ["git", "diff", "--stat"],
        "phase0_git_head.log": ["git", "rev-parse", "HEAD"],
        "phase0_uname.log": ["uname", "-a"],
        "phase0_lscpu.log": ["lscpu"],
    }.items():
        (logs / name).write_text(run_capture(cmd))
    (logs / "phase0_git_diff.patch").write_text(run_capture(["git", "diff"]))
    driver = resolve_repo_path(args.binary_root) / "dynamic_update_suite_driver"
    repack = resolve_repo_path(args.repack_binary)
    write_json(
        evidence / "runner_config.json",
        {
            "repo": str(ROOT),
            "out_dir": str(out),
            "source_experiment": str(args.source_experiment),
            "selected_jsonl": str(selected_path),
            "selected_rows": len(selected_rows),
            "binary_root": str(resolve_repo_path(args.binary_root)),
            "dynamic_update_suite_driver": str(driver),
            "dynamic_update_suite_driver_sha256": sha256_file(driver),
            "repack_binary": str(repack),
            "repack_binary_sha256": sha256_file(repack),
            "index_root": str(args.index_root),
            "cpu_range": f"{args.cpu_start}-{args.cpu_start + args.cpu_cap - 1}",
            "cpu_cap": args.cpu_cap,
            "query_limit": args.query_limit,
            "force_repack": args.force_repack,
            "created_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        },
    )


def selected_payload(row: dict[str, Any]) -> dict[str, Any]:
    payload = row.get("selected")
    return payload if isinstance(payload, dict) else row


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
        raise RuntimeError(f"raw_command missing {name}: {cmd_text(cmd)}") from exc
    if idx + 1 >= len(cmd):
        raise RuntimeError(f"argument {name} has no value in raw_command")
    return cmd[idx + 1]


def packed_disk_layout_meta(disk_path: Path) -> dict[str, int] | None:
    if not disk_path.exists() or disk_path.stat().st_size < 64:
        return None
    with disk_path.open("rb") as f:
        header = f.read(96)
    nr = int.from_bytes(header[0:4], "little")
    if nr < 11 or len(header) < 96:
        return {"layout_version": 1}
    return {
        "nr": nr,
        "npoints": int.from_bytes(header[8:16], "little"),
        "data_dim": int.from_bytes(header[16:24], "little"),
        "max_node_len": int.from_bytes(header[32:40], "little"),
        "nnodes_per_sector": int.from_bytes(header[40:48], "little"),
        "label_size": int.from_bytes(header[56:64], "little"),
        "layout_version": int.from_bytes(header[64:72], "little"),
        "layout_block_bytes": int.from_bytes(header[72:80], "little"),
        "layout_nodes_per_block": int.from_bytes(header[80:88], "little"),
        "layout_read_page_bytes": int.from_bytes(header[88:96], "little"),
    }


def page_aware_slot_offsets(meta: dict[str, int]) -> list[int]:
    nodes_per_block = int(meta.get("layout_nodes_per_block", 0))
    max_node_len = int(meta.get("max_node_len", 0))
    block_bytes = int(meta.get("layout_block_bytes", 0))
    read_page_bytes = int(meta.get("layout_read_page_bytes", 0))
    if (
        nodes_per_block <= 0
        or max_node_len <= 0
        or read_page_bytes <= 0
        or max_node_len > read_page_bytes
        or max_node_len > block_bytes
        or nodes_per_block > block_bytes // max_node_len
    ):
        return [slot * max_node_len for slot in range(max(0, nodes_per_block))]

    padding_budget = block_bytes - nodes_per_block * max_node_len
    if padding_budget > SECTOR_LEN or nodes_per_block > 1024:
        return [slot * max_node_len for slot in range(nodes_per_block)]

    def straddles(slot: int, prefix_padding: int) -> int:
        offset = slot * max_node_len + prefix_padding
        return 1 if (offset % read_page_bytes) + max_node_len > read_page_bytes else 0

    inf = 1 << 30
    current = [inf] * (padding_budget + 1)
    current[0] = 0
    previous: list[list[int | None]] = [[None] * (padding_budget + 1) for _ in range(nodes_per_block)]

    for slot in range(nodes_per_block - 1):
        next_costs = [inf] * (padding_budget + 1)
        for pad, cost_so_far in enumerate(current):
            if cost_so_far == inf:
                continue
            cost = cost_so_far + straddles(slot, pad)
            for next_pad in range(pad, padding_budget + 1):
                if cost < next_costs[next_pad]:
                    next_costs[next_pad] = cost
                    previous[slot + 1][next_pad] = pad
        current = next_costs

    best_cost = inf
    best_pad = 0
    for pad, cost_so_far in enumerate(current):
        if cost_so_far == inf:
            continue
        cost = cost_so_far + straddles(nodes_per_block - 1, pad)
        if cost < best_cost:
            best_cost = cost
            best_pad = pad

    prefix_padding = [0] * nodes_per_block
    prefix_padding[nodes_per_block - 1] = best_pad
    for slot in range(nodes_per_block - 1, 0, -1):
        prev = previous[slot][prefix_padding[slot]]
        prefix_padding[slot - 1] = 0 if prev is None else prev
    return [slot * max_node_len + prefix_padding[slot] for slot in range(nodes_per_block)]


def enrich_layout_meta(meta: dict[str, int]) -> dict[str, Any]:
    enriched: dict[str, Any] = dict(meta)
    if enriched.get("layout_version") == 3:
        offsets = page_aware_slot_offsets(enriched)
        nodes_per_block = len(offsets)
        max_node_len = int(enriched.get("max_node_len", 0))
        read_page_bytes = int(enriched.get("layout_read_page_bytes", 0))
        if nodes_per_block > 0 and max_node_len > 0 and read_page_bytes > 0:
            straddling = sum(1 for offset in offsets if (offset % read_page_bytes) + max_node_len > read_page_bytes)
            enriched["layout_variant"] = "page_aware_slots"
            enriched["straddling_slots_per_block"] = straddling
            enriched["avg_4k_pages_per_record"] = 1.0 + straddling / nodes_per_block
    return enriched


def is_current_v3(prefix: Path) -> bool:
    meta = packed_disk_layout_meta(Path(str(prefix) + "_disk.index"))
    if not meta or meta.get("layout_version") != 3:
        return False
    max_node_len = meta.get("max_node_len", 0)
    expected_nodes = SUPERSECTOR_BYTES // max_node_len if max_node_len > 0 else 0
    return (
        meta.get("layout_block_bytes") == SUPERSECTOR_BYTES
        and meta.get("layout_nodes_per_block") == expected_nodes
        and meta.get("layout_read_page_bytes") == SECTOR_LEN
    )


def v3_prefix_for(src_prefix: Path, index_root: Path) -> Path:
    digest = hashlib.sha1(str(src_prefix).encode("utf-8")).hexdigest()[:10]
    return index_root / f"{src_prefix.name}_{digest}_super32k_v3"


def repack_prefix(src_prefix: Path, args: argparse.Namespace, out: Path) -> dict[str, Any]:
    fingerprint = source_fingerprint(src_prefix)
    dst_prefix = v3_prefix_for(src_prefix, args.index_root)
    manifest_path = Path(str(dst_prefix) + "_source_manifest.json")
    if args.force_repack:
        for path in args.index_root.glob(dst_prefix.name + "*"):
            if path.is_file():
                path.unlink()
    manifest_current = False
    if manifest_path.exists():
        try:
            manifest_current = json.loads(manifest_path.read_text()) == fingerprint
        except json.JSONDecodeError:
            manifest_current = False
    if not args.force_repack and manifest_current and is_current_v3(dst_prefix):
        meta = enrich_layout_meta(packed_disk_layout_meta(Path(str(dst_prefix) + "_disk.index")) or {})
        row = {
            "status": "exists_current",
            "src_prefix": str(src_prefix),
            "dst_prefix": str(dst_prefix),
            "source_fingerprint": fingerprint_token(fingerprint),
            **meta,
            "actual_disk_bytes": file_size(Path(str(dst_prefix) + "_disk.index")),
        }
        append_jsonl(out / "raw" / "repack_super32k.jsonl", row)
        return row
    for path in args.index_root.glob(dst_prefix.name + "*"):
        if path.is_file():
            path.unlink()

    repack = resolve_repo_path(args.repack_binary)
    cmd = [str(repack), "float", str(src_prefix), str(dst_prefix), "supersector32k"]
    elapsed, stdout = run_logged(cmd, out / "logs" / "repack_super32k.log")
    json_line = ""
    for line in stdout.splitlines()[::-1]:
        if line.startswith("{") and line.endswith("}"):
            json_line = line
            break
    if not json_line:
        raise RuntimeError(f"repack command produced no JSON row for {src_prefix}")
    row = json.loads(json_line)
    row["repack_elapsed_s"] = elapsed
    row["status"] = "ok"
    row["source_fingerprint"] = fingerprint_token(fingerprint)
    write_json(manifest_path, fingerprint)
    append_jsonl(out / "raw" / "repack_super32k.jsonl", row)
    return row


def file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def serving_components(prefix: Path) -> dict[str, int]:
    return {
        "disk_index": file_size(Path(str(prefix) + "_disk.index")),
        "pq_codes": file_size(Path(str(prefix) + "_pq_compressed.bin")),
        "pq_pivots": file_size(Path(str(prefix) + "_pq_pivots.bin")),
        "labels_sidecar": file_size(Path(str(prefix) + "_labels.densebit")),
        "hybrid_meta": file_size(Path(str(prefix) + "_hybrid.meta")),
        "disk_tags": file_size(Path(str(prefix) + "_disk.index.tags")),
        "mem_index": file_size(Path(str(prefix) + "_mem.index")),
        "mem_index_tags": file_size(Path(str(prefix) + "_mem.index.tags")),
    }


def replay_row(
    source_row: dict[str, Any],
    row_index: int,
    src_prefix: Path,
    v3_prefix: Path,
    args: argparse.Namespace,
    out: Path,
) -> dict[str, Any]:
    raw = source_row.get("raw_command")
    if not raw:
        raise RuntimeError(f"selected row {row_index} has no raw_command")
    cmd = shlex.split(raw)
    cmd[0] = str(resolve_repo_path(args.binary_root) / "dynamic_update_suite_driver")
    tmp = out / "tmp" / f"selected_{row_index:04d}.jsonl"
    if tmp.exists():
        tmp.unlink()
    replace_arg(cmd, "--source-prefix", str(v3_prefix))
    replace_arg(cmd, "--jsonl-output", str(tmp))
    replace_arg(cmd, "--cpu-cap", str(args.cpu_cap))
    if args.query_limit > 0:
        replace_arg(cmd, "--query-limit", str(args.query_limit))
    invocation = ["taskset", "-c", f"{args.cpu_start}-{args.cpu_start + args.cpu_cap - 1}", *cmd]
    elapsed, _ = run_logged(invocation, out / "logs" / "selected_super32k.log")
    result_rows = read_jsonl(tmp)
    if len(result_rows) != 1:
        raise RuntimeError(f"expected one row in {tmp}, got {len(result_rows)}")
    result = result_rows[0]
    result.update(
        {
            "layout": "supersector32k",
            "layout_version": 3,
            "layout_variant": "page_aware_slots",
            "physical_read_unit_bytes": SECTOR_LEN,
            "per_node_read_request_bytes": SECTOR_LEN,
            "v1_source_prefix": str(src_prefix),
            "v3_source_prefix": str(v3_prefix),
            "source_prefix": str(v3_prefix),
            "replay_elapsed_s": elapsed,
            "replay_invocation": cmd_text(invocation),
            "row_index": row_index,
        }
    )
    for key in ["phase", "cycle", "variant", "selector_type", "bucket", "selection_policy"]:
        if key in source_row and key not in result:
            result[key] = source_row[key]
    append_jsonl(out / "raw" / "selected_super32k.jsonl", result)
    return result


def ratio_or_none(numer: float, denom: float) -> float | None:
    return numer / denom if denom else None


def build_profiles(source_rows: list[dict[str, Any]], replayed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for src, dst in zip(source_rows, replayed):
        mean_n_4k = float(dst.get("mean_n_4k") or 0.0)
        mean_read_size = float(dst.get("mean_read_size") or 0.0)
        profiles.append(
            {
                "row_index": dst.get("row_index"),
                "variant": dst.get("variant", src.get("variant")),
                "selector_type": dst.get("selector_type", src.get("selector_type")),
                "bucket": dst.get("bucket", src.get("bucket")),
                "route": dst.get("route"),
                "actual_route": dst.get("actual_route"),
                "search_l": dst.get("search_l"),
                "v1_recall@10": src.get("recall@10"),
                "v1_avg_latency_ms": float(src.get("avg_latency_us", 0.0)) / 1000.0,
                "v1_p95_latency_ms": float(src.get("p95_latency_us", 0.0)) / 1000.0,
                "v3_recall@10": dst.get("recall@10"),
                "v3_avg_latency_ms": float(dst.get("avg_latency_us", 0.0)) / 1000.0,
                "v3_p95_latency_ms": float(dst.get("p95_latency_us", 0.0)) / 1000.0,
                "v3_mean_n_4k": mean_n_4k,
                "v3_mean_read_size": mean_read_size,
                "v3_bytes_per_4k": ratio_or_none(mean_read_size, mean_n_4k),
                "passes_primary": float(dst.get("recall@10", 0.0)) >= 98.0
                and float(dst.get("avg_latency_us", 1e18)) < 10000.0,
                "passes_p95_10ms": float(dst.get("p95_latency_us", 1e18)) < 10000.0,
            }
        )
    return profiles


def write_audits(
    out: Path,
    source_rows: list[dict[str, Any]],
    replayed: list[dict[str, Any]],
    repack_rows: list[dict[str, Any]],
) -> None:
    raw_by_prefix: dict[str, Path] = {}
    for row in source_rows:
        cmd = shlex.split(str(row["raw_command"]))
        raw_by_prefix[str(resolve_repo_path(row["source_prefix"]))] = resolve_repo_path(get_arg(cmd, "--data-bin"))

    space_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in replayed:
        v1_prefix = str(row["v1_source_prefix"])
        v3_prefix = Path(str(row["v3_source_prefix"]))
        if str(v3_prefix) in seen:
            continue
        seen.add(str(v3_prefix))
        raw_file = raw_by_prefix.get(v1_prefix)
        raw_bytes = file_size(raw_file) if raw_file else 0
        components = serving_components(v3_prefix)
        strict_serving_bytes = sum(components.values())
        space_row = {
            "v1_source_prefix": v1_prefix,
            "v3_source_prefix": str(v3_prefix),
            "raw_vector_file": str(raw_file) if raw_file else "",
            "raw_vector_file_bytes": raw_bytes,
            "strict_serving_bytes": strict_serving_bytes,
            "strict_total_over_raw_x": ratio_or_none(strict_serving_bytes, raw_bytes),
            "strict_excess_over_raw_x": ratio_or_none(strict_serving_bytes - raw_bytes, raw_bytes),
            **components,
        }
        append_jsonl(out / "index_space_audit.jsonl", space_row)
        space_rows.append(space_row)
    write_csv(out / "index_space_audit.csv", space_rows)

    profiles = build_profiles(source_rows, replayed)
    for profile in profiles:
        append_jsonl(out / "targeted_latency_profile.jsonl", profile)
    write_csv(out / "targeted_latency_profile.csv", profiles)
    write_csv(out / "optimized_dynamic_update_results.csv", replayed)
    for row in replayed:
        append_jsonl(out / "optimized_dynamic_update_results.jsonl", row)

    selected_count = len(replayed)
    avg_pass = sum(1 for r in replayed if float(r.get("avg_latency_us", 1e18)) < 10000.0)
    p95_pass = sum(1 for r in replayed if float(r.get("p95_latency_us", 1e18)) < 10000.0)
    recall_pass = sum(1 for r in replayed if float(r.get("recall@10", 0.0)) >= 98.0)
    bytes_per_4k_values = [
        p["v3_bytes_per_4k"] for p in profiles if p["v3_bytes_per_4k"] is not None and p["v3_mean_n_4k"] > 0
    ]
    claim_registry = {
        "format": "pipeann.aris.v100_supersector32k_replay_claim_registry.v1",
        "selected_count": selected_count,
        "repack_rows": len(repack_rows),
        "claims": [
            {
                "id": "C_V3_READ_GRANULARITY_4KB",
                "status": "PASS" if bytes_per_4k_values and all(abs(v - SECTOR_LEN) < 1e-6 for v in bytes_per_4k_values) else "FAIL",
                "metrics": {"bytes_per_4k_values": bytes_per_4k_values},
            },
            {
                "id": "C_V3_SELECTED_RECALL_LATENCY",
                "status": "PASS" if selected_count > 0 and avg_pass == selected_count and recall_pass == selected_count else "FAIL",
                "metrics": {
                    "selected_count": selected_count,
                    "recall_pass": recall_pass,
                    "avg_lt_10ms_pass": avg_pass,
                    "p95_lt_10ms_pass": p95_pass,
                    "min_recall": min((float(r.get("recall@10", 0.0)) for r in replayed), default=0.0),
                    "max_avg_latency_ms": max((float(r.get("avg_latency_us", 0.0)) / 1000.0 for r in replayed), default=0.0),
                    "max_p95_latency_ms": max((float(r.get("p95_latency_us", 0.0)) / 1000.0 for r in replayed), default=0.0),
                },
            },
            {
                "id": "C_V3_STRICT_TOTAL_LT_2X_RAW",
                "status": "PASS"
                if space_rows
                and all(float(r.get("strict_total_over_raw_x") or 1e18) < 2.0 for r in space_rows)
                else "FAIL",
                "metrics": {
                    "max_strict_total_over_raw_x": max(
                        (float(r.get("strict_total_over_raw_x") or 0.0) for r in space_rows), default=0.0
                    ),
                    "max_strict_excess_over_raw_x": max(
                        (float(r.get("strict_excess_over_raw_x") or 0.0) for r in space_rows), default=0.0
                    ),
                },
            },
        ],
    }
    write_json(out / "optimized_claim_registry.json", claim_registry)
    write_json(
        out / "summary.json",
        {
            "created_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
            "selected_count": selected_count,
            "repack_rows": len(repack_rows),
            "recall_pass": recall_pass,
            "avg_lt_10ms_pass": avg_pass,
            "p95_lt_10ms_pass": p95_pass,
            "max_avg_latency_ms": claim_registry["claims"][1]["metrics"]["max_avg_latency_ms"],
            "max_p95_latency_ms": claim_registry["claims"][1]["metrics"]["max_p95_latency_ms"],
            "space_rows": len(space_rows),
        },
    )

    best_space = min(space_rows, key=lambda r: float(r.get("strict_total_over_raw_x") or 1e18), default={})
    (out / "index_space_audit.md").write_text(
        "# V100 Supersector32K Index Space Audit\n\n"
        f"- Replayed selected rows: `{selected_count}`\n"
        f"- V3 repack rows: `{len(repack_rows)}`\n"
        f"- Best strict total/raw: `{float(best_space.get('strict_total_over_raw_x') or 0.0):.6f}x`\n"
        f"- Best strict excess/raw: `{float(best_space.get('strict_excess_over_raw_x') or 0.0):.6f}x`\n"
        "- Strict serving bytes count the active v3 prefix sidecars and exclude transient v1 source copies.\n"
        "- Read granularity remains separate 4KB requests; straddling records use two 4KB reads.\n"
    )
    (out / "label_sidecar_layout_audit.md").write_text(
        "# V100 Label Sidecar Layout Audit\n\n"
        "Supersector32K repacking copies `_labels.densebit` and preserves `label_size=0` in the main disk-index "
        "metadata. Label payload is therefore outside the node record; tag maps and densebit sidecars are still "
        "counted in strict serving footprint.\n"
    )


def clean_outputs(out: Path) -> None:
    for rel in [
        "raw/repack_super32k.jsonl",
        "raw/selected_super32k.jsonl",
        "targeted_latency_profile.jsonl",
        "targeted_latency_profile.csv",
        "optimized_dynamic_update_results.jsonl",
        "optimized_dynamic_update_results.csv",
        "index_space_audit.jsonl",
        "index_space_audit.csv",
        "optimized_claim_registry.json",
        "summary.json",
    ]:
        path = out / rel
        if path.exists():
            path.unlink()
    tmp = out / "tmp"
    if tmp.exists():
        for path in tmp.glob("selected_*.jsonl"):
            path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-experiment", type=Path, required=True)
    parser.add_argument("--selected-jsonl", default="raw/phaseB_selected_route_l.jsonl")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--binary-root", default="build_reviewed_20260601_explicitmat/tests")
    parser.add_argument("--repack-binary", default="build/tests/repack_disk_index_layout")
    parser.add_argument("--index-root", type=Path, default=Path("/mnt/nvme1n1/PipeANN-supersector32k-work/indexes"))
    parser.add_argument("--cpu-start", type=int, default=20)
    parser.add_argument("--cpu-cap", type=int, default=8)
    parser.add_argument("--query-limit", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force-repack", action="store_true")
    args = parser.parse_args()

    if args.cpu_cap <= 0:
        raise RuntimeError("--cpu-cap must be positive")
    source_experiment = normalize_path(resolve_repo_path(args.source_experiment))
    args.out_dir = normalize_path(resolve_repo_path(args.out_dir))
    args.index_root = normalize_path(args.index_root)
    args.binary_root = normalize_path(resolve_repo_path(args.binary_root))
    args.repack_binary = normalize_path(resolve_repo_path(args.repack_binary))
    if is_relative_to(args.index_root, ROOT):
        raise RuntimeError(f"--index-root must be outside the repository to avoid committing large indexes: {args.index_root}")
    if args.out_dir == source_experiment or is_relative_to(args.out_dir, source_experiment):
        raise RuntimeError(f"--out-dir must not be the source experiment or one of its children: {args.out_dir}")
    selected_path = source_experiment / args.selected_jsonl
    rows = [selected_payload(row) for row in read_jsonl(selected_path)]
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError(f"no selected rows found in {selected_path}")
    for idx, row in enumerate(rows):
        if not row.get("raw_command"):
            raise RuntimeError(f"selected row {idx} missing raw_command")
        if not row.get("source_prefix"):
            raise RuntimeError(f"selected row {idx} missing source_prefix")

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "tmp").mkdir(parents=True, exist_ok=True)
    args.index_root.mkdir(parents=True, exist_ok=True)
    clean_outputs(out)
    write_phase0(out, args, selected_path, rows)

    unique_prefixes = sorted({str(resolve_repo_path(row["source_prefix"])) for row in rows})
    repacked: dict[str, dict[str, Any]] = {}
    for prefix_text in unique_prefixes:
        src_prefix = Path(prefix_text)
        repacked[prefix_text] = repack_prefix(src_prefix, args, out)

    replayed: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        src_prefix = resolve_repo_path(row["source_prefix"])
        v3_prefix = Path(str(repacked[str(src_prefix)]["dst_prefix"]))
        replayed.append(replay_row(row, idx, src_prefix, v3_prefix, args, out))

    write_audits(out, rows, replayed, list(repacked.values()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
