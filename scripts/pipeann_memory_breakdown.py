#!/usr/bin/env python3

from __future__ import annotations

import argparse
import errno
import json
import os
import queue
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BUILD_DIR = REPO_ROOT / "build"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "experiments" / "memory_breakdown"
HEADER_RE = re.compile(r"^[0-9a-f]+-[0-9a-f]+\s")
PQ_MODE_RE = re.compile(r"mode:\s+(mmap|heap)\b", re.IGNORECASE)
PHASE_PATTERNS = (
    (re.compile(r"Hybrid densebit runtime loaded from"), "after_densebit_load"),
    (re.compile(r"Hybrid metadata loaded from"), "after_hybrid_metadata_load"),
    (re.compile(r"Load compressed vectors from file:"), "after_pq_load"),
    (re.compile(r"SSDIndex loaded successfully\."), "after_index_load"),
)
DEFAULT_PERF_EVENTS = (
    "page-faults",
    "minor-faults",
    "major-faults",
)
TOOL_PACKAGES = {
    "heaptrack": ("heaptrack",),
    "heaptrack_print": ("heaptrack",),
    "valgrind": ("valgrind",),
    "ms_print": ("valgrind",),
    "perf": ("linux-tools-generic", "linux-tools-$(uname -r)"),
    "pmap": ("procps",),
}


@dataclass(frozen=True)
class PresetSpec:
    name: str
    index_type: str
    index_prefix: Path
    query_bin: Path
    query_labels: Path
    selector_type: str
    threads: int
    beamwidth: int
    k: int
    similarity: str
    nbr_type: str
    route: str
    mem_l: int
    l_values: tuple[int, ...]
    truthset: str = "null"


@dataclass
class SearchCommandSpec:
    name: str
    command: list[str]
    command_source: str
    index_prefix: Path | None
    query_bin: Path | None
    query_labels: Path | None
    selector_type: str | None
    index_type: str | None
    search_binary: Path | None
    truthset: str | None = None
    preset_name: str | None = None


@dataclass(frozen=True)
class ToolStatus:
    name: str
    path: str | None

    @property
    def installed(self) -> bool:
        return self.path is not None


@dataclass(frozen=True)
class PhaseEvent:
    label: str
    line: str
    elapsed_seconds: float


def build_presets() -> dict[str, PresetSpec]:
    return {
        "yfcc10m": PresetSpec(
            name="yfcc10m",
            index_type="uint8",
            index_prefix=REPO_ROOT / "data" / "yfcc100M" / "yfcc10m_pipeann",
            query_bin=(
                REPO_ROOT
                / "data"
                / "yfcc100M"
                / "random_single_label_workloads"
                / "real_selected_labels"
                / "real_t1e-03_l8636"
                / "probe_query.bin"
            ),
            query_labels=(
                REPO_ROOT
                / "data"
                / "yfcc100M"
                / "random_single_label_workloads"
                / "real_selected_labels"
                / "real_t1e-03_l8636"
                / "probe_query.spmat"
            ),
            selector_type="intersect",
            threads=52,
            beamwidth=4,
            k=10,
            similarity="l2",
            nbr_type="pq",
            route="auto",
            mem_l=0,
            l_values=(100,),
        ),
        "sift1m": PresetSpec(
            name="sift1m",
            index_type="float",
            index_prefix=REPO_ROOT / "data" / "sift1m" / "sift1m_pipeann_uniform",
            query_bin=(
                REPO_ROOT
                / "data"
                / "sift1m"
                / "uniform_exact_selectivity"
                / "u1e-03"
                / "probe_query.bin"
            ),
            query_labels=(
                REPO_ROOT
                / "data"
                / "sift1m"
                / "uniform_exact_selectivity"
                / "u1e-03"
                / "probe_query.spmat"
            ),
            selector_type="intersect",
            threads=52,
            beamwidth=4,
            k=10,
            similarity="l2",
            nbr_type="pq",
            route="auto",
            mem_l=0,
            l_values=(100,),
        ),
    }


PRESETS = build_presets()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def validate_output_path(path: Path, description: str) -> Path:
    normalized = path.resolve(strict=False)
    build_root = DEFAULT_BUILD_DIR.resolve(strict=False)
    if normalized == build_root or build_root in normalized.parents:
        raise ValueError(f"{description} must not be written under {build_root}: {normalized}")
    return path


def require_file(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"missing {description}: {path}")
    return path


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as writer:
        json.dump(payload, writer, indent=2, sort_keys=True)
        writer.write("\n")


def pretty_command(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def require_tool(name: str) -> str:
    path = shutil.which(name)
    if path is not None:
        return path
    packages = TOOL_PACKAGES.get(name, (name,))
    raise FileNotFoundError(
        f"required tool '{name}' is not installed; try: sudo apt-get install {' '.join(packages)}"
    )


def sudo_non_interactive_available() -> bool:
    result = subprocess.run(
        ["sudo", "-n", "true"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def detect_tools() -> list[ToolStatus]:
    return [ToolStatus(name=name, path=shutil.which(name)) for name in sorted(TOOL_PACKAGES)]


def find_binary(build_dir: Path, name: str) -> Path:
    candidates = sorted(
        path
        for path in build_dir.rglob(name)
        if path.is_file() and os.access(path, os.X_OK)
    )
    if not candidates:
        raise FileNotFoundError(
            f"missing built binary '{name}' under {build_dir}. "
            f"Run cmake --build {build_dir} --target {name}."
        )
    return candidates[0]


def resolve_search_binary(command: Sequence[str]) -> Path | None:
    if not command:
        return None
    candidate = Path(command[0])
    if candidate.is_absolute() or "/" in command[0]:
        return candidate.resolve(strict=False)
    resolved = shutil.which(command[0])
    return None if resolved is None else Path(resolved)


def parse_l_values(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("provide at least one L value")
    return values


def build_search_command(spec: PresetSpec, build_dir: Path, l_values: Sequence[int], route: str) -> SearchCommandSpec:
    search_binary = find_binary(build_dir, "search_disk_index_hybrid")
    require_file(Path(f"{spec.index_prefix}_disk.index"), "disk index")
    require_file(spec.query_bin, "query bin")
    require_file(spec.query_labels, "query labels")

    command = [
        str(search_binary),
        spec.index_type,
        str(spec.index_prefix),
        str(spec.threads),
        str(spec.beamwidth),
        str(spec.query_bin),
        spec.truthset,
        str(spec.k),
        spec.similarity,
        spec.nbr_type,
        spec.selector_type,
        str(spec.query_labels),
        route,
        "0",
        str(spec.mem_l),
        *[str(value) for value in l_values],
    ]
    return SearchCommandSpec(
        name=spec.name,
        command=command,
        command_source="preset",
        index_prefix=spec.index_prefix,
        query_bin=spec.query_bin,
        query_labels=spec.query_labels,
        selector_type=spec.selector_type,
        index_type=spec.index_type,
        search_binary=search_binary,
        truthset=spec.truthset,
        preset_name=spec.name,
    )


def build_command_spec(args: argparse.Namespace) -> SearchCommandSpec:
    if args.command and args.preset:
        raise ValueError("use either --preset or --command, not both")
    if not args.command and not args.preset:
        raise ValueError("one of --preset or --command is required")

    if args.command:
        if not args.name:
            raise ValueError("--name is required with --command")
        command = shlex.split(args.command)
        if not command:
            raise ValueError("--command must not be empty")
        search_binary = resolve_search_binary(command)
        return SearchCommandSpec(
            name=args.name,
            command=command,
            command_source="explicit",
            index_prefix=resolve_path(args.index_prefix) if args.index_prefix else None,
            query_bin=resolve_path(args.query_bin) if args.query_bin else None,
            query_labels=resolve_path(args.query_labels) if args.query_labels else None,
            selector_type=args.selector_type,
            index_type=args.index_type,
            search_binary=search_binary,
            truthset=args.truthset,
        )

    preset = PRESETS[args.preset]
    build_dir = resolve_path(args.build_dir)
    l_values = parse_l_values(args.l_values) if args.l_values else preset.l_values
    route = args.route or preset.route
    spec = build_search_command(preset, build_dir, l_values, route)
    if args.name:
        spec.name = args.name
    return spec


def parse_rollup(pid: int) -> dict[str, int]:
    stats: dict[str, int] = {}
    with open(f"/proc/{pid}/smaps_rollup", "r", encoding="utf-8") as handle:
        for line in handle:
            if ":" not in line:
                continue
            key, rest = line.split(":", 1)
            parts = rest.strip().split()
            if parts and parts[0].isdigit():
                stats[key.strip()] = int(parts[0])
    return stats


def parse_smaps(pid: int) -> list[dict[str, int | str]]:
    entries: list[dict[str, int | str]] = []
    current: dict[str, int | str] | None = None
    with open(f"/proc/{pid}/smaps", "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if HEADER_RE.match(line):
                if current is not None:
                    entries.append(current)
                parts = line.split(None, 5)
                pathname = parts[5] if len(parts) >= 6 else ""
                current = {
                    "path": pathname,
                    "rss_kb": 0,
                    "pss_kb": 0,
                    "private_kb": 0,
                    "shared_kb": 0,
                    "size_kb": 0,
                }
                continue
            if current is None or ":" not in line:
                continue
            key, rest = line.split(":", 1)
            parts = rest.strip().split()
            value = int(parts[0]) if parts and parts[0].isdigit() else 0
            if key == "Size":
                current["size_kb"] = value
            elif key == "Rss":
                current["rss_kb"] = value
            elif key == "Pss":
                current["pss_kb"] = value
            elif key in {"Private_Clean", "Private_Dirty"}:
                current["private_kb"] = int(current["private_kb"]) + value
            elif key in {"Shared_Clean", "Shared_Dirty"}:
                current["shared_kb"] = int(current["shared_kb"]) + value
    if current is not None:
        entries.append(current)
    return entries


def classify_mapping(pathname: str, spec: SearchCommandSpec) -> str:
    index_prefix = None if spec.index_prefix is None else str(spec.index_prefix)
    search_binary = None if spec.search_binary is None else str(spec.search_binary)
    if index_prefix is not None:
        known_files = {
            f"{index_prefix}_labels.densebit": "densebit_sidecar",
            f"{index_prefix}_hybrid.meta": "hybrid_metadata",
            f"{index_prefix}_pq_compressed.bin": "pq_compressed",
            f"{index_prefix}_pq_pivots.bin": "pq_pivots",
            f"{index_prefix}_disk.index.tags": "disk_tags",
            f"{index_prefix}_mem.index.tags": "mem_tags",
            f"{index_prefix}_mem.index": "mem_index",
            f"{index_prefix}_disk.index": "disk_index_mapping",
        }
        if pathname in known_files:
            return known_files[pathname]
    if search_binary is not None and pathname == search_binary:
        return "search_binary"
    if pathname == "[heap]":
        return "heap_mapping"
    if pathname.startswith("[stack"):
        return "stack"
    if pathname in {"", "[anon]"}:
        return "anonymous"
    if pathname.startswith("["):
        return pathname.strip("[]")
    if pathname.startswith("/usr/lib") or pathname.startswith("/lib"):
        return "shared_libs"
    return "other_file_backed"


def aggregate_categories(entries: list[dict[str, int | str]], spec: SearchCommandSpec) -> list[dict[str, object]]:
    totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {"rss_kb": 0, "pss_kb": 0, "private_kb": 0, "shared_kb": 0, "size_kb": 0}
    )
    for entry in entries:
        category = classify_mapping(str(entry["path"]), spec)
        bucket = totals[category]
        bucket["rss_kb"] += int(entry["rss_kb"])
        bucket["pss_kb"] += int(entry["pss_kb"])
        bucket["private_kb"] += int(entry["private_kb"])
        bucket["shared_kb"] += int(entry["shared_kb"])
        bucket["size_kb"] += int(entry["size_kb"])
    ordered = sorted(totals.items(), key=lambda item: item[1]["rss_kb"], reverse=True)
    return [{"category": name, **values} for name, values in ordered]


def top_mappings(entries: list[dict[str, int | str]], limit: int = 20) -> list[dict[str, object]]:
    ordered = sorted(entries, key=lambda entry: int(entry["rss_kb"]), reverse=True)
    return [
        {
            "path": str(entry["path"]),
            "rss_kb": int(entry["rss_kb"]),
            "pss_kb": int(entry["pss_kb"]),
            "private_kb": int(entry["private_kb"]),
            "shared_kb": int(entry["shared_kb"]),
            "size_kb": int(entry["size_kb"]),
        }
        for entry in ordered[:limit]
    ]


def safe_run_capture(command: Sequence[str]) -> str | None:
    try:
        result = subprocess.run(
            list(command),
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0 and not result.stdout and result.stderr:
        return result.stderr
    return result.stdout or result.stderr


def capture_snapshot(
    pid: int,
    spec: SearchCommandSpec,
    snapshot_dir: Path,
    label: str,
    elapsed_seconds: float,
) -> dict[str, object]:
    rollup = parse_rollup(pid)
    entries = parse_smaps(pid)
    categories = aggregate_categories(entries, spec)
    mappings = top_mappings(entries)
    payload = {
        "label": label,
        "elapsed_seconds": elapsed_seconds,
        "rollup_kb": rollup,
        "categories": categories,
        "top_mappings": mappings,
    }
    json_path = snapshot_dir / f"{label}.json"
    write_json(json_path, payload)

    pmap_output = safe_run_capture([require_tool("pmap"), "-x", str(pid)]) if shutil.which("pmap") else None
    pmap_path = None
    if pmap_output is not None:
        pmap_path = snapshot_dir / f"{label}.pmap.txt"
        ensure_parent(pmap_path)
        pmap_path.write_text(pmap_output, encoding="utf-8")

    return {
        "label": label,
        "elapsed_seconds": elapsed_seconds,
        "rollup_rss_kb": rollup.get("Rss", 0),
        "rollup_pss_kb": rollup.get("Pss", 0),
        "snapshot_json": str(json_path),
        "pmap_txt": None if pmap_path is None else str(pmap_path),
    }


def file_kb(path: Path) -> int | None:
    if not path.exists():
        return None
    return int((path.stat().st_size + 1023) // 1024)


def infer_default_pq_mode() -> str:
    value = os.environ.get("PIPEANN_PQ_MMAP")
    if value is None:
        return "mmap"
    normalized = value.strip().lower()
    if normalized in {"0", "false", "off", "no"}:
        return "heap"
    return "mmap"


def known_heap_inputs(spec: SearchCommandSpec, pq_mode: str | None) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []

    def add_item(name: str, path: Path | None, note: str, estimated: bool = True) -> None:
        if path is None:
            return
        size_kb = file_kb(path)
        if size_kb is None:
            return
        items.append(
            {
                "name": name,
                "path": str(path),
                "estimated_kb": size_kb,
                "estimated": estimated,
                "note": note,
            }
        )

    index_prefix = spec.index_prefix
    if index_prefix is not None:
        pq_mode_value = pq_mode or infer_default_pq_mode()
        if pq_mode_value == "heap":
            add_item(
                "pq_compressed_heap_load",
                Path(f"{index_prefix}_pq_compressed.bin"),
                "PQ compressed vectors were heap-loaded instead of mmap-backed.",
            )
        add_item(
            "pq_pivots_heap_load",
            Path(f"{index_prefix}_pq_pivots.bin"),
            "PQ pivots are loaded into process heap and expanded into lookup tables.",
        )
        add_item(
            "disk_tags_heap_load",
            Path(f"{index_prefix}_disk.index.tags"),
            "Disk tags are read into in-process tag tables.",
        )
        add_item(
            "hybrid_meta_heap_parse",
            Path(f"{index_prefix}_hybrid.meta"),
            "Hybrid metadata is parsed into heap structures after file read.",
        )
        add_item(
            "mem_index_heap_load",
            Path(f"{index_prefix}_mem.index"),
            "Mem index is loaded into process heap when mem_L is non-zero.",
        )

    add_item(
        "query_bin_heap_load",
        spec.query_bin,
        "Query vectors are loaded via load_bin into anonymous heap-backed memory.",
    )
    add_item(
        "query_labels_heap_load",
        spec.query_labels,
        "SPMAT query labels are expanded into heap vectors.",
    )
    return items


def valgrind_failed_due_to_unsupported_instruction(output: str) -> bool:
    lower = output.lower()
    return "unrecognised instruction" in lower or "illegal opcode" in lower or "sigill" in lower


def perf_failed_due_to_permissions(output: str) -> bool:
    lower = output.lower()
    return "access to performance monitoring" in lower or "perf_event_paranoid" in lower


class OutputReader(threading.Thread):
    def __init__(self, process: subprocess.Popen[str], log_path: Path, start_time: float):
        super().__init__(daemon=True)
        self.process = process
        self.log_path = log_path
        self.start_time = start_time
        self.events: queue.Queue[PhaseEvent] = queue.Queue()
        self.detected_pq_mode: str | None = None
        self._seen_labels: set[str] = set()

    def run(self) -> None:
        ensure_parent(self.log_path)
        with self.log_path.open("w", encoding="utf-8") as writer:
            if self.process.stdout is None:
                return
            for line in self.process.stdout:
                writer.write(line)
                writer.flush()
                pq_match = PQ_MODE_RE.search(line)
                if pq_match is not None:
                    self.detected_pq_mode = pq_match.group(1).lower()
                for pattern, label in PHASE_PATTERNS:
                    if label in self._seen_labels:
                        continue
                    if pattern.search(line):
                        self._seen_labels.add(label)
                        self.events.put(
                            PhaseEvent(
                                label=label,
                                line=line.rstrip("\n"),
                                elapsed_seconds=time.perf_counter() - self.start_time,
                            )
                        )


def drain_phase_events(reader: OutputReader) -> list[PhaseEvent]:
    events: list[PhaseEvent] = []
    while True:
        try:
            events.append(reader.events.get_nowait())
        except queue.Empty:
            return events


def run_resident(args: argparse.Namespace, spec: SearchCommandSpec) -> Path:
    run_root = validate_output_path(resolve_path(args.output_root), "output root") / spec.name
    resident_root = run_root / "resident"
    snapshots_root = resident_root / "snapshots"
    resident_root.mkdir(parents=True, exist_ok=True)
    snapshots_root.mkdir(parents=True, exist_ok=True)

    spec_path = resident_root / "resolved_command.json"
    write_json(
        spec_path,
        {
            "format": "pipeann.memory.command.v1",
            "name": spec.name,
            "preset": spec.preset_name,
            "command": spec.command,
            "command_source": spec.command_source,
            "index_prefix": None if spec.index_prefix is None else str(spec.index_prefix),
            "query_bin": None if spec.query_bin is None else str(spec.query_bin),
            "query_labels": None if spec.query_labels is None else str(spec.query_labels),
            "selector_type": spec.selector_type,
            "index_type": spec.index_type,
            "search_binary": None if spec.search_binary is None else str(spec.search_binary),
        },
    )

    start_time = time.perf_counter()
    process = subprocess.Popen(
        spec.command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        env=os.environ.copy(),
    )
    reader = OutputReader(process, resident_root / "command.log", start_time)
    reader.start()

    snapshots: list[dict[str, object]] = []
    phase_events: list[dict[str, object]] = []
    snapshot_index = 0
    next_interval = start_time

    def maybe_capture(label: str, elapsed_seconds: float) -> None:
        nonlocal snapshot_index
        try:
            snapshot = capture_snapshot(process.pid, spec, snapshots_root, f"{snapshot_index:03d}_{label}", elapsed_seconds)
        except OSError as exc:
            if exc.errno in {errno.ENOENT, errno.ESRCH}:
                return
            raise
        snapshots.append(snapshot)
        snapshot_index += 1

    maybe_capture("startup", 0.0)
    next_interval = start_time + (args.sample_interval_ms / 1000.0)

    while process.poll() is None:
        now = time.perf_counter()
        for event in drain_phase_events(reader):
            phase_events.append(asdict(event))
            maybe_capture(event.label, event.elapsed_seconds)
        if now >= next_interval:
            maybe_capture("interval", now - start_time)
            next_interval = now + (args.sample_interval_ms / 1000.0)
        time.sleep(min(args.sample_interval_ms / 4000.0, 0.05))

    exit_code = process.wait()
    reader.join(timeout=1.0)
    for event in drain_phase_events(reader):
        phase_events.append(asdict(event))

    elapsed_seconds = time.perf_counter() - start_time
    pq_mode = reader.detected_pq_mode or infer_default_pq_mode()
    peak_snapshot = max(snapshots, key=lambda item: int(item["rollup_rss_kb"]), default=None)
    peak_payload = None
    peak_path = None if peak_snapshot is None else Path(str(peak_snapshot["snapshot_json"]))
    if peak_path is not None and peak_path.exists():
        peak_payload = json.loads(peak_path.read_text(encoding="utf-8"))

    summary = {
        "format": "pipeann.memory.resident.v1",
        "name": spec.name,
        "preset": spec.preset_name,
        "command": spec.command,
        "command_source": spec.command_source,
        "elapsed_seconds": elapsed_seconds,
        "exit_code": exit_code,
        "sample_interval_ms": args.sample_interval_ms,
        "pq_mode": pq_mode,
        "phase_events": phase_events,
        "snapshots": snapshots,
        "peak_snapshot": peak_snapshot,
        "peak_rollup_kb": None if peak_payload is None else peak_payload["rollup_kb"],
        "peak_categories": [] if peak_payload is None else peak_payload["categories"][:15],
        "peak_top_mappings": [] if peak_payload is None else peak_payload["top_mappings"][:15],
        "known_heap_inputs": known_heap_inputs(spec, pq_mode),
        "notes": [
            "Resident snapshots come from /proc/<pid>/smaps and /proc/<pid>/smaps_rollup, not from code instrumentation.",
            "Use the sibling massif artifact for heap callsite attribution; use the sibling perf artifact for fault/callgraph evidence.",
        ],
    }
    summary_path = resident_root / "summary.json"
    write_json(summary_path, summary)

    if exit_code != 0:
        raise RuntimeError(
            f"resident profiling command failed with exit code {exit_code}; see {resident_root / 'command.log'}"
        )
    print(f"[ok] wrote resident summary to {summary_path}")
    return summary_path


def run_massif(args: argparse.Namespace, spec: SearchCommandSpec) -> Path:
    valgrind = require_tool("valgrind")
    run_root = validate_output_path(resolve_path(args.output_root), "output root") / spec.name
    massif_root = run_root / "massif"
    massif_root.mkdir(parents=True, exist_ok=True)

    massif_out = massif_root / "massif.out"
    log_path = massif_root / "valgrind.log"
    command = [
        valgrind,
        "--tool=massif",
        "--pages-as-heap=yes",
        f"--massif-out-file={massif_out}",
        "--time-unit=ms",
        *spec.command,
    ]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=os.environ.copy(),
    )
    ensure_parent(log_path)
    valgrind_output = result.stdout + result.stderr
    log_path.write_text(valgrind_output, encoding="utf-8")

    backend = "massif"
    heaptrack_data = None
    heaptrack_log_path = None
    heaptrack_txt_path = None
    fallback_reason = None

    if result.returncode != 0 and valgrind_failed_due_to_unsupported_instruction(valgrind_output):
        fallback_reason = "valgrind-unsupported-instruction"
        heaptrack = require_tool("heaptrack")
        heaptrack_data = massif_root / "heaptrack.raw.gz"
        heaptrack_log_path = massif_root / "heaptrack.log"
        heaptrack_command = [heaptrack, "-o", str(heaptrack_data), *spec.command]
        heaptrack_result = subprocess.run(
            heaptrack_command,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
            env=os.environ.copy(),
        )
        heaptrack_log_path.write_text(heaptrack_result.stdout + heaptrack_result.stderr, encoding="utf-8")
        if heaptrack_result.returncode != 0:
            raise RuntimeError(
                "massif failed because Valgrind cannot execute this binary, and heaptrack fallback also failed; "
                f"see {log_path} and {heaptrack_log_path}"
            )
        backend = "heaptrack"
        result = heaptrack_result

        if shutil.which("heaptrack_print") and heaptrack_data.exists():
            heaptrack_txt_path = massif_root / "heaptrack.txt"
            print_result = subprocess.run(
                [require_tool("heaptrack_print"), str(heaptrack_data)],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            heaptrack_txt_path.write_text(print_result.stdout + print_result.stderr, encoding="utf-8")

    ms_print_output = None
    if backend == "massif" and shutil.which("ms_print") and massif_out.exists():
        ms_print_command = [require_tool("ms_print"), str(massif_out)]
        ms_result = subprocess.run(
            ms_print_command,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        ms_print_output = massif_root / "massif.txt"
        ms_print_output.write_text(ms_result.stdout + ms_result.stderr, encoding="utf-8")

    summary_path = massif_root / "summary.json"
    write_json(
        summary_path,
        {
            "format": "pipeann.memory.massif.v1",
            "name": spec.name,
            "command": command,
            "command_source": spec.command_source,
            "exit_code": result.returncode,
            "backend": backend,
            "fallback_reason": fallback_reason,
            "massif_out": str(massif_out) if massif_out.exists() else None,
            "massif_txt": None if ms_print_output is None else str(ms_print_output),
            "heaptrack_data": None if heaptrack_data is None else str(heaptrack_data),
            "heaptrack_txt": None if heaptrack_txt_path is None else str(heaptrack_txt_path),
            "log_path": str(log_path),
            "heaptrack_log_path": None if heaptrack_log_path is None else str(heaptrack_log_path),
            "pages_as_heap": True,
        },
    )
    if result.returncode != 0:
        raise RuntimeError(f"massif command failed with exit code {result.returncode}; see {log_path}")
    print(f"[ok] wrote massif summary to {summary_path}")
    return summary_path


def run_perf(args: argparse.Namespace, spec: SearchCommandSpec) -> Path:
    perf = require_tool("perf")
    run_root = validate_output_path(resolve_path(args.output_root), "output root") / spec.name
    perf_root = run_root / "perf"
    perf_root.mkdir(parents=True, exist_ok=True)

    perf_data = perf_root / "perf.data"
    record_log = perf_root / "record.log"
    events = args.perf_events or ",".join(DEFAULT_PERF_EVENTS)
    record_command = [
        perf,
        "record",
        "-o",
        str(perf_data),
        "-g",
        "-e",
        events,
        "--call-graph",
        "dwarf",
        "--",
        *spec.command,
    ]
    record_result = subprocess.run(
        record_command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=os.environ.copy(),
    )
    used_sudo = False
    record_output = record_result.stdout + record_result.stderr
    if record_result.returncode != 0 and perf_failed_due_to_permissions(record_output) and sudo_non_interactive_available():
        sudo_record_command = ["sudo", "-n", *record_command]
        sudo_result = subprocess.run(
            sudo_record_command,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
            env=os.environ.copy(),
        )
        record_output = record_output + "\n\n[retry-with-sudo]\n" + sudo_result.stdout + sudo_result.stderr
        record_result = sudo_result
        used_sudo = record_result.returncode == 0
        if used_sudo and perf_data.exists():
            subprocess.run(
                ["sudo", "-n", "chown", f"{os.getuid()}:{os.getgid()}", str(perf_data)],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
    record_log.write_text(record_output, encoding="utf-8")

    report_path = None
    report_timed_out = False
    report_exit_code = None
    if perf_data.exists():
        report_path = perf_root / "report.txt"
        report_command = [
            perf,
            "report",
            "--stdio",
            "--no-children",
            "--call-graph",
            "none",
            "-i",
            str(perf_data),
            "--sort",
            "dso,symbol",
        ]
        try:
            report_result = subprocess.run(
                report_command,
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
                timeout=args.perf_report_timeout_sec,
            )
            report_exit_code = report_result.returncode
            report_path.write_text(report_result.stdout + report_result.stderr, encoding="utf-8")
        except subprocess.TimeoutExpired:
            report_timed_out = True
            report_path.write_text(
                f"perf report timed out after {args.perf_report_timeout_sec}s; perf.data is still available for manual inspection.\n",
                encoding="utf-8",
            )

    summary_path = perf_root / "summary.json"
    write_json(
        summary_path,
        {
            "format": "pipeann.memory.perf.v1",
            "name": spec.name,
            "command": record_command,
            "command_source": spec.command_source,
            "events": events.split(","),
            "exit_code": record_result.returncode,
            "used_sudo": used_sudo,
            "perf_data": str(perf_data),
            "record_log": str(record_log),
            "report_txt": None if report_path is None else str(report_path),
            "report_timed_out": report_timed_out,
            "report_exit_code": report_exit_code,
        },
    )
    if record_result.returncode != 0:
        raise RuntimeError(f"perf record failed with exit code {record_result.returncode}; see {record_log}")
    print(f"[ok] wrote perf summary to {summary_path}")
    return summary_path


def add_common_command_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--preset", choices=sorted(PRESETS))
    parser.add_argument("--command", help="Raw command string to profile instead of a built-in preset.")
    parser.add_argument("--name", help="Run name. Required with --command; overrides preset name when provided.")
    parser.add_argument("--build-dir", default=str(DEFAULT_BUILD_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--route", help="Override preset route, for example auto or graph.")
    parser.add_argument("--l-values", help="Comma-separated L values used when building a preset command.")
    parser.add_argument("--index-prefix")
    parser.add_argument("--query-bin")
    parser.add_argument("--query-labels")
    parser.add_argument("--selector-type")
    parser.add_argument("--index-type")
    parser.add_argument("--truthset")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="External memory breakdown helpers for PipeANN hybrid search.")
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    doctor_parser = subparsers.add_parser("doctor", help="List installed and missing external profiling tools.")
    doctor_parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))

    resident_parser = subparsers.add_parser("resident", help="Sample /proc/<pid>/smaps during a live run.")
    add_common_command_args(resident_parser)
    resident_parser.add_argument("--sample-interval-ms", type=int, default=100)

    massif_parser = subparsers.add_parser("massif", help="Run valgrind massif and save raw artifacts.")
    add_common_command_args(massif_parser)

    perf_parser = subparsers.add_parser("perf", help="Run perf record and save perf.data plus text report.")
    add_common_command_args(perf_parser)
    perf_parser.add_argument("--perf-events", help="Comma-separated perf events. Defaults to page-faults,minor-faults,major-faults.")
    perf_parser.add_argument("--perf-report-timeout-sec", type=int, default=15)

    all_parser = subparsers.add_parser("all", help="Run resident, massif and perf in sequence.")
    add_common_command_args(all_parser)
    all_parser.add_argument("--sample-interval-ms", type=int, default=100)
    all_parser.add_argument("--perf-events", help="Comma-separated perf events. Defaults to page-faults,minor-faults,major-faults.")
    all_parser.add_argument("--perf-report-timeout-sec", type=int, default=15)

    return parser


def run_doctor(args: argparse.Namespace) -> int:
    output_root = validate_output_path(resolve_path(args.output_root), "output root")
    output_root.mkdir(parents=True, exist_ok=True)
    statuses = detect_tools()
    payload = {
        "format": "pipeann.memory.doctor.v1",
        "tools": [
            {
                "name": status.name,
                "installed": status.installed,
                "path": status.path,
                "install_packages": TOOL_PACKAGES[status.name],
            }
            for status in statuses
        ],
    }
    summary_path = output_root / "doctor.json"
    write_json(summary_path, payload)
    for status in statuses:
        if status.installed:
            print(f"[ok] {status.name}: {status.path}")
        else:
            print(f"[missing] {status.name}: sudo apt-get install {' '.join(TOOL_PACKAGES[status.name])}")
    print(f"[ok] wrote tool report to {summary_path}")
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if args.command_name == "doctor":
        return run_doctor(args)

    spec = build_command_spec(args)
    if args.command_name == "resident":
        run_resident(args, spec)
        return 0
    if args.command_name == "massif":
        run_massif(args, spec)
        return 0
    if args.command_name == "perf":
        run_perf(args, spec)
        return 0
    if args.command_name == "all":
        run_resident(args, spec)
        run_massif(args, spec)
        run_perf(args, spec)
        return 0
    raise ValueError(f"unsupported command: {args.command_name}")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)