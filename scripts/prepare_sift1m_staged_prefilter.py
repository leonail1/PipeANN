#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_STAGE_COUNTS = (250_000, 500_000, 750_000, 1_000_000)
DEFAULT_PQ_BITS = (8, 16, 32)


@dataclass(frozen=True)
class BuildConfig:
    r: int
    l: int
    memory_gb: int
    threads: int
    metric: str
    nbr_type: str


@dataclass(frozen=True)
class StageRecord:
    name: str
    short_name: str
    npoints: int
    base_bin: str
    workload_dir: str
    runtime_dir: str
    results_dir: str
    probe_dir: str
    source_prefixes: dict[str, str]
    compare_png: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare staged SIFT1M subset bins and experiment manifest for dynamic prefilter PQ experiments."
    )
    parser.add_argument(
        "--base-bin",
        type=Path,
        default=Path("data/sift1m/sift_base.bin"),
        help="Path to the full SIFT1M base bin.",
    )
    parser.add_argument(
        "--query-bin",
        type=Path,
        default=Path("data/sift1m/sift_query.bin"),
        help="Path to the shared query bin.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/sift1m_uniform_prefilter_pq_staged_compare"),
        help="Root output directory for staged assets and experiment outputs.",
    )
    parser.add_argument(
        "--stage-count",
        type=int,
        action="append",
        dest="stage_counts",
        help="Visible corpus sizes to materialize. Repeat to override defaults.",
    )
    parser.add_argument(
        "--pq-bits",
        type=int,
        action="append",
        dest="pq_bits",
        help="PQ variants to track. Repeat to override defaults.",
    )
    parser.add_argument("--build-r", type=int, default=64)
    parser.add_argument("--build-l", type=int, default=96)
    parser.add_argument("--build-memory-gb", type=int, default=64)
    parser.add_argument("--build-threads", type=int, default=52)
    parser.add_argument("--metric", default="l2")
    parser.add_argument("--nbr-type", default="pq")
    return parser.parse_args()


def stage_short_name(npoints: int) -> str:
    if npoints % 1_000_000 == 0:
        return f"{npoints // 1_000_000}m"
    if npoints % 1_000 == 0:
        return f"{npoints // 1_000}k"
    return str(npoints)


def read_bin_header(path: Path) -> tuple[int, int]:
    with path.open("rb") as reader:
        raw = reader.read(8)
    if len(raw) != 8:
        raise ValueError(f"failed to read bin header from {path}")
    npts, dim = struct.unpack("ii", raw)
    return npts, dim


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def copy_prefix_bin(source: Path, destination: Path, npoints: int, dim: int) -> None:
    ensure_parent(destination)
    bytes_per_vector = dim * 4
    payload_bytes = npoints * bytes_per_vector
    with source.open("rb") as src, destination.open("wb") as dst:
        src.seek(8)
        dst.write(struct.pack("ii", npoints, dim))
        remaining = payload_bytes
        chunk_size = 32 * 1024 * 1024
        while remaining > 0:
            chunk = src.read(min(chunk_size, remaining))
            if not chunk:
                raise ValueError(f"unexpected EOF while copying first {npoints} vectors from {source}")
            dst.write(chunk)
            remaining -= len(chunk)


def build_stage_records(root: Path, stage_counts: Iterable[int], pq_bits: Iterable[int]) -> list[StageRecord]:
    stage_records: list[StageRecord] = []
    for count in stage_counts:
        short_name = stage_short_name(count)
        stage_dir = root / f"stage_{short_name}"
        stage_records.append(
            StageRecord(
                name=f"stage_{short_name}",
                short_name=short_name,
                npoints=count,
                base_bin=str((root / "data" / f"sift_base_{short_name}.bin").resolve()),
                workload_dir=str((stage_dir / "workloads").resolve()),
                runtime_dir=str((stage_dir / "runtime_indexes").resolve()),
                results_dir=str((stage_dir / "results").resolve()),
                probe_dir=str((stage_dir / "probes").resolve()),
                source_prefixes={
                    str(bits): str((stage_dir / "source_indexes" / f"sift1m_stage_{short_name}_pq{bits}").resolve())
                    for bits in pq_bits
                },
                compare_png=str((stage_dir / f"sift1m_uniform_prefilter_pq_compare_{short_name}_l100.png").resolve()),
            )
        )
    return stage_records


def main() -> int:
    args = parse_args()
    base_bin = args.base_bin.resolve()
    query_bin = args.query_bin.resolve()
    out_dir = args.out_dir.resolve()

    stage_counts = tuple(args.stage_counts or DEFAULT_STAGE_COUNTS)
    pq_bits = tuple(args.pq_bits or DEFAULT_PQ_BITS)

    if sorted(stage_counts) != list(stage_counts):
        raise ValueError("stage counts must be strictly increasing")
    if len(set(stage_counts)) != len(stage_counts):
        raise ValueError("duplicate stage counts are not allowed")
    if len(set(pq_bits)) != len(pq_bits):
        raise ValueError("duplicate PQ variants are not allowed")

    total_points, dim = read_bin_header(base_bin)
    if stage_counts[-1] > total_points:
        raise ValueError(
            f"largest stage ({stage_counts[-1]}) exceeds base corpus size {total_points} from {base_bin}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    stage_records = build_stage_records(out_dir, stage_counts, pq_bits)
    for stage in stage_records:
        stage_dir = Path(stage.workload_dir).parent
        stage_dir.mkdir(parents=True, exist_ok=True)
        Path(stage.runtime_dir).mkdir(parents=True, exist_ok=True)
        Path(stage.results_dir).mkdir(parents=True, exist_ok=True)
        Path(stage.probe_dir).mkdir(parents=True, exist_ok=True)
        (stage_dir / "source_indexes").mkdir(parents=True, exist_ok=True)

        stage_base = Path(stage.base_bin)
        if stage.npoints == total_points:
            if stage_base != base_bin:
                if stage_base.exists() or stage_base.is_symlink():
                    stage_base.unlink()
                try:
                    os.symlink(base_bin, stage_base)
                except OSError:
                    copy_prefix_bin(base_bin, stage_base, stage.npoints, dim)
        else:
            copy_prefix_bin(base_bin, stage_base, stage.npoints, dim)

    manifest = {
        "format": "pipeann.dynamic_prefilter_stages.v1",
        "base_bin": str(base_bin),
        "query_bin": str(query_bin),
        "total_points": total_points,
        "dim": dim,
        "pq_bits": list(pq_bits),
        "build_config": asdict(
            BuildConfig(
                r=args.build_r,
                l=args.build_l,
                memory_gb=args.build_memory_gb,
                threads=args.build_threads,
                metric=args.metric,
                nbr_type=args.nbr_type,
            )
        ),
        "stages": [asdict(stage) for stage in stage_records],
    }

    manifest_path = out_dir / "stage_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as writer:
        json.dump(manifest, writer, indent=2)
        writer.write("\n")

    print(f"[ok] wrote stage manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())