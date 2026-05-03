#!/usr/bin/env python3
"""Prepare a 2M SIFT/BIGANN float dataset for exp3 1M->2M insertion tests."""

from __future__ import annotations

import argparse
import os
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np


BASE_URL = "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/base.1B.u8bin"
QUERY_URL = "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/query.public.10K.u8bin"


def log(message: str) -> None:
    print(message, flush=True)


def read_bin_header(path: Path) -> tuple[int, int]:
    with path.open("rb") as reader:
        raw = reader.read(8)
    if len(raw) != 8:
        raise ValueError(f"failed to read header from {path}")
    return struct.unpack("<II", raw)


def download_range(url: str, destination: Path, byte_count: int) -> None:
    if destination.exists() and destination.stat().st_size == byte_count:
        log(f"reuse {destination}")
        return

    tmp = destination.with_suffix(destination.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"Range": f"bytes=0-{byte_count - 1}"})
    log(f"download {url} -> {destination} ({byte_count} bytes)")
    with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as writer:
        status = getattr(response, "status", None)
        if status not in (200, 206, None):
            raise RuntimeError(f"unexpected HTTP status {status} for {url}")
        copied = 0
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            writer.write(chunk)
            copied += len(chunk)
            if copied % (128 * 1024 * 1024) < len(chunk):
                log(f"  {copied / (1024 * 1024):.1f} MiB")
    if tmp.stat().st_size != byte_count:
        raise RuntimeError(f"downloaded {tmp.stat().st_size} bytes, expected {byte_count}")
    os.replace(tmp, destination)


def u8bin_prefix_to_float_bin(source: Path, destination: Path, npoints: int, chunk_points: int = 100_000) -> None:
    total, dim = read_bin_header(source)
    if total < npoints:
        raise RuntimeError(f"{source} has only {total} vectors, need {npoints}")
    expected_u8_size = 8 + npoints * dim
    if source.stat().st_size < expected_u8_size:
        raise RuntimeError(f"{source} is too small for {npoints}x{dim} u8 vectors")
    if destination.exists() and read_bin_header(destination) == (npoints, dim):
        log(f"reuse {destination}")
        return

    tmp = destination.with_suffix(destination.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    log(f"convert {source} -> {destination} ({npoints}x{dim} float32)")
    with source.open("rb") as reader, tmp.open("wb") as writer:
        reader.seek(8)
        writer.write(struct.pack("<ii", npoints, dim))
        remaining = npoints
        while remaining:
            take = min(chunk_points, remaining)
            raw = reader.read(take * dim)
            if len(raw) != take * dim:
                raise EOFError(f"unexpected EOF in {source}")
            vectors = np.frombuffer(raw, dtype=np.uint8).astype("<f4", copy=False)
            vectors.tofile(writer)
            remaining -= take
            done = npoints - remaining
            if done % 500_000 == 0 or remaining == 0:
                log(f"  converted {done}/{npoints}")
    os.replace(tmp, destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/bigann"))
    parser.add_argument("--base-points", type=int, default=2_000_000)
    parser.add_argument("--query-points", type=int, default=10_000)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--query-url", default=QUERY_URL)
    args = parser.parse_args()

    dim = 128
    base_u8 = args.out_dir / f"bigann_base_{args.base_points}_u8bin_head.u8bin"
    query_u8 = args.out_dir / f"bigann_query_{args.query_points}_u8bin_head.u8bin"
    base_float = args.out_dir / f"sift_base_{args.base_points // 1_000_000}m_float.bin"
    query_float = args.out_dir / f"sift_query_{args.query_points}_float.bin"

    download_range(args.base_url, base_u8, 8 + args.base_points * dim)
    download_range(args.query_url, query_u8, 8 + args.query_points * dim)
    u8bin_prefix_to_float_bin(base_u8, base_float, args.base_points)
    u8bin_prefix_to_float_bin(query_u8, query_float, args.query_points)

    log(f"base_bin={base_float}")
    log(f"query_bin={query_float}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
