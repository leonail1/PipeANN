#!/usr/bin/env python3
"""Build the PQ32 and PQ16 exact indexes for the hybrid rebuild."""

from __future__ import annotations

import argparse
import os

from exact_hybrid_common import (
    build_index_bin_path,
    data_labels_path,
    ensure_source_sift1m_assets,
    pq16_exact_prefix,
    pq32_exact_prefix,
    resolve_path,
    run_command,
    sift1m_dir,
)


def build_one(prefix, pq_bytes: int, data_file, label_file, r: int, l_build: int, m_gb: int, threads: int,
              force: bool) -> None:
    disk_file = resolve_path(f"{prefix}_disk.index")
    disk_file.parent.mkdir(parents=True, exist_ok=True)
    if disk_file.exists() and not force:
        print(f"[skip] {disk_file}")
        return

    cmd = [
        str(build_index_bin_path()),
        "float",
        str(resolve_path(data_file)),
        str(resolve_path(prefix)),
        str(r),
        str(l_build),
        str(pq_bytes),
        str(m_gb),
        str(threads),
        "l2",
        "pq",
        "spmat",
        str(resolve_path(label_file)),
    ]
    result = run_command(cmd, timeout=7200)
    if result.returncode != 0:
        raise RuntimeError(result.stdout + result.stderr)
    print(f"[ok] {prefix} (PQ{pq_bytes})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(sift1m_dir() / "sift_base.bin"))
    parser.add_argument("--labels", default=str(data_labels_path()))
    parser.add_argument("--r", type=int, default=64)
    parser.add_argument("--l-build", type=int, default=96)
    parser.add_argument("--m-gb", type=int, default=32)
    parser.add_argument("--threads", type=int, default=min(64, os.cpu_count() or 1))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ensure_source_sift1m_assets(force=False)
    build_one(pq32_exact_prefix(), 32, args.data, args.labels, args.r, args.l_build, args.m_gb, args.threads,
              args.force)
    build_one(pq16_exact_prefix(), 16, args.data, args.labels, args.r, args.l_build, args.m_gb, args.threads,
              args.force)


if __name__ == "__main__":
    main()
