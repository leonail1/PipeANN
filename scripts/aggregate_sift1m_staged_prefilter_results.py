#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate staged SIFT1M dynamic prefilter PQ results across full runs, probes, and insert batches."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path, required=True)
    return parser.parse_args()


def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as reader:
        return json.load(reader)


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as reader:
        for line in reader:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as reader:
        return json.load(reader)


def load_probe_rows(probe_dir: Path) -> list[dict]:
    for candidate in (probe_dir / "during_insert_probe.jsonl", probe_dir / "mid_insert_probe.jsonl"):
        if candidate.exists():
            return load_jsonl(candidate)
    return []


def normalize_row(base: dict, pq_bits: int, stage: dict, row: dict) -> dict:
    normalized = dict(base)
    normalized.update(row)
    normalized.setdefault("pq_bits", pq_bits)
    normalized.setdefault("stage", stage["name"])
    normalized.setdefault("stage_short_name", stage["short_name"])
    normalized.setdefault("npoints", stage["npoints"])
    return normalized


def collect_rows(manifest: dict) -> list[dict]:
    rows: list[dict] = []
    for stage in manifest["stages"]:
        for pq_bits in manifest["pq_bits"]:
            base = {
                "pq_bits": pq_bits,
                "stage": stage["name"],
                "stage_short_name": stage["short_name"],
                "npoints": stage["npoints"],
            }

            full_results = Path(stage["results_dir"]) / f"pq{pq_bits}" / "run" / "results.jsonl"
            for row in load_jsonl(full_results):
                row["mode"] = "post_stage_full"
                rows.append(normalize_row(base, pq_bits, stage, row))

            probe_dir = Path(stage["probe_dir"]) / f"pq{pq_bits}"
            for row in load_probe_rows(probe_dir):
                rows.append(normalize_row(base, pq_bits, stage, row))

            insert_summary = load_json(probe_dir / "insert_batch_summary.json")
            if insert_summary is not None:
                insert_summary["mode"] = "insert_batch"
                rows.append(normalize_row(base, pq_bits, stage, insert_summary))
    return rows


def write_json(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as writer:
        json.dump(rows, writer, indent=2)
        writer.write("\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as writer:
        csv_writer = csv.DictWriter(writer, fieldnames=fieldnames)
        csv_writer.writeheader()
        for row in rows:
            csv_writer.writerow(row)


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    rows = collect_rows(manifest)
    write_json(args.json_out, rows)
    write_csv(args.csv_out, rows)
    print(f"[ok] wrote {len(rows)} aggregated rows to {args.json_out} and {args.csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())