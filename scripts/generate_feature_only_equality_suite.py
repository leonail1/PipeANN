#!/usr/bin/env python3
"""Generate equality-label suites and feature-only threshold feature tables.

This script produces prediction-time observable features only.  It does not run
query search, read route latency curves, or compute oracle thresholds.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import feature_only_threshold_common as common  # noqa: E402


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    base_bin: Path
    query_bin: Path | None
    dtype: str
    index_meta: Path | None
    index_prefix: Path | None
    max_points: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/mnt/nvme1n1/PipeANN-github"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/feature_only_threshold_20260603T_suite"))
    parser.add_argument("--dataset", action="append", help="dataset_id:base_bin[:query_bin[:dtype[:index_meta[:index_prefix]]]]")
    parser.add_argument("--selectivity", type=float, action="append", dest="selectivities")
    parser.add_argument("--label-family", action="append", dest="families")
    parser.add_argument("--vector-sample-size", type=int, default=2048)
    parser.add_argument("--target-sample-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--hardware-json", type=Path)
    parser.add_argument("--write-spmat", action="store_true", help="Write generated base/query spmat files for later query-sweep validation.")
    return parser.parse_args()


def resolve(repo: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else repo / path


def default_datasets(repo: Path) -> list[DatasetSpec]:
    specs: list[DatasetSpec] = []
    for dataset_id, base, query in [
        ("fashion_mnist784", "data/fashion_mnist784/base.bin", "data/fashion_mnist784/query.bin"),
        ("glove100", "data/glove100/base.bin", "data/glove100/query.bin"),
        ("gist960", "data/gist960/base.bin", "data/gist960/query.bin"),
        ("sift1m", "data/sift1m/sift_base.bin", "data/sift1m/sift_query.bin"),
    ]:
        base_path = repo / base
        if not base_path.exists():
            continue
        meta = repo / f"data/{dataset_id}/index_build_meta.json"
        prefix = None
        if meta.exists():
            payload = common.read_json_if_exists(meta)
            prefix = resolve(repo, payload.get("index_prefix"))
        specs.append(DatasetSpec(dataset_id, base_path, repo / query, "float32", meta if meta.exists() else None, prefix))
    return specs


def parse_dataset_spec(repo: Path, spec: str) -> DatasetSpec:
    parts = spec.split(":")
    if len(parts) < 2:
        raise ValueError("--dataset must be dataset_id:base_bin[:query_bin[:dtype[:index_meta[:index_prefix]]]]")
    dataset_id = parts[0]
    base_bin = resolve(repo, parts[1])
    query_bin = resolve(repo, parts[2]) if len(parts) > 2 and parts[2] else None
    dtype = parts[3] if len(parts) > 3 and parts[3] else "float32"
    index_meta = resolve(repo, parts[4]) if len(parts) > 4 and parts[4] else None
    index_prefix = resolve(repo, parts[5]) if len(parts) > 5 and parts[5] else None
    if base_bin is None:
        raise ValueError(f"missing base path in dataset spec {spec}")
    return DatasetSpec(dataset_id, base_bin, query_bin, dtype, index_meta, index_prefix)


def selectivities_from_args(args: argparse.Namespace) -> list[float]:
    values = args.selectivities or list(common.DEFAULT_SELECTIVITIES)
    cleaned = sorted({round(float(value), 8) for value in values if 0.0 < float(value) <= 1.0})
    if not cleaned:
        raise ValueError("at least one positive selectivity is required")
    return cleaned


def label_family_description(families: list[str]) -> str:
    lines = [
        "# Equality Label Distribution Suite",
        "",
        "All generated filters are equality filters: a query asks for exactly one target label.",
        "Feature extraction is allowed to inspect vectors, target-label membership ids, label counts, index/layout metadata, and independent hardware constants.",
        "Feature extraction is not allowed to inspect graph/prefilter latency curves, sparse probes, oracle thresholds, or query-sweep results.",
        "",
        "Generated label families:",
    ]
    for family in families:
        lines.append(f"- `{family}`")
    lines.extend([
        "",
        "Families are used for benchmark generation and held-out splitting. They must not be used as a shortcut deployment feature unless a leakage/ablation audit explicitly permits it.",
    ])
    return "\n".join(lines) + "\n"


def rank_percentile(counts: list[int], target_count: int) -> float:
    if not counts:
        return 0.0
    sorted_counts = sorted(counts)
    less_equal = sum(1 for value in sorted_counts if value <= target_count)
    return less_equal / len(sorted_counts)


def query_count_for_spec(spec: DatasetSpec) -> int:
    if not spec.query_bin or not spec.query_bin.exists():
        return 0
    nqueries, _dim = common.read_bin_header(spec.query_bin)
    return nqueries


def build_dataset_rows(spec: DatasetSpec, out: Path, args: argparse.Namespace,
                       selectivities: list[float], families: list[str],
                       hardware: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    npoints, dim = common.read_bin_header(spec.base_bin)
    effective_npoints = min(npoints, spec.max_points) if spec.max_points else npoints
    vector_features = common.vector_global_features(
        spec.base_bin,
        dtype=spec.dtype,
        sample_size=args.vector_sample_size,
        seed=args.seed + len(spec.dataset_id),
    )
    vector_features["npoints"] = effective_npoints
    vector_features["dim"] = dim
    meta = common.read_json_if_exists(spec.index_meta) if spec.index_meta else {}
    meta.setdefault("dim", dim)
    index_features = common.static_index_features(meta, spec.index_prefix, effective_npoints)
    nodes_per_page = int(index_features.get("index_nodes_per_page", 1) or 1)
    sample_for_projection = common.sample_ids(effective_npoints, min(args.vector_sample_size, effective_npoints), args.seed)
    projection_order = None
    if any(family.startswith("projection_") for family in families):
        scores = common.projection_scores(spec.base_bin, dtype=spec.dtype, sample_ids_for_fit=sample_for_projection, seed=args.seed + 101)
        projection_order = np.argsort(scores[:effective_npoints], kind="mergesort")
    query_count = query_count_for_spec(spec)

    rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for family_index, family in enumerate(families):
        memberships: list[np.ndarray] = []
        label_records: list[dict[str, Any]] = []
        for selectivity_index, target_selectivity in enumerate(selectivities):
            target_count = max(1, min(effective_npoints, int(round(effective_npoints * target_selectivity))))
            label_id = selectivity_index
            ids = common.synthetic_membership_ids(
                family,
                effective_npoints,
                target_count,
                args.seed + family_index * 1009 + selectivity_index * 17,
                projection_order=projection_order,
            )
            memberships.append(ids)
            label_records.append({
                "label_id": label_id,
                "target_selectivity": target_selectivity,
                "target_count": int(ids.size),
                "membership_sha1_hint": f"{int(ids[:min(ids.size, 8)].sum())}-{int(ids[-min(ids.size, 8):].sum()) if ids.size else 0}",
            })
        label_counts = [int(ids.size) for ids in memberships]
        global_label_features = common.label_global_features(label_counts, effective_npoints)
        base_spmat = out / "labels" / spec.dataset_id / f"{family}.base.spmat"
        if args.write_spmat:
            common.write_spmat_from_memberships(base_spmat, effective_npoints, len(memberships), memberships)
        manifest_rows.append({
            "dataset_id": spec.dataset_id,
            "label_family": family,
            "base_spmat": str(base_spmat) if args.write_spmat else None,
            "label_records": label_records,
        })
        for label_id, ids in enumerate(memberships):
            target_count = int(ids.size)
            query_spmat = out / "labels" / spec.dataset_id / family / f"query_eq_label_{label_id:03d}.spmat"
            if args.write_spmat and query_count > 0:
                query_memberships = [np.empty(0, dtype=np.uint32) for _ in memberships]
                query_memberships[label_id] = np.arange(query_count, dtype=np.uint32)
                common.write_spmat_from_memberships(query_spmat, query_count, len(memberships), query_memberships)
            target_features = common.target_label_features(
                ids,
                effective_npoints,
                vector_features,
                spec.base_bin,
                dtype=spec.dtype,
                sample_size=args.target_sample_size,
                seed=args.seed + label_id,
                nodes_per_page=nodes_per_page,
            )
            row: dict[str, Any] = {
                "case_id": f"{spec.dataset_id}__{family}__eq{label_id:03d}",
                "dataset_id": spec.dataset_id,
                "base_bin": str(spec.base_bin),
                "query_bin": str(spec.query_bin) if spec.query_bin else None,
                "query_count": query_count,
                "base_label_spmat": str(base_spmat) if args.write_spmat else None,
                "query_label_spmat": str(query_spmat) if args.write_spmat and query_count > 0 else None,
                "selector_type": "equality",
                "label_family": family,
                "label_id": label_id,
                "target_label_rank_by_frequency": sorted(label_counts, reverse=True).index(target_count) + 1,
                "target_label_percentile": rank_percentile(label_counts, target_count),
                "feature_only_input": True,
                "forbidden_inputs": "no query latency, no calibration sweep, no sparse probe, no oracle threshold",
            }
            row.update(vector_features)
            row.update(global_label_features)
            row.update(index_features)
            row.update(hardware)
            row.update(target_features)
            rows.append(row)
    return rows, manifest_rows


def write_claim_registry(out: Path, rows: list[dict[str, Any]], families: list[str]) -> None:
    feature_names = sorted({key for row in rows for key in row})
    leakage_findings = common.leakage_audit_for_features(feature_names)
    common.write_json(out / "feature_only_threshold_claim_registry.json", {
        "claims": [
            {
                "id": "F0_CLEANUP_PREVIOUS_PREDICTOR",
                "status": "PASS",
                "claim": "Calibration-assisted threshold predictor artifacts were removed before starting the feature-only goal.",
                "evidence": ["../feature_only_threshold_20260603T_cleanup/cleanup_previous_threshold_predictor.md"],
            },
            {
                "id": "F1_FEATURE_ONLY_INPUTS",
                "status": "PASS" if not leakage_findings else "FAIL",
                "claim": "Generated feature table contains only prediction-time observable feature names.",
                "evidence": ["dataset_label_feature_table.jsonl", "leakage_audit.md"],
                "note": "No leakage-token feature names found." if not leakage_findings else f"{len(leakage_findings)} suspicious feature names found.",
            },
            {
                "id": "F2_EQUALITY_LABEL_SUITE",
                "status": "PASS" if rows and len(families) >= 4 else "INSUFFICIENT",
                "claim": "Equality-label benchmark suite covers multiple label distribution families.",
                "evidence": ["equality_label_distribution_suite.md", "dataset_label_feature_table.jsonl"],
                "note": f"families={families}",
            },
            {
                "id": "F3_HELD_OUT_THRESHOLD_ACCURACY",
                "status": "PENDING",
                "claim": "Held-out feature-only model reaches >=90% cases within 5% threshold error.",
                "evidence": [],
            },
        ]
    })
    lines = ["# Leakage Audit", ""]
    if leakage_findings:
        for finding in leakage_findings:
            lines.append(f"- REJECT `{finding['feature']}`: {finding['reason']}")
    else:
        lines.append("No feature names matched the blocked leakage tokens.")
    lines.append("")
    lines.append("Blocked concepts: target query latency, calibration sweeps, sparse probes, known thresholds, recall labels as model inputs.")
    (out / "leakage_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    out = args.out_dir if args.out_dir.is_absolute() else repo / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    datasets = [parse_dataset_spec(repo, spec) for spec in args.dataset] if args.dataset else default_datasets(repo)
    if not datasets:
        raise RuntimeError("no datasets available; pass --dataset")
    selectivities = selectivities_from_args(args)
    families = args.families or list(common.DEFAULT_LABEL_FAMILIES)
    hardware = common.load_hardware_constants(resolve(repo, str(args.hardware_json)) if args.hardware_json else None)
    rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "format": "pipeann.feature_only_equality_suite.v1",
        "feature_policy": "prediction-time observable only; no target query latency/calibration/sparse probe/oracle threshold as model input",
        "datasets": [],
        "selectivities": selectivities,
        "families": families,
        "write_spmat": bool(args.write_spmat),
    }
    for spec in datasets:
        dataset_rows, label_manifests = build_dataset_rows(spec, out, args, selectivities, families, hardware)
        rows.extend(dataset_rows)
        manifest["datasets"].append({
            "dataset_id": spec.dataset_id,
            "base_bin": str(spec.base_bin),
            "query_bin": str(spec.query_bin) if spec.query_bin else None,
            "dtype": spec.dtype,
            "case_count": len(dataset_rows),
            "labels": label_manifests,
        })
    common.write_json(out / "equality_label_suite_manifest.json", manifest)
    (out / "equality_label_distribution_suite.md").write_text(label_family_description(families), encoding="utf-8")
    common.write_csv(out / "dataset_label_feature_table.csv", rows)
    with (out / "dataset_label_feature_table.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(common.sanitize_json(row), sort_keys=True, allow_nan=False) + "\n")
    common.write_json(out / "feature_table_summary.json", {
        "datasets": [spec.dataset_id for spec in datasets],
        "families": families,
        "selectivities": selectivities,
        "case_count": len(rows),
        "feature_count": len({key for row in rows for key in row}),
    })
    write_claim_registry(out, rows, families)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
