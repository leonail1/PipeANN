#!/usr/bin/env python3
"""Build graph/prefilter selectivity-threshold ground truth from PipeANN calibration logs."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import threshold_prediction_common as common  # noqa: E402


RELATED_WORK = """# Threshold Prediction Related Work

The target here is narrower than general ANN hyperparameter search: predict the
selectivity threshold where graph-search latency and prefilter latency cross.

- Faiss ParameterSpace autotuning builds operating points and keeps Pareto
  optimal recall/time configurations. This is useful as an oracle and validation
  pattern, but it still explores parameter combinations rather than predicting a
  PipeANN graph/prefilter route threshold from features.
  Source: https://github.com/facebookresearch/faiss/wiki/Index-IO%2C-cloning-and-hyper-parameter-tuning
- Google's constrained-optimization framing for ANN configuration treats recall,
  latency and resource limits as constrained objectives. The useful transfer is
  the explicit constrained selection layer: predict performance first, then pick
  a route only if it satisfies recall/latency gates.
  Source: https://arxiv.org/abs/2301.01702
- VDTuner studies automatic tuning for vector data management systems. It
  supports the idea that vector DB tuning should use workload/index/system
  features, but PipeANN still needs disk-graph, 4KB IO and dynamic-update
  features that are not generic DB knobs.
  Source: https://arxiv.org/abs/2404.10413
- FastPGT targets proximity-graph construction parameter tuning and reduces
  repeated graph-build cost by estimating multiple candidate parameter settings
  together. It is relevant to offline graph-build tuning, while this goal is an
  online route-threshold predictor over already-built graph/prefilter curves.
  Source: https://arxiv.org/abs/2602.11573
- RP-Tuning adjusts DiskANN-style graph reachability parameters by pruning
  rather than rebuilding the whole graph. It is relevant for future graph-quality
  maintenance, but it does not predict filter selectivity thresholds.
  Source: https://arxiv.org/abs/2602.08097
- Learned adaptive early termination predicts when ANN search can stop. The
  transferable idea is a learned cost/risk model over query/index state, though
  this goal predicts route thresholds across filter selectivity rather than
  stopping depth inside one route.
  Source: https://www.pdl.cmu.edu/PDL-FTP/BigLearning/mod0246-liA.pdf
- Learning-to-rank / learned routing work for ANN suggests query- or workload-
  aware route choice. The PipeANN variant here uses route latency curves and
  4KB IO statistics as the supervised signal.
  Source: https://arxiv.org/abs/2404.11731
"""


def discover_calibration_files(repo: Path, roots: list[str]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        base = repo / root
        if base.is_file():
            files.append(base)
            continue
        if not base.exists():
            raise FileNotFoundError(base)
        files.extend(base.glob("**/raw/calibration_*.jsonl"))
        files.extend(base.glob("**/calibration_results.jsonl"))
    return sorted(set(path.resolve() for path in files))


def dataset_id(experiment_dir: str) -> str:
    text = experiment_dir.lower()
    if "bigann" in text:
        return "bigann"
    if "sift" in text or "r116" in text or "v100" in text or "pq" in text:
        return "sift"
    if "yfcc" in text:
        return "yfcc"
    return "unknown"


def cycle_index(cycle: str) -> int:
    match = re.search(r"cycle[_-]?0*([0-9]+)", cycle or "")
    if match:
        return int(match.group(1))
    return 0


def choose_best_route_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    recall_pass = [row for row in rows if common.recall(row) >= 98.0]
    candidates = recall_pass or rows
    best = min(candidates, key=lambda row: common.avg_ms(row))
    return {
        "avg_ms": common.avg_ms(best),
        "p95_ms": common.p95_ms(best),
        "p99_ms": common.p99_ms(best),
        "recall": common.recall(best),
        "recall_pass": common.recall(best) >= 98.0,
        "search_l": int(common.fnum(best.get("search_l", best.get("L", best.get("configured_L", 0))))),
        "candidate_count": common.fnum(best.get("mean_candidate_count", best.get("candidate_count")), math.nan),
        "mean_n_4k": common.fnum(best.get("mean_n_4k", best.get("mean_ios", best.get("mean_n_ios"))), math.nan),
        "mean_n_cmps": common.fnum(best.get("mean_n_cmps"), math.nan),
        "mean_route_overhead_us": common.fnum(best.get("mean_route_overhead_us"), math.nan),
        "source_status": "recall_pass_best" if recall_pass else "fastest_but_recall_below_98",
    }


def build_curves(repo: Path, files: list[Path], min_selectivity_points: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], dict[float, dict[str, list[dict[str, Any]]]]] = {}
    metadata: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path in files:
        rows = common.read_jsonl(path, strict=True)
        for row in rows:
            route = common.route_name(row)
            if route not in {"graph", "prefilter"}:
                continue
            ctx = common.infer_calibration_context(path, row)
            selector = ctx["selector_type"]
            if selector not in {"intersect", "range"}:
                continue
            selectivity = common.row_selectivity({**row, "bucket": ctx["bucket"]})
            if selectivity is None:
                continue
            key = (ctx["experiment_dir"], ctx["cycle"], selector)
            grouped.setdefault(key, {}).setdefault(selectivity, {}).setdefault(route, []).append(row)
            metadata.setdefault(key, {
                "experiment_dir": ctx["experiment_dir"],
                "cycle": ctx["cycle"],
                "selector_type": selector,
                "dataset_id": dataset_id(ctx["experiment_dir"]),
            })

    curves: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    for key, by_selectivity in sorted(grouped.items()):
        experiment_dir, cycle, selector = key
        points: list[dict[str, Any]] = []
        for selectivity, routes in sorted(by_selectivity.items()):
            if "graph" not in routes or "prefilter" not in routes:
                continue
            graph = choose_best_route_row(routes["graph"])
            prefilter = choose_best_route_row(routes["prefilter"])
            point = {
                "selectivity": selectivity,
                "graph_avg_ms": graph["avg_ms"],
                "graph_p95_ms": graph["p95_ms"],
                "graph_p99_ms": graph["p99_ms"],
                "graph_recall": graph["recall"],
                "graph_recall_pass": graph["recall_pass"],
                "graph_search_l": graph["search_l"],
                "graph_candidate_count": None if math.isnan(graph["candidate_count"]) else graph["candidate_count"],
                "graph_mean_n_4k": None if math.isnan(graph["mean_n_4k"]) else graph["mean_n_4k"],
                "graph_mean_n_cmps": None if math.isnan(graph["mean_n_cmps"]) else graph["mean_n_cmps"],
                "graph_source_status": graph["source_status"],
                "prefilter_avg_ms": prefilter["avg_ms"],
                "prefilter_p95_ms": prefilter["p95_ms"],
                "prefilter_p99_ms": prefilter["p99_ms"],
                "prefilter_recall": prefilter["recall"],
                "prefilter_recall_pass": prefilter["recall_pass"],
                "prefilter_search_l": prefilter["search_l"],
                "prefilter_candidate_count": None if math.isnan(prefilter["candidate_count"]) else prefilter["candidate_count"],
                "prefilter_mean_n_4k": None if math.isnan(prefilter["mean_n_4k"]) else prefilter["mean_n_4k"],
                "prefilter_mean_n_cmps": None if math.isnan(prefilter["mean_n_cmps"]) else prefilter["mean_n_cmps"],
                "prefilter_source_status": prefilter["source_status"],
                "latency_diff_graph_minus_prefilter_ms": graph["avg_ms"] - prefilter["avg_ms"],
                "oracle_route": "graph" if graph["avg_ms"] <= prefilter["avg_ms"] else "prefilter",
                "oracle_avg_ms": min(graph["avg_ms"], prefilter["avg_ms"]),
            }
            points.append(point)
        if len(points) < min_selectivity_points:
            continue
        case_id = common.case_id_for(experiment_dir, cycle, selector)
        threshold = common.interpolate_threshold(points)
        max_points = max((common.fnum(point.get("prefilter_candidate_count")) for point in points), default=0.0)
        live_points = max((common.fnum(point.get("prefilter_candidate_count")) / max(common.fnum(point.get("selectivity")), 1e-12) for point in points), default=0.0)
        curve = {
            **metadata[key],
            "case_id": case_id,
            "cycle_idx": cycle_index(cycle),
            "selectivity_point_count": len(points),
            "min_selectivity": min(point["selectivity"] for point in points),
            "max_selectivity": max(point["selectivity"] for point in points),
            "estimated_live_points": live_points if live_points > 0 else None,
            "max_candidate_count": max_points,
            "curve_points": points,
            **threshold,
        }
        curves.append(curve)
        for point in points:
            training_rows.append({
                **{k: v for k, v in curve.items() if k != "curve_points"},
                **point,
            })
    return curves, training_rows


def write_feature_schema(path: Path) -> None:
    path.write_text(
        "# Threshold Feature Schema\n\n"
        "Each curve is keyed by `case_id = experiment_dir x cycle x selector_type`.\n\n"
        "- `selectivity`: mean candidate_count / live_point_count, falling back to bucket name parsing.\n"
        "- `graph_*` and `prefilter_*`: fastest measured row for that route at the selectivity. If no row reaches recall>=98, the fastest row is retained and marked by `*_source_status`.\n"
        "- `latency_diff_graph_minus_prefilter_ms`: positive means prefilter is faster; negative means graph is faster.\n"
        "- `s_exp`: piecewise-linear intersection of graph and prefilter avg-latency curves.\n"
        "- `threshold_status`: `crossing`, `boundary`, `insufficient`, or `non_monotonic_no_single_crossing`.\n"
        "- `boundary_route`: route that is no slower across the observed selectivity range when no crossing exists.\n"
        "- Validation uses sparse calibration points from each curve to predict `s_pred`, then compares against `s_exp`.\n",
        encoding="utf-8",
    )


def write_claim_registry(path: Path, curves: list[dict[str, Any]]) -> None:
    crossing = sum(1 for curve in curves if curve.get("threshold_status") == "crossing")
    boundary = sum(1 for curve in curves if curve.get("threshold_status") == "boundary")
    common.write_json(path, {
        "claims": [
            {
                "id": "T1_GROUND_TRUTH_CURVES",
                "status": "READY",
                "claim": "Ground-truth graph/prefilter latency-vs-selectivity curves are extracted from original-query calibration artifacts.",
                "evidence": ["threshold_ground_truth_curves.jsonl", "threshold_training_dataset.jsonl"],
                "note": f"Extracted {len(curves)} curves: {crossing} crossing and {boundary} boundary cases.",
            },
            {
                "id": "T2_THRESHOLD_PREDICTION_ACCURACY",
                "status": "PENDING",
                "claim": "At least 90% of held-out crossing cases are within 5% relative threshold error.",
                "evidence": [],
            },
            {
                "id": "T3_BOUNDARY_ACCURACY",
                "status": "PENDING",
                "claim": "Boundary cases are classified with at least 90% accuracy.",
                "evidence": [],
            },
            {
                "id": "T4_LATENCY_REGRET",
                "status": "PENDING",
                "claim": "Route decisions from predicted thresholds have <=10% latency regret versus oracle.",
                "evidence": [],
            },
        ]
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/mnt/nvme1n1/PipeANN-github"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--roots", nargs="*", default=["experiments"], help="Experiment roots or files to scan.")
    parser.add_argument("--min-selectivity-points", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    out = args.out_dir if args.out_dir.is_absolute() else repo / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    files = discover_calibration_files(repo, args.roots)
    curves, training_rows = build_curves(repo, files, args.min_selectivity_points)
    if not curves:
        raise RuntimeError("no eligible graph/prefilter threshold curves were extracted")
    common.write_csv(out / "threshold_training_dataset.csv", training_rows)
    (out / "threshold_training_dataset.jsonl").write_text(
        "".join(json.dumps(common.sanitize_json(row), sort_keys=True, allow_nan=False) + "\n" for row in training_rows),
        encoding="utf-8",
    )
    common.write_csv(out / "threshold_ground_truth_curves.csv", [{k: v for k, v in curve.items() if k != "curve_points"} for curve in curves])
    (out / "threshold_ground_truth_curves.jsonl").write_text(
        "".join(json.dumps(common.sanitize_json(curve), sort_keys=True, allow_nan=False) + "\n" for curve in curves),
        encoding="utf-8",
    )
    write_feature_schema(out / "threshold_feature_schema.md")
    (out / "threshold_prediction_related_work.md").write_text(RELATED_WORK, encoding="utf-8")
    write_claim_registry(out / "threshold_prediction_claim_registry.json", curves)
    common.write_json(out / "threshold_dataset_summary.json", {
        "calibration_files_scanned": len(files),
        "curve_count": len(curves),
        "training_rows": len(training_rows),
        "crossing_cases": sum(1 for curve in curves if curve.get("threshold_status") == "crossing"),
        "boundary_cases": sum(1 for curve in curves if curve.get("threshold_status") == "boundary"),
        "source_roots": args.roots,
    })
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
