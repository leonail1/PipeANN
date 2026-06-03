#!/usr/bin/env python3
"""Validate graph/prefilter threshold prediction against full calibration curves."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import threshold_prediction_common as common  # noqa: E402


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            pass
    return ImageFont.load_default()


def point_for_route(point: dict[str, Any], route: str) -> dict[str, float]:
    return {
        "avg_ms": common.fnum(point.get(f"{route}_avg_ms"), math.inf),
        "p95_ms": common.fnum(point.get(f"{route}_p95_ms"), math.inf),
        "recall": common.fnum(point.get(f"{route}_recall"), 0.0),
    }


def evaluate_curve(curve: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    threshold_status = curve.get("threshold_status")
    s_exp = curve.get("s_exp")
    s_pred = prediction.get("s_pred")
    rel = common.relative_error(common.fnum(s_pred) if s_pred not in (None, "") else None,
                                common.fnum(s_exp) if s_exp not in (None, "") else None)
    points = curve.get("curve_points", [])
    regrets: list[float] = []
    p95_regrets: list[float] = []
    recall_pass = 0
    latency_pass = 0
    p95_pass = 0
    decisions = []
    for point in points:
        selectivity = common.fnum(point.get("selectivity"))
        pred_route = common.route_from_threshold(selectivity, prediction)
        graph = point_for_route(point, "graph")
        prefilter = point_for_route(point, "prefilter")
        oracle_route = "graph" if graph["avg_ms"] <= prefilter["avg_ms"] else "prefilter"
        oracle = graph if oracle_route == "graph" else prefilter
        chosen = graph if pred_route == "graph" else prefilter
        if math.isfinite(chosen["avg_ms"]) and math.isfinite(oracle["avg_ms"]) and oracle["avg_ms"] > 0:
            regrets.append(max(0.0, chosen["avg_ms"] / oracle["avg_ms"] - 1.0))
        if math.isfinite(chosen["p95_ms"]) and math.isfinite(oracle["p95_ms"]) and oracle["p95_ms"] > 0:
            p95_regrets.append(max(0.0, chosen["p95_ms"] / oracle["p95_ms"] - 1.0))
        if chosen["recall"] >= 98.0:
            recall_pass += 1
        if chosen["avg_ms"] < 10.0:
            latency_pass += 1
        if chosen["p95_ms"] < 10.0:
            p95_pass += 1
        decisions.append({
            "selectivity": selectivity,
            "pred_route": pred_route,
            "oracle_route": oracle_route,
            "pred_avg_ms": chosen["avg_ms"],
            "oracle_avg_ms": oracle["avg_ms"],
            "pred_recall": chosen["recall"],
        })
    boundary_correct = None
    if threshold_status == "boundary":
        boundary_correct = prediction.get("boundary_route_pred") == curve.get("boundary_route")
    return {
        "case_id": curve.get("case_id"),
        "experiment_dir": curve.get("experiment_dir"),
        "cycle": curve.get("cycle"),
        "cycle_idx": curve.get("cycle_idx"),
        "selector_type": curve.get("selector_type"),
        "threshold_status": threshold_status,
        "s_exp": s_exp,
        "s_pred": s_pred,
        "threshold_relative_error": rel,
        "within_5pct": bool(rel is not None and rel <= 0.05),
        "boundary_route": curve.get("boundary_route"),
        "boundary_route_pred": prediction.get("boundary_route_pred"),
        "boundary_correct": boundary_correct,
        "prediction_method": prediction.get("prediction_method"),
        "sparse_points_used": prediction.get("sparse_points_used"),
        "full_points": len(points),
        "calibration_fraction": (common.fnum(prediction.get("sparse_points_used")) / len(points)) if points else 0.0,
        "max_latency_regret": max(regrets, default=0.0),
        "mean_latency_regret": sum(regrets) / len(regrets) if regrets else 0.0,
        "max_p95_regret": max(p95_regrets, default=0.0),
        "recall_pass_points": recall_pass,
        "avg_lt_10ms_points": latency_pass,
        "p95_lt_10ms_points": p95_pass,
        "decision_points": len(points),
        "decisions": decisions,
    }


def validation_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    crossing = [row for row in rows if row.get("threshold_status") == "crossing"]
    boundary = [row for row in rows if row.get("threshold_status") == "boundary"]
    single_threshold = crossing + boundary
    return {
        "evaluated_cases": len(rows),
        "crossing_cases": len(crossing),
        "boundary_cases": len(boundary),
        "single_threshold_cases": len(single_threshold),
        "multi_or_non_single_threshold_cases": len(rows) - len(single_threshold),
        "within_5pct_count": sum(1 for row in crossing if row.get("within_5pct")),
        "within_5pct_rate": (sum(1 for row in crossing if row.get("within_5pct")) / len(crossing)) if crossing else None,
        "boundary_correct_count": sum(1 for row in boundary if row.get("boundary_correct")),
        "boundary_accuracy": (sum(1 for row in boundary if row.get("boundary_correct")) / len(boundary)) if boundary else None,
        "max_latency_regret": max((common.fnum(row.get("max_latency_regret")) for row in rows), default=None),
        "crossing_max_latency_regret": max((common.fnum(row.get("max_latency_regret")) for row in crossing), default=None),
        "single_threshold_max_latency_regret": max((common.fnum(row.get("max_latency_regret")) for row in single_threshold), default=None),
        "mean_case_latency_regret": (sum(common.fnum(row.get("mean_latency_regret")) for row in rows) / len(rows)) if rows else 0.0,
        "max_p95_regret": max((common.fnum(row.get("max_p95_regret")) for row in rows), default=None),
        "single_threshold_max_p95_regret": max((common.fnum(row.get("max_p95_regret")) for row in single_threshold), default=None),
        "mean_calibration_fraction": (sum(common.fnum(row.get("calibration_fraction")) for row in rows) / len(rows)) if rows else 0.0,
        "cases_with_avg_lt_10_all_points": sum(1 for row in rows if row.get("avg_lt_10ms_points") == row.get("decision_points")),
        "cases_with_p95_lt_10_all_points": sum(1 for row in rows if row.get("p95_lt_10ms_points") == row.get("decision_points")),
        "cases_with_recall_pass_all_points": sum(1 for row in rows if row.get("recall_pass_points") == row.get("decision_points")),
    }


def leave_one_out_validate(curves: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    validation_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for i, curve in enumerate(curves):
        train_curves = [item for idx, item in enumerate(curves) if idx != i]
        model = common.train_correction_model(train_curves)
        prediction = common.predict_with_model(curve, model)
        prediction_row = {
            "case_id": curve.get("case_id"),
            "experiment_dir": curve.get("experiment_dir"),
            "cycle": curve.get("cycle"),
            "selector_type": curve.get("selector_type"),
            "truth_threshold_status": curve.get("threshold_status"),
            "truth_s_exp": curve.get("s_exp"),
            "truth_boundary_route": curve.get("boundary_route"),
            "prediction_threshold_status": prediction.get("threshold_prediction_status"),
            "s_pred": prediction.get("s_pred"),
            "boundary_route_pred": prediction.get("boundary_route_pred"),
            "fallback_route_pred": prediction.get("fallback_route_pred"),
        }
        for key, value in prediction.items():
            if key == "threshold_status":
                prediction_row["sparse_curve_threshold_status"] = value
            elif key == "threshold_prediction_status":
                continue
            elif key in {"s_exp", "boundary_route", "s_pred", "boundary_route_pred", "fallback_route_pred"}:
                prediction_row[f"sparse_curve_{key}"] = value
            else:
                prediction_row[key] = value
        prediction_rows.append(prediction_row)
        validation_rows.append(evaluate_curve(curve, prediction))
    return validation_rows, prediction_rows


def write_claim_registry(path: Path, summary: dict[str, Any]) -> None:
    within = summary.get("within_5pct_rate")
    boundary = summary.get("boundary_accuracy")
    single_regret = summary.get("single_threshold_max_latency_regret")
    all_regret = summary.get("max_latency_regret")
    crossing_cases = int(summary.get("crossing_cases") or 0)
    evaluated_cases = int(summary.get("evaluated_cases") or 0)
    boundary_cases = int(summary.get("boundary_cases") or 0)
    common.write_json(path, {
        "claims": [
            {
                "id": "T1_GROUND_TRUTH_CURVES",
                "status": "PASS" if evaluated_cases > 0 else "INSUFFICIENT",
                "claim": "Ground-truth graph/prefilter latency-vs-selectivity curves were extracted from original-query calibration artifacts.",
                "evidence": ["threshold_ground_truth_curves.jsonl", "threshold_training_dataset.jsonl"],
                "note": f"Evaluated {summary['evaluated_cases']} cases.",
            },
            {
                "id": "T2_THRESHOLD_PREDICTION_ACCURACY",
                "status": "INSUFFICIENT" if crossing_cases == 0 else ("PASS" if within is not None and within >= 0.90 else "FAIL"),
                "claim": "At least 90% of held-out crossing cases are within 5% relative threshold error.",
                "evidence": ["threshold_predictor_validation.jsonl", "threshold_predictor_results_summary.md"],
                "note": "No crossing cases." if crossing_cases == 0 else f"within_5pct_rate={within:.3f}.",
            },
            {
                "id": "T3_BOUNDARY_ACCURACY",
                "status": "INSUFFICIENT" if boundary_cases == 0 else ("PASS" if boundary is not None and boundary >= 0.90 else "FAIL"),
                "claim": "Boundary cases are classified with at least 90% accuracy.",
                "evidence": ["threshold_predictor_validation.jsonl"],
                "note": "No boundary cases." if boundary_cases == 0 else f"boundary_accuracy={boundary:.3f}.",
            },
            {
                "id": "T4_SINGLE_THRESHOLD_LATENCY_REGRET",
                "status": "INSUFFICIENT" if single_regret is None else ("PASS" if single_regret <= 0.10 else "FAIL"),
                "claim": "Route decisions from predicted thresholds have <=10% max latency regret versus oracle on single-threshold crossing/boundary cases.",
                "evidence": ["threshold_predictor_validation.jsonl"],
                "note": "No single-threshold regret rows." if single_regret is None else f"single_threshold_max_latency_regret={single_regret:.3f}.",
            },
            {
                "id": "T5_ALL_CASE_ROUTE_RISK",
                "status": "INSUFFICIENT" if all_regret is None else ("PASS" if all_regret <= 0.10 else "FAIL"),
                "claim": "All extracted cases, including multi-crossing curves, stay within <=10% max latency regret versus oracle.",
                "evidence": ["threshold_predictor_validation.jsonl"],
                "note": "No evaluated regret rows." if all_regret is None else f"all_case_max_latency_regret={all_regret:.3f}; multi_or_non_single_threshold_cases={summary['multi_or_non_single_threshold_cases']}.",
            },
        ]
    })


def write_summary_docs(out: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    worst = sorted(rows, key=lambda row: common.fnum(row.get("threshold_relative_error"), -1), reverse=True)[:5]
    within = summary.get("within_5pct_rate")
    boundary = summary.get("boundary_accuracy")
    status = "PASS" if within is not None and within >= 0.90 and (boundary is None or boundary >= 0.90) else "FAIL"
    max_regret = summary["max_latency_regret"] if summary["max_latency_regret"] is not None else float("nan")
    single_max_regret = summary["single_threshold_max_latency_regret"] if summary["single_threshold_max_latency_regret"] is not None else float("nan")
    max_p95_regret = summary["max_p95_regret"] if summary["max_p95_regret"] is not None else float("nan")
    single_max_p95_regret = summary["single_threshold_max_p95_regret"] if summary["single_threshold_max_p95_regret"] is not None else float("nan")
    (out / "threshold_predictor_results_summary.md").write_text(
        "# Threshold Predictor Results Summary\n\n"
        f"- Verdict: `{status}` for the 5% threshold-accuracy gate.\n"
        f"- Evaluated cases: `{summary['evaluated_cases']}`.\n"
        f"- Crossing cases: `{summary['crossing_cases']}`; within 5%: `{summary['within_5pct_count']}`; rate: `{within if within is not None else 'n/a'}`.\n"
        f"- Boundary cases: `{summary['boundary_cases']}`; boundary accuracy: `{boundary if boundary is not None else 'n/a'}`.\n"
        f"- Single-threshold max latency regret: `{single_max_regret:.4f}`; all-case max latency regret: `{max_regret:.4f}`; mean case latency regret: `{summary['mean_case_latency_regret']:.4f}`.\n"
        f"- Single-threshold max p95 regret: `{single_max_p95_regret:.4f}`; all-case max p95 regret: `{max_p95_regret:.4f}`.\n"
        f"- Multi/non-single-threshold cases: `{summary['multi_or_non_single_threshold_cases']}`.\n"
        f"- Mean calibration fraction: `{summary['mean_calibration_fraction']:.4f}`.\n",
        encoding="utf-8",
    )
    lines = ["# Threshold Prediction Error Analysis\n\n", "## Worst Threshold Errors\n"]
    for row in worst:
        lines.append(
            f"- `{row['case_id']}`: status={row['threshold_status']}, s_exp={row.get('s_exp')}, "
            f"s_pred={row.get('s_pred')}, rel_err={row.get('threshold_relative_error')}\n"
        )
    lines.append("\n## Notes\n- Validation is an offline replay of calibration artifacts produced with original query files. It does not start new query binaries.\n")
    (out / "threshold_prediction_error_analysis.md").write_text("".join(lines), encoding="utf-8")
    (out / "aris_threshold_prediction_final_review.md").write_text(
        "# ARIS Threshold Prediction Final Review\n\n"
        f"- Claim T2 threshold accuracy status: `{status}`.\n"
        f"- Main gate: 90% of crossing cases within 5% relative error; observed `{within if within is not None else 'n/a'}`.\n"
        f"- Boundary accuracy: `{boundary if boundary is not None else 'n/a'}`.\n"
        f"- Single-threshold max latency regret: `{single_max_regret:.4f}`; all-case max latency regret: `{max_regret:.4f}`.\n"
        "- Caveat: current model is a transparent sparse cost model, not a high-capacity learned model. It should be treated as a feasibility baseline and upgraded if the 5% gate fails.\n",
        encoding="utf-8",
    )
    (out / "ppt_ready_threshold_prediction_summary.md").write_text(
        "# PPT-ready Threshold Prediction Summary\n\n"
        f"- Task: predict graph/prefilter latency crossing selectivity `s*`.\n"
        f"- Accuracy: `{summary['within_5pct_count']}/{summary['crossing_cases']}` crossing cases within 5%; rate `{within if within is not None else 'n/a'}`.\n"
        f"- Boundary accuracy: `{boundary if boundary is not None else 'n/a'}`.\n"
        f"- Single-threshold max latency regret: `{single_max_regret:.3f}`; all-case max latency regret: `{max_regret:.3f}`; mean calibration cost `{summary['mean_calibration_fraction']:.3f}` of full sweep.\n"
        "- Use: predictor can reduce sweep cost when curves are close to linear; failed cases indicate where extra calibration points or richer features are needed.\n",
        encoding="utf-8",
    )


def write_case_study(out: Path, curves: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    matches = [row for row in rows if "cycle03_pq_only" in str(row.get("cycle")) and row.get("selector_type") == "intersect"]
    curves_by_id = {curve["case_id"]: curve for curve in curves}
    lines = ["# Case Study: PQ-only Cycle3 Failure\n\n"]
    if not matches:
        lines.append("No matching cycle3 PQ-only intersect case was found in the extracted curves.\n")
    for row in matches:
        curve = curves_by_id.get(row["case_id"], {})
        points = curve.get("curve_points", [])
        max_graph_recall = max((common.fnum(point.get("graph_recall")) for point in points), default=0.0)
        min_prefilter_pass = min(
            (common.fnum(point.get("selectivity")) for point in points if common.fnum(point.get("prefilter_recall")) >= 98.0),
            default=None,
        )
        lines.append(
            f"- `{row['case_id']}`: threshold_status={row['threshold_status']}, "
            f"s_exp={row.get('s_exp')}, s_pred={row.get('s_pred')}, rel_err={row.get('threshold_relative_error')}.\n"
        )
        lines.append(f"  Max graph recall in curve: `{max_graph_recall:.2f}`; first prefilter recall-passing selectivity: `{min_prefilter_pass}`.\n")
        lines.append("  Interpretation: this remains a graph/prefilter route-threshold and graph-quality problem, not something PQ-only sidecar rebuild can solve by itself.\n")
    (out / "threshold_predictor_case_study_pq_only_failure.md").write_text("".join(lines), encoding="utf-8")


def write_chart(out: Path, rows: list[dict[str, Any]]) -> None:
    crossing = [row for row in rows if row.get("threshold_status") == "crossing"]
    W, H = 1050, 650
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    title = load_font(27, True)
    font = load_font(17)
    small = load_font(13)
    draw.text((45, 28), "Threshold Prediction: s_pred vs s_exp", font=title, fill=(20, 20, 20))
    x0, y0, x1, y1 = 85, 100, 980, 560
    draw.rectangle((x0, y0, x1, y1), outline=(60, 60, 60), width=2)
    for frac in [0, 0.25, 0.5, 0.75, 1.0]:
        x = x0 + frac * (x1 - x0)
        y = y1 - frac * (y1 - y0)
        draw.line((x, y0, x, y1), fill=(230, 230, 230))
        draw.line((x0, y, x1, y), fill=(230, 230, 230))
        draw.text((x - 10, y1 + 8), f"{frac:.2f}", font=small, fill=(70, 70, 70))
        draw.text((35, y - 8), f"{frac:.2f}", font=small, fill=(70, 70, 70))
    draw.line((x0, y1, x1, y0), fill=(80, 80, 80), width=2)
    for row in crossing:
        truth = common.fnum(row.get("s_exp"), math.nan)
        pred = common.fnum(row.get("s_pred"), math.nan)
        if not math.isfinite(truth) or not math.isfinite(pred):
            continue
        x = x0 + truth * (x1 - x0)
        y = y1 - pred * (y1 - y0)
        color = (70, 150, 95) if row.get("within_5pct") else (205, 75, 65)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
    draw.text((445, 600), "s_exp", font=font, fill=(40, 40, 40))
    draw.text((10, 305), "s_pred", font=font, fill=(40, 40, 40))
    draw.rectangle((720, 35, 740, 55), fill=(70, 150, 95))
    draw.text((750, 32), "within 5%", font=font, fill=(30, 30, 30))
    draw.rectangle((850, 35, 870, 55), fill=(205, 75, 65))
    draw.text((880, 32), "miss", font=font, fill=(30, 30, 30))
    img.save(out / "threshold_prediction_scatter.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/mnt/nvme1n1/PipeANN-github"))
    parser.add_argument("--curves", type=Path, required=True)
    parser.add_argument("--training-dataset", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    curves_path = args.curves if args.curves.is_absolute() else repo / args.curves
    out = args.out_dir if args.out_dir.is_absolute() else repo / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    curves = common.read_jsonl(curves_path, strict=True)
    if not curves:
        raise RuntimeError(f"no curves found in required input: {curves_path}")
    validation_rows, prediction_rows = leave_one_out_validate(curves)
    summary = validation_summary(validation_rows)
    common.write_csv(out / "threshold_predictor_validation.csv", validation_rows)
    (out / "threshold_predictor_validation.jsonl").write_text(
        "".join(json.dumps(common.sanitize_json(row), sort_keys=True, allow_nan=False) + "\n" for row in validation_rows),
        encoding="utf-8",
    )
    common.write_csv(out / "threshold_predictions.csv", prediction_rows)
    (out / "threshold_predictions.jsonl").write_text(
        "".join(json.dumps(common.sanitize_json(row), sort_keys=True, allow_nan=False) + "\n" for row in prediction_rows),
        encoding="utf-8",
    )
    common.write_json(out / "threshold_predictor_validation_summary.json", summary)
    write_summary_docs(out, summary, validation_rows)
    write_case_study(out, curves, validation_rows)
    write_chart(out, validation_rows)
    write_claim_registry(out / "threshold_prediction_claim_registry.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
