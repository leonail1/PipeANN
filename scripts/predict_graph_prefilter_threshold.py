#!/usr/bin/env python3
"""Predict graph/prefilter crossing thresholds for PipeANN calibration curves."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import threshold_prediction_common as common  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/mnt/nvme1n1/PipeANN-github"))
    parser.add_argument("--curves", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    curves_path = args.curves if args.curves.is_absolute() else repo / args.curves
    model_path = args.model if args.model.is_absolute() else repo / args.model
    output_jsonl = args.output_jsonl if args.output_jsonl.is_absolute() else repo / args.output_jsonl
    output_csv = args.output_csv if not args.output_csv or args.output_csv.is_absolute() else repo / args.output_csv
    model = json.loads(model_path.read_text(encoding="utf-8"))
    curves = common.read_jsonl(curves_path, strict=True)
    if not curves:
        raise RuntimeError(f"no curves found in required input: {curves_path}")
    predictions = []
    for curve in curves:
        pred = common.predict_with_model(curve, model)
        row = {
            "case_id": curve.get("case_id"),
            "experiment_dir": curve.get("experiment_dir"),
            "cycle": curve.get("cycle"),
            "selector_type": curve.get("selector_type"),
            "truth_threshold_status": curve.get("threshold_status"),
            "truth_s_exp": curve.get("s_exp"),
            "truth_boundary_route": curve.get("boundary_route"),
            "prediction_threshold_status": pred.get("threshold_prediction_status"),
            "s_pred": pred.get("s_pred"),
            "boundary_route_pred": pred.get("boundary_route_pred"),
            "fallback_route_pred": pred.get("fallback_route_pred"),
        }
        for key, value in pred.items():
            if key == "threshold_status":
                row["sparse_curve_threshold_status"] = value
            elif key == "threshold_prediction_status":
                continue
            elif key in {"s_exp", "boundary_route", "s_pred", "boundary_route_pred", "fallback_route_pred"}:
                row[f"sparse_curve_{key}"] = value
            else:
                row[key] = value
        predictions.append(row)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text(
        "".join(json.dumps(common.sanitize_json(row), sort_keys=True, allow_nan=False) + "\n" for row in predictions),
        encoding="utf-8",
    )
    if output_csv:
        common.write_csv(output_csv, predictions)
    print(output_jsonl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
