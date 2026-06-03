#!/usr/bin/env python3
"""Train a lightweight JSON threshold predictor from ground-truth curves."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import threshold_prediction_common as common  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/mnt/nvme1n1/PipeANN-github"))
    parser.add_argument("--curves", type=Path, required=True)
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
    model = common.train_correction_model(curves)
    model.update({
        "created_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "training_curves": str(curves_path),
        "features": [
            "selector_type",
            "sparse graph/prefilter latency points",
            "sparse selectivity positions",
            "median crossing threshold prior by selector/range class",
            "crossing orientation prior by selector/range class",
            "boundary route prior by selector/range class",
            "lower-anchor probe and one-sided boundary extrapolation for high-selectivity windows",
        ],
        "prediction_target": "graph/prefilter avg-latency crossing selectivity",
        "model_artifact_policy": "JSON only; no large binary model weights",
    })
    common.write_json(out / "threshold_predictor_model.json", model)
    (out / "threshold_predictor_model_card.md").write_text(
        "# Threshold Predictor Model Card\n\n"
        f"- Model type: `{model['model_type']}`.\n"
        f"- Training curves: `{model['training_total_cases']}`.\n"
        f"- Crossing cases used for threshold/orientation priors: `{model['training_crossing_cases']}`.\n"
        f"- Sparse-correction cases: `{model.get('training_sparse_correction_cases', 0)}`; correction remains neutral in this model.\n"
        f"- Global correction: `{model['global_correction']:.6f}` (kept neutral; no residual-correction claim).\n"
        "- Inputs: sparse calibration points from graph and prefilter latency curves, selector type, and route-latency differences.\n"
        "- Output: crossing threshold `s_pred`, or a boundary route when no crossing is predicted.\n"
        "- Rationale: with limited historical curves, a transparent sparse piecewise-linear cost model is safer than a high-capacity model. "
        "The learned part is a selector/range median crossing prior, orientation prior, and an exact selector/range boundary-route prior, re-estimated during leave-one-out validation. "
        "Broader boundary priors are used only as final route fallbacks for no-single-threshold cases. "
        "When sparse probes show graph already faster at the lowest sampled selectivity under a low-to-high crossing orientation, the model extrapolates the crossing just left of that probe rather than treating it as a no-crossing boundary.\n"
        "- Limitations: this is not a per-query neural model; it predicts query-set thresholds from original-query calibration summaries. "
        "Low-confidence or boundary cases should fall back to extra calibration points.\n",
        encoding="utf-8",
    )
    print(out / "threshold_predictor_model.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
