#!/usr/bin/env python3
"""Train auditable feature-only threshold predictors.

The feature table must contain only prediction-time observable statistics.  An
oracle file may provide labels (`case_id`, `s_exp`) for training/evaluation, but
the oracle fields are never used as model inputs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import feature_only_threshold_common as common  # noqa: E402


NON_FEATURE_COLUMNS = {
    "case_id",
    "dataset_id",
    "base_bin",
    "query_bin",
    "selector_type",
    "label_family",
    "label_id",
    "feature_only_input",
    "forbidden_inputs",
}
TARGET_COLUMNS = {"s_exp", "oracle_threshold", "threshold_status"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/mnt/nvme1n1/PipeANN-github"))
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, help="JSONL/CSV-like JSONL with case_id and s_exp, produced only after predictions are fixed.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--holdout", choices=["dataset", "label-family", "scale", "none"], default="dataset")
    parser.add_argument("--max-features", type=int, default=16)
    return parser.parse_args()


def load_features(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return common.read_jsonl(path, strict=True)
    raise ValueError("feature table must be JSONL for strict loading")


def load_oracle(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    rows = common.read_jsonl(path, strict=True)
    oracle = {}
    for row in rows:
        case_id = str(row.get("case_id") or "")
        if not case_id:
            raise ValueError(f"oracle row missing case_id: {row}")
        if row.get("s_exp") in (None, ""):
            raise ValueError(f"oracle row missing s_exp: {case_id}")
        oracle[case_id] = row
    return oracle


def is_numeric(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def candidate_features(rows: list[dict[str, Any]]) -> list[str]:
    keys = sorted({key for row in rows for key in row})
    leakage = {finding["feature"] for finding in common.leakage_audit_for_features(keys)}
    candidates = []
    for key in keys:
        if key in NON_FEATURE_COLUMNS or key in TARGET_COLUMNS or key in leakage:
            continue
        values = [row.get(key) for row in rows]
        if any(is_numeric(value) for value in values):
            candidates.append(key)
    return candidates


def matrix(rows: list[dict[str, Any]], features: list[str]) -> np.ndarray:
    X = np.zeros((len(rows), len(features)), dtype=np.float64)
    for i, row in enumerate(rows):
        for j, key in enumerate(features):
            value = row.get(key)
            X[i, j] = float(value) if is_numeric(value) else 0.0
    return X


def split_rows(rows: list[dict[str, Any]], holdout: str) -> list[tuple[str, list[int], list[int]]]:
    if holdout == "none":
        return [("all", list(range(len(rows))), list(range(len(rows))))]
    key = "dataset_id" if holdout == "dataset" else ("label_family" if holdout == "label-family" else "npoints")
    values = sorted({str(row.get(key)) for row in rows})
    splits = []
    for value in values:
        test = [i for i, row in enumerate(rows) if str(row.get(key)) == value]
        train = [i for i in range(len(rows)) if i not in set(test)]
        if train and test:
            splits.append((f"{key}={value}", train, test))
    return splits


def evaluate_predictions(rows: list[dict[str, Any]], y_true: np.ndarray, y_pred: np.ndarray) -> list[dict[str, Any]]:
    out = []
    for row, truth, pred in zip(rows, y_true.tolist(), y_pred.tolist()):
        pred = min(1.0, max(1e-9, float(pred)))
        truth = float(truth)
        rel = abs(pred - truth) / truth if truth > 0 else None
        out.append({
            "case_id": row["case_id"],
            "dataset_id": row.get("dataset_id"),
            "label_family": row.get("label_family"),
            "target_selectivity": row.get("target_selectivity"),
            "s_exp": truth,
            "s_pred": pred,
            "threshold_relative_error": rel,
            "within_5pct": bool(rel is not None and rel <= 0.05),
        })
    return out


def fit_predict_model(name: str, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, Any]:
    if name == "constant_median":
        value = float(np.median(y_train))
        return np.full(X_test.shape[0], value), {"value": value}
    if name == "target_selectivity_identity":
        return X_test[:, 0].copy(), {"note": "first column must be target_selectivity"}
    if name == "log_linear":
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_train)
        model = Ridge(alpha=1.0)
        model.fit(Xs, np.log(np.maximum(y_train, 1e-9)))
        pred = np.exp(model.predict(scaler.transform(X_test)))
        return pred, {"coef": model.coef_.tolist(), "intercept": float(model.intercept_)}
    if name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=20260603)
        model.fit(X_train, y_train)
        return model.predict(X_test), {"feature_importances": model.feature_importances_.tolist()}
    if name == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingRegressor

        model = GradientBoostingRegressor(random_state=20260603, max_depth=2, n_estimators=120, learning_rate=0.05)
        model.fit(X_train, y_train)
        return model.predict(X_test), {"feature_importances": model.feature_importances_.tolist()}
    raise ValueError(f"unknown model {name}")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rels = [float(row["threshold_relative_error"]) for row in rows if row.get("threshold_relative_error") is not None]
    return {
        "case_count": len(rows),
        "within_5pct_count": sum(1 for row in rows if row.get("within_5pct")),
        "within_5pct_rate": (sum(1 for row in rows if row.get("within_5pct")) / len(rows)) if rows else None,
        "median_relative_error": float(np.median(rels)) if rels else None,
        "p90_relative_error": float(np.percentile(rels, 90)) if rels else None,
        "max_relative_error": max(rels) if rels else None,
    }


def permutation_importance(rows: list[dict[str, Any]], features: list[str], y: np.ndarray) -> list[dict[str, Any]]:
    if len(rows) < 8 or len(features) == 0:
        return []
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.inspection import permutation_importance as sklearn_permutation_importance
    except Exception:
        return []
    X = matrix(rows, features)
    model = RandomForestRegressor(n_estimators=100, random_state=20260603, min_samples_leaf=2)
    model.fit(X, y)
    result = sklearn_permutation_importance(model, X, y, n_repeats=5, random_state=20260603)
    records = []
    for idx, feature in enumerate(features):
        records.append({
            "feature": feature,
            "importance_mean": float(result.importances_mean[idx]),
            "importance_std": float(result.importances_std[idx]),
        })
    return sorted(records, key=lambda row: row["importance_mean"], reverse=True)


def write_model_card(out: Path, best: dict[str, Any], selected: list[str], rejected: list[dict[str, Any]]) -> None:
    lines = [
        "# Feature-only Threshold Model Card",
        "",
        f"- Selected model: `{best.get('model_name')}`.",
        f"- Holdout mode: `{best.get('holdout')}`.",
        f"- Within 5% rate: `{best.get('within_5pct_rate')}`.",
        "- Inputs: prediction-time observable dataset/vector/equality-label/index/layout/hardware features only.",
        "- Forbidden inputs: target query latency, graph/prefilter calibration curves, sparse probes, known thresholds, or recall results.",
        "",
        "## Selected Features",
    ]
    for feature in selected:
        lines.append(f"- `{feature}`")
    lines.extend(["", "## Rejected Features"])
    for row in rejected:
        lines.append(f"- `{row['feature']}`: {row['reason']}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    feature_path = args.features if args.features.is_absolute() else repo / args.features
    oracle_path = args.oracle if args.oracle is None or args.oracle.is_absolute() else repo / args.oracle
    out = args.out_dir if args.out_dir.is_absolute() else repo / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    rows = load_features(feature_path)
    oracle = load_oracle(oracle_path)
    common.write_json(out / "training_input_policy.json", {
        "features": str(feature_path),
        "oracle": str(oracle_path) if oracle_path else None,
        "policy": "oracle labels may train/evaluate models but are never feature inputs",
        "forbidden_inputs": sorted(common.LEAKAGE_BLOCKLIST),
    })
    if not oracle:
        common.write_json(out / "feature_only_threshold_validation_summary.json", {
            "status": "PENDING_ORACLE",
            "note": "Feature extraction/model-search scaffold is ready, but no post-prediction oracle threshold file was supplied.",
            "feature_rows": len(rows),
        })
        common.write_json(out / "feature_only_threshold_claim_registry.json", {
            "claims": [
                {"id": "F3_HELD_OUT_THRESHOLD_ACCURACY", "status": "PENDING", "claim": "Requires post-prediction oracle sweep."}
            ]
        })
        print(out)
        return 0
    labeled_rows = []
    targets = []
    for row in rows:
        case_id = str(row.get("case_id"))
        if case_id not in oracle:
            continue
        merged = dict(row)
        merged["s_exp"] = float(oracle[case_id]["s_exp"])
        labeled_rows.append(merged)
        targets.append(float(oracle[case_id]["s_exp"]))
    if len(labeled_rows) < 4:
        raise RuntimeError(f"not enough labeled rows for model search: {len(labeled_rows)}")
    features = candidate_features(labeled_rows)
    if "target_selectivity" in features:
        features = ["target_selectivity"] + [feature for feature in features if feature != "target_selectivity"]
    y = np.asarray(targets, dtype=np.float64)
    models = ["constant_median", "target_selectivity_identity", "log_linear", "random_forest", "gradient_boosting"]
    splits = split_rows(labeled_rows, args.holdout)
    if not splits:
        common.write_json(out / "feature_only_threshold_validation_summary.json", {
            "status": "INSUFFICIENT_SPLITS",
            "holdout": args.holdout,
            "labeled_rows": len(labeled_rows),
            "note": "No non-empty train/test split could be formed. Use more held-out groups or --holdout none for smoke only.",
        })
        common.write_json(out / "feature_only_threshold_claim_registry.json", {
            "claims": [
                {
                    "id": "F3_HELD_OUT_THRESHOLD_ACCURACY",
                    "status": "INSUFFICIENT",
                    "claim": "Held-out feature-only model reaches >=90% cases within 5% threshold error.",
                    "evidence": ["feature_only_threshold_validation_summary.json"],
                    "note": "No valid held-out split.",
                }
            ]
        })
        print(out)
        return 0
    split_records: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for model_name in models:
        for split_name, train_idx, test_idx in splits:
            train_rows = [labeled_rows[i] for i in train_idx]
            test_rows = [labeled_rows[i] for i in test_idx]
            X_train = matrix(train_rows, features)
            X_test = matrix(test_rows, features)
            if model_name == "target_selectivity_identity" and (not features or features[0] != "target_selectivity"):
                continue
            pred, model_info = fit_predict_model(model_name, X_train, y[train_idx], X_test)
            eval_rows = evaluate_predictions(test_rows, y[test_idx], pred)
            summary = summarize(eval_rows)
            split_records.append({
                "model_name": model_name,
                "split": split_name,
                "holdout": args.holdout,
                "model_info": model_info,
                **summary,
            })
            for row in eval_rows:
                row["model_name"] = model_name
                row["split"] = split_name
                prediction_rows.append(row)
    common.write_csv(out / "model_search_results.csv", split_records)
    with (out / "model_search_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in split_records:
            handle.write(json.dumps(common.sanitize_json(row), sort_keys=True, allow_nan=False) + "\n")
    common.write_csv(out / "feature_only_threshold_predictions.csv", prediction_rows)
    with (out / "feature_only_threshold_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in prediction_rows:
            handle.write(json.dumps(common.sanitize_json(row), sort_keys=True, allow_nan=False) + "\n")
    by_model: list[dict[str, Any]] = []
    for model_name in sorted({row["model_name"] for row in prediction_rows}):
        rows_for_model = [row for row in prediction_rows if row["model_name"] == model_name]
        by_model.append({"model_name": model_name, **summarize(rows_for_model)})
    best = max(by_model, key=lambda row: (row.get("within_5pct_rate") or 0.0, -(row.get("median_relative_error") or 999.0)))
    best["holdout"] = args.holdout
    importance = permutation_importance(labeled_rows, features, y)
    selected = [row["feature"] for row in importance[:args.max_features]] if importance else features[:args.max_features]
    selected_set = set(selected)
    rejected = []
    for feature in features:
        if feature not in selected_set:
            rejected.append({"feature": feature, "reason": "low importance or redundant beyond max feature budget"})
    common.write_json(out / "feature_only_threshold_model.json", {
        "model_policy": "feature-only; no target calibration/probe/latency inputs",
        "best_model": best,
        "selected_features": selected,
        "candidate_feature_count": len(features),
        "training_case_count": len(labeled_rows),
    })
    common.write_json(out / "final_selected_features.json", {"features": selected})
    common.write_json(out / "rejected_features.json", {"rejected_features": rejected})
    common.write_csv(out / "feature_importance.csv", importance)
    common.write_json(out / "feature_only_threshold_validation_summary.json", {
        "status": "PASS" if (best.get("within_5pct_rate") or 0.0) >= 0.90 else "FAIL",
        "best_model": best,
        "model_summaries": by_model,
    })
    common.write_json(out / "feature_only_threshold_claim_registry.json", {
        "claims": [
            {
                "id": "F3_HELD_OUT_THRESHOLD_ACCURACY",
                "status": "PASS" if (best.get("within_5pct_rate") or 0.0) >= 0.90 else "FAIL",
                "claim": "Held-out feature-only model reaches >=90% cases within 5% threshold error.",
                "evidence": ["feature_only_threshold_predictions.jsonl", "feature_only_threshold_validation_summary.json"],
                "note": f"best_model={best.get('model_name')}, within_5pct_rate={best.get('within_5pct_rate')}",
            }
        ]
    })
    write_model_card(out / "feature_only_threshold_model_card.md", best, selected, rejected)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
