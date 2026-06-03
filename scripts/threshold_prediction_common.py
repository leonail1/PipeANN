#!/usr/bin/env python3
"""Shared helpers for graph/prefilter selectivity threshold prediction."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


SELECTOR_RE = re.compile(r"(?P<selector>intersect|range)_(?P<bucket>u[0-9A-Za-z.+-]+)")
CALIBRATION_RE = re.compile(r"calibration_(?P<cycle>.+)_(?P<selector>intersect|range)_(?P<bucket>u[0-9A-Za-z.+-]+)\.jsonl$")


def sanitize_json(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: sanitize_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_json(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_json(item) for item in value]
    return value


def read_jsonl(path: Path, *, strict: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        if strict:
            raise FileNotFoundError(path)
        return rows
    errors = "strict" if strict else "replace"
    for line_no, line in enumerate(path.read_text(encoding="utf-8", errors=errors).splitlines(), start=1):
        if line.strip():
            try:
                decoded = json.loads(line)
            except json.JSONDecodeError as exc:
                if strict:
                    raise ValueError(f"malformed JSONL at {path}:{line_no}: {exc}") from exc
                continue
            if strict and not isinstance(decoded, dict):
                raise ValueError(f"JSONL row at {path}:{line_no} is {type(decoded).__name__}, expected object")
            if isinstance(decoded, dict):
                rows.append(decoded)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize_json(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(sanitize_json(row), sort_keys=True, allow_nan=False) + "\n")


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(sanitize_json(value), sort_keys=True, allow_nan=False)
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in keys})


def fnum(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def avg_ms(row: dict[str, Any]) -> float:
    if row.get("avg_latency_ms") not in (None, ""):
        return fnum(row.get("avg_latency_ms"))
    if row.get("latency_avg_ms") not in (None, ""):
        return fnum(row.get("latency_avg_ms"))
    return fnum(row.get("avg_latency_us")) / 1000.0


def p95_ms(row: dict[str, Any]) -> float:
    if row.get("p95_latency_ms") not in (None, ""):
        return fnum(row.get("p95_latency_ms"))
    return fnum(row.get("p95_latency_us")) / 1000.0


def p99_ms(row: dict[str, Any]) -> float:
    if row.get("p99_latency_ms") not in (None, ""):
        return fnum(row.get("p99_latency_ms"))
    return fnum(row.get("p99_latency_us")) / 1000.0


def recall(row: dict[str, Any]) -> float:
    for key in ["recall@10", "recall"]:
        if row.get(key) not in (None, ""):
            return fnum(row.get(key))
    return 0.0


def route_name(row: dict[str, Any]) -> str:
    route = str(row.get("configured_route") or row.get("route") or "")
    if route in {"graph", "prefilter"}:
        return route
    if fnum(row.get("prefilter_count")) > 0:
        return "prefilter"
    if fnum(row.get("graph_count")) > 0:
        return "graph"
    return ""


def parse_bucket_selectivity(bucket: str) -> float | None:
    bucket = str(bucket or "")
    match = re.fullmatch(r"u([0-9]+)", bucket)
    if match:
        return int(match.group(1)) / 100.0
    match = re.fullmatch(r"u([0-9]+(?:\.[0-9]+)?)e-?([0-9]+)", bucket)
    if match:
        return float(match.group(1)) * (10 ** (-int(match.group(2))))
    return None


def row_selectivity(row: dict[str, Any]) -> float | None:
    candidates = fnum(row.get("mean_candidate_count", row.get("candidate_count")), math.nan)
    points = fnum(row.get("live_point_count", row.get("points")), math.nan)
    if math.isfinite(candidates) and math.isfinite(points) and points > 0:
        return max(0.0, min(1.0, candidates / points))
    return parse_bucket_selectivity(str(row.get("bucket") or ""))


def infer_calibration_context(path: Path, row: dict[str, Any]) -> dict[str, Any]:
    selector = str(row.get("selector_type") or "")
    bucket = str(row.get("bucket") or "")
    cycle = str(row.get("cycle") or "")
    name = path.name
    match = CALIBRATION_RE.match(name)
    if match:
        cycle = cycle or match.group("cycle")
        selector = selector or match.group("selector")
        bucket = bucket or match.group("bucket")
    if not selector or not bucket:
        match = SELECTOR_RE.search(str(path))
        if match:
            selector = selector or match.group("selector")
            bucket = bucket or match.group("bucket")
    experiment_dir = path.parent.parent if path.parent.name == "raw" else path.parent
    return {
        "experiment_dir": str(experiment_dir),
        "cycle": cycle or "static",
        "selector_type": selector,
        "bucket": bucket,
    }


def case_id_for(experiment_dir: str, cycle: str, selector_type: str) -> str:
    text = f"{experiment_dir}|{cycle}|{selector_type}"
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    safe_exp = Path(experiment_dir).name
    safe_cycle = re.sub(r"[^A-Za-z0-9_.-]+", "_", cycle or "static")
    return f"{safe_exp}__{safe_cycle}__{selector_type}__{digest}"


def interpolate_threshold(points: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [
        p for p in sorted(points, key=lambda item: fnum(item.get("selectivity")))
        if p.get("graph_avg_ms") not in (None, "") and p.get("prefilter_avg_ms") not in (None, "")
    ]
    if len(usable) < 2:
        return {"threshold_status": "insufficient", "s_exp": None, "boundary_route": None}
    diffs = []
    for point in usable:
        diff = fnum(point["graph_avg_ms"]) - fnum(point["prefilter_avg_ms"])
        diffs.append((fnum(point["selectivity"]), diff, point))
    nonzero = [(s, d, p) for s, d, p in diffs if abs(d) >= 1e-12]
    if not nonzero:
        return {"threshold_status": "boundary", "s_exp": None, "boundary_route": "graph", "s_exp_source": "all_exact_tie_boundary"}
    if all(diff <= 0 for _s, diff, _point in nonzero):
        return {"threshold_status": "boundary", "s_exp": None, "boundary_route": "graph", "s_exp_source": "graph_always_no_slower"}
    if all(diff >= 0 for _s, diff, _point in nonzero):
        return {"threshold_status": "boundary", "s_exp": None, "boundary_route": "prefilter", "s_exp_source": "prefilter_always_no_slower"}

    sign_changes: list[tuple[float, float, float, float]] = []
    for (s0, d0, _p0), (s1, d1, _p1) in zip(diffs, diffs[1:]):
        if abs(d0) < 1e-12 or abs(d1) < 1e-12:
            continue
        if (d0 > 0 and d1 < 0) or (d0 < 0 and d1 > 0):
            sign_changes.append((s0, d0, s1, d1))
    zero_crossings: list[tuple[float, float, float, float, float]] = []
    i = 0
    while i < len(diffs):
        if abs(diffs[i][1]) >= 1e-12:
            i += 1
            continue
        start = i
        while i + 1 < len(diffs) and abs(diffs[i + 1][1]) < 1e-12:
            i += 1
        end = i
        before = next((diffs[j] for j in range(start - 1, -1, -1) if abs(diffs[j][1]) >= 1e-12), None)
        after = next((diffs[j] for j in range(end + 1, len(diffs)) if abs(diffs[j][1]) >= 1e-12), None)
        if before and after and ((before[1] < 0 and after[1] > 0) or (before[1] > 0 and after[1] < 0)):
            low_s = diffs[start][0]
            high_s = diffs[end][0]
            zero_crossings.append(((low_s + high_s) / 2.0, before[1], after[1], low_s, high_s))
        i += 1
    crossing_count = len(sign_changes) + len(zero_crossings)
    if crossing_count > 1:
        return {
            "threshold_status": "multi_crossing",
            "s_exp": None,
            "boundary_route": None,
            "crossing_count": crossing_count,
            "s_exp_source": "multiple_piecewise_sign_changes",
        }
    if zero_crossings:
        selectivity, prev_diff, next_diff, low_s, high_s = zero_crossings[0]
        orientation = "graph_below_prefilter_above" if prev_diff < 0 and next_diff > 0 else "prefilter_below_graph_above"
        return {
            "threshold_status": "crossing",
            "s_exp": selectivity,
            "boundary_route": None,
            "crossing_low_selectivity": low_s,
            "crossing_high_selectivity": high_s,
            "crossing_low_diff_ms": 0.0,
            "crossing_high_diff_ms": 0.0,
            "orientation": orientation,
            "interpolation_span": abs(high_s - low_s),
            "s_exp_source": "exact_zero_plateau_midpoint",
        }
    if sign_changes:
        s0, d0, s1, d1 = sign_changes[0]
        s_exp = s0 + (0.0 - d0) * (s1 - s0) / (d1 - d0)
        orientation = "prefilter_below_graph_above" if d0 > 0 else "graph_below_prefilter_above"
        return {
            "threshold_status": "crossing",
            "s_exp": max(min(s_exp, max(s0, s1)), min(s0, s1)),
            "boundary_route": None,
            "crossing_low_selectivity": s0,
            "crossing_high_selectivity": s1,
            "crossing_low_diff_ms": d0,
            "crossing_high_diff_ms": d1,
            "orientation": orientation,
            "interpolation_span": abs(s1 - s0),
            "s_exp_source": "piecewise_linear_avg_latency",
        }
    return {"threshold_status": "non_monotonic_no_single_crossing", "s_exp": None, "boundary_route": None}


def select_sparse_points(points: list[dict[str, Any]], max_fraction: float = 0.2, min_points: int = 2) -> list[dict[str, Any]]:
    ordered = sorted(points, key=lambda item: fnum(item.get("selectivity")))
    if not ordered:
        return []
    budget = max(1, int(math.floor(len(ordered) * max_fraction)))
    budget = max(min_points, budget)
    budget = min(len(ordered), budget)
    if budget == 1:
        return [ordered[0]]
    indices = sorted({round(i * (len(ordered) - 1) / (budget - 1)) for i in range(budget)})
    return [ordered[int(index)] for index in indices]


def select_points_around_prior(points: list[dict[str, Any]], prior: float | None,
                               max_fraction: float = 0.2, min_points: int = 2) -> list[dict[str, Any]]:
    ordered = sorted(points, key=lambda item: fnum(item.get("selectivity")))
    if not ordered:
        return []
    if prior is None:
        return select_sparse_points(ordered, max_fraction=max_fraction, min_points=min_points)
    probe_prior = probe_prior_for_points(ordered, prior)
    budget = max(1, int(math.floor(len(ordered) * max_fraction)))
    budget = max(min_points, budget)
    budget = min(len(ordered), budget)
    below = [point for point in ordered if fnum(point.get("selectivity")) <= probe_prior]
    above = [point for point in ordered if fnum(point.get("selectivity")) >= probe_prior]
    selected: list[dict[str, Any]] = []
    if below:
        selected.append(below[-1])
    if above and above[0] not in selected:
        selected.append(above[0])
    for point in sorted(ordered, key=lambda item: abs(fnum(item.get("selectivity")) - probe_prior)):
        if len(selected) >= budget:
            break
        if point not in selected:
            selected.append(point)
    return sorted(selected, key=lambda item: fnum(item.get("selectivity")))


def probe_prior_for_points(points: list[dict[str, Any]], prior: float | None) -> float | None:
    if prior is None:
        return None
    ordered = sorted(points, key=lambda item: fnum(item.get("selectivity")))
    if not ordered:
        return prior
    min_selectivity = fnum(ordered[0].get("selectivity"), math.nan)
    if math.isfinite(min_selectivity) and min_selectivity >= 0.1 and prior > 0.4:
        return max(min_selectivity, prior * 0.75)
    return prior


def curve_range_class(curve: dict[str, Any]) -> str:
    min_sel = fnum(curve.get("min_selectivity"), math.nan)
    max_sel = fnum(curve.get("max_selectivity"), math.nan)
    if math.isfinite(min_sel) and min_sel >= 0.1:
        return "high_selectivity_window"
    if math.isfinite(max_sel) and max_sel <= 0.1:
        return "low_selectivity_window"
    return "wide_selectivity_window"


def one_sided_extrapolation(raw_boundary_route: str, orientation: str,
                            sparse_selectivities: list[float]) -> dict[str, Any] | None:
    if not sparse_selectivities:
        return None
    finite = [value for value in sparse_selectivities if math.isfinite(value)]
    if not finite:
        return None
    first = min(finite)
    last = max(finite)
    if orientation == "prefilter_below_graph_above":
        if raw_boundary_route == "graph":
            return {"side": "left", "s_pred": max(0.0, first * 0.88)}
        if raw_boundary_route == "prefilter":
            return {"side": "right", "s_pred": min(1.0, last * 1.12)}
    if orientation == "graph_below_prefilter_above":
        if raw_boundary_route == "prefilter":
            return {"side": "left", "s_pred": max(0.0, first * 0.88)}
        if raw_boundary_route == "graph":
            return {"side": "right", "s_pred": min(1.0, last * 1.12)}
    return None


def sparse_linear_prediction(points: list[dict[str, Any]], correction: float = 1.0,
                             prior: float | None = None, orientation_prior: str | None = None,
                             boundary_route_prior: str | None = None) -> dict[str, Any]:
    sparse = select_points_around_prior(points, prior)
    raw = interpolate_threshold(sparse)
    sparse_selectivities = [fnum(point.get("selectivity")) for point in sparse]
    result = dict(raw)
    result["sparse_points_used"] = len(sparse)
    result["sparse_selectivities"] = sparse_selectivities
    result["threshold_prior"] = prior
    result["probe_prior"] = probe_prior_for_points(points, prior)
    result["prediction_method"] = "prior_guided_sparse_piecewise_linear" if prior is not None else "sparse_piecewise_linear"
    if raw.get("threshold_status") == "crossing" and raw.get("s_exp") is not None:
        result["s_pred"] = max(0.0, min(1.0, fnum(raw["s_exp"]) * correction))
        result["threshold_prediction_status"] = "crossing"
    elif raw.get("threshold_status") == "boundary":
        if boundary_route_prior and raw.get("boundary_route") == boundary_route_prior:
            result["s_pred"] = None
            result["threshold_prediction_status"] = "boundary"
            result["boundary_route_pred"] = boundary_route_prior
            result["prediction_method"] = "boundary_prior_confirmed_by_sparse"
            return result
        if prior is not None:
            orientation = orientation_prior or "prefilter_below_graph_above"
            extrapolated = one_sided_extrapolation(str(raw.get("boundary_route") or ""), orientation, sparse_selectivities)
            if extrapolated:
                side = str(extrapolated["side"])
                result["s_pred"] = extrapolated["s_pred"]
                result["threshold_prediction_status"] = f"crossing_{side}_boundary_extrapolation"
                result["orientation"] = orientation
                result["sparse_boundary_route"] = raw.get("boundary_route")
                result["prediction_method"] = f"{side}_boundary_extrapolation_after_sparse_{raw.get('boundary_route')}_boundary"
                return result
            result["s_pred"] = max(0.0, min(1.0, prior * correction))
            result["threshold_prediction_status"] = "crossing_prior_fallback"
            result["orientation"] = orientation
            result["sparse_boundary_route"] = raw.get("boundary_route")
            result["prediction_method"] = "prior_fallback_after_sparse_boundary"
            return result
        result["s_pred"] = None
        result["threshold_prediction_status"] = "boundary"
        result["boundary_route_pred"] = raw.get("boundary_route")
    else:
        # Preserve the "no single threshold" status; threshold fallback is explicit.
        if prior is not None:
            result["s_pred"] = max(0.0, min(1.0, prior * correction))
            result["threshold_prediction_status"] = "crossing_prior_fallback"
            result["orientation"] = orientation_prior or "prefilter_below_graph_above"
            result["prediction_method"] = "prior_fallback_after_sparse_no_single_threshold"
            return result
        if sparse:
            last = sparse[-1]
            route = "graph" if fnum(last.get("graph_avg_ms"), math.inf) <= fnum(last.get("prefilter_avg_ms"), math.inf) else "prefilter"
        else:
            route = "graph"
        result["s_pred"] = None
        result["threshold_prediction_status"] = str(raw.get("threshold_status") or "insufficient")
        result["fallback_route_pred"] = route
        result["prediction_method"] = "sparse_endpoint_route_fallback_no_single_threshold"
    return result


def train_correction_model(curves: list[dict[str, Any]]) -> dict[str, Any]:
    ratios: list[float] = []
    by_selector: dict[str, list[float]] = {}
    priors: dict[str, list[float]] = {}
    orientations: dict[str, list[str]] = {}
    boundary_routes: dict[str, list[str]] = {}
    crossing_case_count = 0
    for curve in curves:
        key = f"{curve.get('selector_type','')}|{curve_range_class(curve)}"
        selector_key = str(curve.get("selector_type") or "")
        if curve.get("threshold_status") == "boundary" and curve.get("boundary_route"):
            boundary_routes.setdefault(key, []).append(str(curve.get("boundary_route")))
            boundary_routes.setdefault(f"{selector_key}|*", []).append(str(curve.get("boundary_route")))
            boundary_routes.setdefault("*|*", []).append(str(curve.get("boundary_route")))
        if curve.get("threshold_status") != "crossing" or curve.get("s_exp") in (None, ""):
            continue
        crossing_case_count += 1
        priors.setdefault(key, []).append(fnum(curve.get("s_exp")))
        priors.setdefault(f"{selector_key}|*", []).append(fnum(curve.get("s_exp")))
        priors.setdefault("*|*", []).append(fnum(curve.get("s_exp")))
        if curve.get("orientation"):
            orientations.setdefault(key, []).append(str(curve.get("orientation")))
            orientations.setdefault(f"{selector_key}|*", []).append(str(curve.get("orientation")))
            orientations.setdefault("*|*", []).append(str(curve.get("orientation")))
        pred = sparse_linear_prediction(curve.get("curve_points", []), correction=1.0, prior=None)
        if pred.get("threshold_prediction_status") != "crossing" or pred.get("s_pred") in (None, ""):
            continue
        raw = fnum(pred.get("s_pred"))
        truth = fnum(curve.get("s_exp"))
        if raw > 0 and truth > 0:
            ratio = truth / raw
            ratios.append(ratio)
            by_selector.setdefault(str(curve.get("selector_type") or ""), []).append(ratio)

    def median(values: list[float], default: float = 1.0) -> float:
        if not values:
            return default
        ordered = sorted(values)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[mid]
        return 0.5 * (ordered[mid - 1] + ordered[mid])

    def route_mode(values: list[str]) -> str:
        if not values:
            return "graph"
        return max(sorted(set(values)), key=values.count)
    def orientation_mode(values: list[str]) -> str:
        if not values:
            return "prefilter_below_graph_above"
        return max(sorted(set(values)), key=values.count)
    return {
        "model_type": "prior_guided_sparse_linear_threshold",
        "global_correction": 1.0,
        "selector_corrections": {key: 1.0 for key in by_selector},
        "threshold_priors": {key: median(vals) for key, vals in priors.items()},
        "orientation_priors": {key: orientation_mode(vals) for key, vals in orientations.items()},
        "boundary_route_priors": {key: route_mode(vals) for key, vals in boundary_routes.items()},
        "training_crossing_cases": crossing_case_count,
        "training_sparse_correction_cases": len(ratios),
        "training_total_cases": len(curves),
    }


def predict_with_model(curve: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    selector = str(curve.get("selector_type") or "")
    range_class = curve_range_class(curve)
    prior_key = f"{selector}|{range_class}"
    prior = model.get("threshold_priors", {}).get(prior_key)
    if prior is None:
        prior = model.get("threshold_priors", {}).get(f"{selector}|*")
    if prior is None:
        prior = model.get("threshold_priors", {}).get("*|*")
    orientation_prior = model.get("orientation_priors", {}).get(prior_key)
    if orientation_prior is None:
        orientation_prior = model.get("orientation_priors", {}).get(f"{selector}|*")
    if orientation_prior is None:
        orientation_prior = model.get("orientation_priors", {}).get("*|*")
    # Boundary confirmation must stay exact. A selector/global boundary fallback
    # can incorrectly override a well-supported crossing prior on wide curves.
    boundary_route_prior = model.get("boundary_route_priors", {}).get(prior_key)
    fallback_route_prior = boundary_route_prior
    if fallback_route_prior is None:
        fallback_route_prior = model.get("boundary_route_priors", {}).get(f"{selector}|*")
    if fallback_route_prior is None:
        fallback_route_prior = model.get("boundary_route_priors", {}).get("*|*")
    correction = fnum(model.get("selector_corrections", {}).get(selector), fnum(model.get("global_correction"), 1.0))
    pred = sparse_linear_prediction(curve.get("curve_points", []), correction=correction,
                                    prior=fnum(prior) if prior is not None else None,
                                    orientation_prior=str(orientation_prior) if orientation_prior else None,
                                    boundary_route_prior=str(boundary_route_prior) if boundary_route_prior else None)
    pred["model_correction"] = correction
    pred["prior_key"] = prior_key
    pred["range_class"] = range_class
    crossing_like = {"crossing", "crossing_prior_fallback", "crossing_left_boundary_extrapolation", "crossing_right_boundary_extrapolation"}
    if pred.get("threshold_prediction_status") not in (crossing_like | {"boundary"}):
        if fallback_route_prior:
            pred["fallback_route_pred"] = fallback_route_prior
    return pred


def route_from_threshold(selectivity: float, prediction: dict[str, Any]) -> str:
    status = prediction.get("threshold_prediction_status")
    if status in {"crossing_prior_fallback", "crossing_left_boundary_extrapolation", "crossing_right_boundary_extrapolation"}:
        status = "crossing"
    if status not in {"boundary", "crossing"} and prediction.get("fallback_route_pred"):
        return str(prediction.get("fallback_route_pred"))
    if status == "boundary":
        return str(prediction.get("boundary_route_pred") or prediction.get("boundary_route") or "graph")
    threshold = fnum(prediction.get("s_pred"), math.nan)
    orientation = str(prediction.get("orientation") or "prefilter_below_graph_above")
    if not math.isfinite(threshold):
        return "graph"
    if orientation == "graph_below_prefilter_above":
        return "graph" if selectivity <= threshold else "prefilter"
    return "prefilter" if selectivity <= threshold else "graph"


def relative_error(pred: float | None, truth: float | None) -> float | None:
    if pred is None or truth in (None, 0):
        return None
    return abs(pred - truth) / abs(truth)
