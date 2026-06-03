#!/usr/bin/env python3
"""Generate small ARIS/PPT artifacts for the v100 early-PQ triggered run.

This is post-processing only. It reads the early-PQ no-retrain baseline
comparison, triggered-maintenance targeted evidence, optional chain evidence,
and existing v100 layout/space audits. It writes only small JSON/CSV/MD/SVG
artifacts that are safe to commit.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
MAX_OUTPUT_ARTIFACT_BYTES = 2_000_000


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def resolve(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, sort_keys=True)
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


def copy_if_small(src: Path, dst: Path, max_bytes: int = 2_000_000) -> bool:
    if not src.exists() or src.stat().st_size > max_bytes:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix == ".csv":
        text = src.read_text(encoding="utf-8")
        dst.write_text(text.replace("\r\n", "\n").replace("\r", "\n"), encoding="utf-8")
        return True
    shutil.copy2(src, dst)
    return True


def validate_output_size(out: Path, max_bytes: int = MAX_OUTPUT_ARTIFACT_BYTES) -> None:
    oversized = [
        (path, path.stat().st_size)
        for path in out.rglob("*")
        if path.is_file() and path.stat().st_size > max_bytes
    ]
    if oversized:
        details = ", ".join(f"{path.relative_to(out)}={size}" for path, size in oversized[:10])
        raise RuntimeError(f"generated artifact exceeds {max_bytes} bytes: {details}")


def fnum(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def avg_ms(row: dict[str, Any]) -> float:
    if row.get("avg_latency_ms") not in (None, ""):
        return fnum(row, "avg_latency_ms")
    return fnum(row, "avg_latency_us") / 1000.0


def p95_ms(row: dict[str, Any]) -> float:
    if row.get("p95_latency_ms") not in (None, ""):
        return fnum(row, "p95_latency_ms")
    return fnum(row, "p95_latency_us") / 1000.0


def recall(row: dict[str, Any]) -> float:
    return fnum(row, "recall@10", fnum(row, "recall"))


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def source_tag(row: dict[str, Any], source: str) -> dict[str, Any]:
    tagged = dict(row)
    tagged["evidence_source"] = source
    return tagged


def summarize_selected(rows: list[dict[str, Any]], latency_ms: float, recall_floor: float) -> dict[str, Any]:
    fail_rows = []
    for row in rows:
        row_p95_ms = p95_ms(row)
        if recall(row) < recall_floor or avg_ms(row) >= latency_ms or (row_p95_ms > 0 and row_p95_ms >= latency_ms):
            fail_rows.append(str(row.get("case_id") or ""))
    return {
        "rows": len(rows),
        "recall_present": sum(1 for row in rows if recall(row) > 0),
        "avg_latency_present": sum(1 for row in rows if avg_ms(row) > 0),
        "p95_latency_present": sum(1 for row in rows if p95_ms(row) > 0),
        "recall_pass": sum(1 for row in rows if recall(row) >= recall_floor),
        "avg_lt_latency_ms": sum(1 for row in rows if avg_ms(row) < latency_ms),
        "p95_lt_latency_ms": sum(1 for row in rows if p95_ms(row) and p95_ms(row) < latency_ms),
        "min_recall": min((recall(row) for row in rows), default=0.0),
        "max_avg_latency_ms": max((avg_ms(row) for row in rows), default=0.0),
        "max_p95_latency_ms": max((p95_ms(row) for row in rows), default=0.0),
        "cycles": sorted({int(row.get("cycle_idx") or 0) for row in rows}),
        "fail_rows": fail_rows[:20],
        "fail_row_count": len(fail_rows),
    }


def summarize_maintenance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "cycles": sorted({int(row.get("cycle_idx") or 0) for row in rows}),
        "max_delete_ms_per_vector": max((fnum(row, "delete_ms_per_vector") for row in rows), default=0.0),
        "max_delete_merge_s": max((fnum(row, "delete_merge_s") for row in rows), default=0.0),
        "max_insert_merge_s": max((fnum(row, "insert_merge_s") for row in rows), default=0.0),
        "max_maintenance_build_wall_s": max((fnum(row, "maintenance_build_wall_s") for row in rows), default=0.0),
        "max_pq_train_wall_s": max((fnum(row, "pq_train_wall_s") for row in rows), default=0.0),
        "max_pq_recode_wall_s": max((fnum(row, "pq_recode_wall_s") for row in rows), default=0.0),
    }


def summarize_background_interference(bg_dir: Path | None, latency_ms: float, recall_floor: float) -> dict[str, Any]:
    if bg_dir is None:
        return {"status": "NEEDS_EVIDENCE", "rows": 0}
    summary = read_json(bg_dir / "background_interference_summary.json")
    foreground_rows = [
        row for row in read_jsonl(bg_dir / "foreground_search_results.jsonl")
        if row.get("condition") == "during_background"
    ]
    if not summary or not foreground_rows:
        return {
            "status": "NEEDS_EVIDENCE",
            "rows": len(foreground_rows),
            "background_interference_dir": rel(bg_dir),
            "raw_foreground_rows_present": bool(foreground_rows),
            "summary_present": bool(summary),
        }
    during_rows = len(foreground_rows)
    required_rows = int(summary.get("required_during_rows") or during_rows)
    recall_pass = sum(1 for row in foreground_rows if recall(row) >= recall_floor)
    avg_pass = sum(1 for row in foreground_rows if avg_ms(row) < latency_ms)
    p95_pass = sum(1 for row in foreground_rows if p95_ms(row) > 0 and p95_ms(row) < latency_ms)
    max_avg = max((avg_ms(row) for row in foreground_rows), default=0.0)
    max_p95 = max((p95_ms(row) for row in foreground_rows), default=0.0)
    foreground_status = (
        "PASS"
        if during_rows > 0
        and during_rows >= required_rows
        and recall_pass == during_rows
        and avg_pass == during_rows
        and p95_pass == during_rows
        and max_avg < latency_ms
        and max_p95 < latency_ms
        else "NEEDS_EVIDENCE"
    )
    source_claim_status = str(summary.get("claim_status") or "")
    source_four_k_status = str(summary.get("four_k_read_size_status") or "")
    layout_gate_caveat = (
        bool(source_claim_status and source_claim_status != "PASS")
        or bool(source_four_k_status and source_four_k_status != "PASS")
    )
    status = "PASS_WITH_CAVEAT" if foreground_status == "PASS" and layout_gate_caveat else foreground_status
    return {
        "status": status,
        "foreground_overlap_status": foreground_status,
        "layout_gate_caveat": layout_gate_caveat,
        "background_interference_dir": rel(bg_dir),
        "rows": during_rows,
        "required_rows": required_rows,
        "recall_pass": recall_pass,
        "avg_lt_latency_ms_pass": avg_pass,
        "p95_lt_latency_ms_pass": p95_pass,
        "max_avg_latency_ms": max_avg,
        "max_p95_latency_ms": max_p95,
        "background_cpu_cap": summary.get("background_cpu_cap"),
        "background_cpu_range": summary.get("background_cpu_range"),
        "foreground_cpu_range": summary.get("foreground_cpu_range"),
        "background_elapsed_wall_s": summary.get("background_elapsed_wall_s"),
        "background_pq_train_wall_s": summary.get("background_pq_train_wall_s"),
        "background_pq_recode_wall_s": summary.get("background_pq_recode_wall_s"),
        "source_claim_status": source_claim_status,
        "source_four_k_read_size_status": source_four_k_status,
        "raw_foreground_rows_present": True,
    }


def overall_status(statuses: list[str]) -> str:
    if any(status in {"NEEDS_EVIDENCE", "IN_PROGRESS"} for status in statuses):
        return "NEEDS_EVIDENCE"
    if any(status in {"PASS_WITH_CAVEAT", "EVIDENCE_PRIOR"} for status in statuses):
        return "PASS_WITH_CAVEAT"
    return "PASS"


def svg_text(x: float, y: float, text: Any, size: int = 13, weight: str = "400", anchor: str = "start") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="#111827">'
        f"{html.escape(str(text))}</text>"
    )


def write_svg(path: Path, width: int, height: int, body: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="#ffffff"/>\n'
        + "\n".join(body)
        + "\n</svg>\n",
        encoding="utf-8",
    )


def plot_latency_recall(out: Path, baseline: list[dict[str, Any]], triggered: list[dict[str, Any]],
                        latency_ms: float, recall_floor: float) -> Path | None:
    if not baseline and not triggered:
        return None
    width, height = 930, 520
    left, right, top, bottom = 80, 35, 62, 76
    plot_w, plot_h = width - left - right, height - top - bottom
    xs = [avg_ms(row) for row in baseline + triggered]
    ys = [recall(row) for row in baseline + triggered]
    x_min = 0.0
    x_max = max(latency_ms * 1.18, max(xs, default=latency_ms) * 1.05)
    y_min = min(recall_floor - 0.7, min(ys, default=recall_floor) - 0.2)
    y_max = max(100.1, max(ys, default=100.0) + 0.2)
    sx = lambda x: left + (x - x_min) / max(x_max - x_min, 1e-9) * plot_w
    sy = lambda y: top + (y_max - y) / max(y_max - y_min, 1e-9) * plot_h
    body = [
        svg_text(width / 2, 32, "Early-PQ drift: no-retrain failures vs triggered maintenance", 20, "700", "middle"),
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827"/>',
        f'<line x1="{sx(latency_ms):.1f}" y1="{top}" x2="{sx(latency_ms):.1f}" y2="{top + plot_h}" stroke="#dc2626" stroke-dasharray="7 5"/>',
        f'<line x1="{left}" y1="{sy(recall_floor):.1f}" x2="{left + plot_w}" y2="{sy(recall_floor):.1f}" stroke="#111827" stroke-dasharray="3 5"/>',
        svg_text(left + plot_w / 2, height - 26, "Avg latency (ms)", 13, "400", "middle"),
        svg_text(20, top + plot_h / 2, "Recall@10 (%)", 13, "400", "middle"),
        svg_text(width - 260, 78, "baseline no-retrain", 13),
        '<circle cx="650" cy="74" r="5" fill="#dc2626" fill-opacity="0.8"/>',
        svg_text(width - 260, 100, "triggered retrain", 13),
        '<circle cx="650" cy="96" r="5" fill="#0f766e" fill-opacity="0.8"/>',
    ]
    for value in [0, 2, 4, 6, 8, 10, 12, 14]:
        if value <= x_max + 1e-6:
            body.append(svg_text(sx(value), top + plot_h + 24, value, 12, "400", "middle"))
    for value in [98, 99, 100]:
        if y_min <= value <= y_max:
            body.append(svg_text(left - 10, sy(value) + 4, value, 12, "400", "end"))
    for row in baseline:
        body.append(f'<circle cx="{sx(avg_ms(row)):.1f}" cy="{sy(recall(row)):.1f}" r="4.6" fill="#dc2626" fill-opacity="0.78"/>')
    for row in triggered:
        body.append(f'<circle cx="{sx(avg_ms(row)):.1f}" cy="{sy(recall(row)):.1f}" r="4.6" fill="#0f766e" fill-opacity="0.78"/>')
    path = out / "figures" / "early_pq_latency_recall_targeted.svg"
    write_svg(path, width, height, body)
    return path


def plot_chain_latency(out: Path, rows: list[dict[str, Any]], latency_ms: float) -> Path | None:
    if not rows:
        return None
    by_cycle: dict[int, dict[str, float]] = {}
    for row in rows:
        cycle = int(row.get("cycle_idx") or 0)
        item = by_cycle.setdefault(cycle, {"avg": 0.0, "p95": 0.0})
        item["avg"] = max(item["avg"], avg_ms(row))
        item["p95"] = max(item["p95"], p95_ms(row))
    cycles = sorted(by_cycle)
    width, height = 860, 430
    left, right, top, bottom = 74, 30, 54, 68
    plot_w, plot_h = width - left - right, height - top - bottom
    max_y = max(latency_ms * 1.1, *(by_cycle[c]["p95"] for c in cycles), 1.0)
    sx = lambda idx: left + (idx + 0.5) / max(len(cycles), 1) * plot_w
    sy = lambda y: top + plot_h - y / max_y * plot_h
    body = [
        svg_text(width / 2, 30, "Triggered chain partial/full latency by cycle", 19, "700", "middle"),
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{sy(latency_ms):.1f}" x2="{left + plot_w}" y2="{sy(latency_ms):.1f}" stroke="#dc2626" stroke-dasharray="6 4"/>',
        svg_text(left + plot_w / 2, height - 24, "Cycle", 13, "400", "middle"),
        svg_text(18, top + plot_h / 2, "Latency (ms)", 13, "400", "middle"),
        '<rect x="640" y="58" width="16" height="10" fill="#0f766e"/>',
        svg_text(664, 68, "max avg", 12),
        '<rect x="640" y="80" width="16" height="10" fill="#f59e0b"/>',
        svg_text(664, 90, "max p95", 12),
    ]
    bar_w = min(42, plot_w / max(len(cycles), 1) * 0.28)
    for idx, cycle in enumerate(cycles):
        x = sx(idx)
        avg_h = by_cycle[cycle]["avg"] / max_y * plot_h
        p95_h = by_cycle[cycle]["p95"] / max_y * plot_h
        body.append(f'<rect x="{x - bar_w - 2:.1f}" y="{top + plot_h - avg_h:.1f}" width="{bar_w:.1f}" height="{avg_h:.1f}" fill="#0f766e"/>')
        body.append(f'<rect x="{x + 2:.1f}" y="{top + plot_h - p95_h:.1f}" width="{bar_w:.1f}" height="{p95_h:.1f}" fill="#f59e0b"/>')
        body.append(svg_text(x, top + plot_h + 22, cycle, 12, "400", "middle"))
        body.append(svg_text(x, min(sy(by_cycle[cycle]["p95"]) - 8, top + plot_h - 6), f'{by_cycle[cycle]["avg"]:.2f}/{by_cycle[cycle]["p95"]:.2f}', 10, "400", "middle"))
    path = out / "figures" / "early_pq_chain_latency_by_cycle.svg"
    write_svg(path, width, height, body)
    return path


def plot_maintenance(out: Path, rows: list[dict[str, Any]]) -> Path | None:
    if not rows:
        return None
    cycles = [int(row.get("cycle_idx") or 0) for row in rows]
    width, height = 960, 470
    left, right, top, bottom = 78, 34, 58, 76
    plot_w, plot_h = width - left - right, height - top - bottom
    metrics = [
        ("PQ train", "pq_train_wall_s", "#0f766e"),
        ("PQ recode", "pq_recode_wall_s", "#14b8a6"),
        ("build", "maintenance_build_wall_s", "#64748b"),
        ("delete merge", "delete_merge_s", "#f59e0b"),
        ("insert merge", "insert_merge_s", "#cbd5e1"),
    ]
    max_y = max((sum(fnum(row, key) for _, key, _ in metrics) for row in rows), default=1.0)
    max_y = max(max_y * 1.12, 1.0)
    sx = lambda idx: left + (idx + 0.5) / max(len(rows), 1) * plot_w
    body = [
        svg_text(width / 2, 32, "Triggered maintenance cost per 60% replacement cycle", 19, "700", "middle"),
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827"/>',
        svg_text(left + plot_w / 2, height - 24, "Cycle", 13, "400", "middle"),
        svg_text(18, top + plot_h / 2, "Wall time (s)", 13, "400", "middle"),
    ]
    for idx, (name, _, color) in enumerate(metrics):
        x = 640 + (idx % 2) * 150
        y = 58 + (idx // 2) * 22
        body.append(f'<rect x="{x}" y="{y - 11}" width="16" height="12" fill="{color}"/>')
        body.append(svg_text(x + 23, y, name, 12))
    bar_w = min(64, plot_w / max(len(rows), 1) * 0.42)
    for idx, row in enumerate(rows):
        x = sx(idx)
        y_cursor = top + plot_h
        total = 0.0
        for _, key, color in metrics:
            value = fnum(row, key)
            total += value
            h = value / max_y * plot_h
            y_cursor -= h
            body.append(f'<rect x="{x - bar_w / 2:.1f}" y="{y_cursor:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}"/>')
        body.append(svg_text(x, top + plot_h + 22, cycles[idx], 12, "400", "middle"))
        body.append(svg_text(x, max(y_cursor - 8, top + 12), f"{total:.1f}s", 10, "400", "middle"))
    path = out / "figures" / "early_pq_maintenance_costs.svg"
    write_svg(path, width, height, body)
    return path


def load_chain_selected(chain_dir: Path | None) -> list[dict[str, Any]]:
    if chain_dir is None:
        return []
    for rel_path in ["optimized_dynamic_update_results.jsonl", "raw/chain_selected_route_l.jsonl"]:
        rows = read_jsonl(chain_dir / rel_path)
        if rows:
            return rows
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targeted-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--chain-dir", type=Path, default=None)
    parser.add_argument("--smoke-dir", type=Path, default=None)
    parser.add_argument("--space-artifacts-dir", type=Path, default=None)
    parser.add_argument("--background-interference-dir", type=Path, default=None)
    parser.add_argument("--latency-ms", type=float, default=10.0)
    parser.add_argument("--recall-floor", type=float, default=98.0)
    parser.add_argument("--expected-chain-cycles", type=int, default=5)
    parser.add_argument("--expected-chain-rows", type=int, default=30)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    targeted_dir = resolve(args.targeted_dir)
    chain_dir = resolve(args.chain_dir)
    smoke_dir = resolve(args.smoke_dir)
    space_dir = resolve(args.space_artifacts_dir)
    background_interference_dir = resolve(args.background_interference_dir)
    out = resolve(args.out_dir)
    assert targeted_dir is not None and out is not None

    if out.exists():
        raise RuntimeError(f"--out-dir already exists; choose a fresh output directory: {out}")
    out.mkdir(parents=True)

    targeted_summary = read_json(targeted_dir / "summary.json")
    chain_summary = read_json(chain_dir / "summary.json") if chain_dir else {}
    smoke_summary = read_json(smoke_dir / "summary.json") if smoke_dir else {}

    baseline_rows = [source_tag(row, "targeted_no_retrain_baseline")
                     for row in read_jsonl(targeted_dir / "targeted_latency_profile.jsonl")]
    targeted_rows = [source_tag(row, "targeted_triggered_retrain")
                     for row in read_jsonl(targeted_dir / "optimized_dynamic_update_results.jsonl")]
    chain_rows = [source_tag(row, "chain_triggered_retrain") for row in load_chain_selected(chain_dir)]
    smoke_rows = [source_tag(row, "smoke_triggered_retrain")
                  for row in read_jsonl(smoke_dir / "optimized_dynamic_update_results.jsonl")] if smoke_dir else []
    maintenance_rows = [source_tag(row, "chain_triggered_retrain") for row in read_jsonl(chain_dir / "early_pq_delete_insert_maintenance.jsonl")] if chain_dir else []
    if not maintenance_rows and chain_dir:
        maintenance_rows = [source_tag(row, "chain_triggered_retrain") for row in read_jsonl(chain_dir / "raw/chain_maintenance.jsonl")]
    interference_rows = [source_tag(row, "targeted_prior_background")
                         for row in read_jsonl(targeted_dir / "pq_retrain_interference_profile.jsonl")]
    if chain_dir:
        interference_rows += [source_tag(row, "chain_triggered_timing")
                              for row in read_jsonl(chain_dir / "pq_retrain_interference_profile.jsonl")]
    background_interference_metrics = summarize_background_interference(
        background_interference_dir, args.latency_ms, args.recall_floor
    )
    if background_interference_dir:
        interference_rows += [
            source_tag(row, "early_pq_background_foreground")
            for row in read_jsonl(background_interference_dir / "foreground_search_results.jsonl")
        ]
        interference_rows += [
            source_tag(row, "early_pq_background_maintenance")
            for row in read_jsonl(background_interference_dir / "background_maintenance_results.jsonl")
        ]
        if background_interference_metrics.get("rows"):
            interference_rows.append(source_tag(background_interference_metrics, "early_pq_background_summary"))
    compare_rows = [source_tag(row, "targeted_compare")
                    for row in read_jsonl(targeted_dir / "pq_drift_strategy_compare.jsonl")]
    if chain_dir:
        compare_rows += [source_tag(row, "chain_compare")
                         for row in read_jsonl(chain_dir / "pq_drift_strategy_compare.jsonl")]

    dynamic_rows = targeted_rows + chain_rows
    write_jsonl(out / "targeted_latency_profile.jsonl", baseline_rows)
    write_csv(out / "targeted_latency_profile.csv", baseline_rows)
    write_jsonl(out / "optimized_dynamic_update_results.jsonl", dynamic_rows)
    write_csv(out / "optimized_dynamic_update_results.csv", dynamic_rows)
    write_jsonl(out / "pq_drift_strategy_compare.jsonl", compare_rows)
    write_csv(out / "pq_drift_strategy_compare.csv", compare_rows)
    write_jsonl(out / "pq_retrain_interference_profile.jsonl", interference_rows)
    write_csv(out / "pq_retrain_interference_profile.csv", interference_rows)
    write_jsonl(out / "early_pq_delete_insert_maintenance.jsonl", maintenance_rows)
    write_csv(out / "early_pq_delete_insert_maintenance.csv", maintenance_rows)

    copied_audits: list[str] = []
    if space_dir:
        for name in ["index_space_audit.md", "label_sidecar_layout_audit.md", "index_space_audit.csv", "index_space_audit.jsonl"]:
            if copy_if_small(space_dir / name, out / name):
                copied_audits.append(name)
    if not (out / "index_space_audit.md").exists():
        (out / "index_space_audit.md").write_text(
            "# Early-PQ Index Space Audit\n\n"
            "No space-artifacts directory was provided. Reuse the reviewed v100 Supersector32K space audit before making a space claim.\n",
            encoding="utf-8",
        )
    if not (out / "label_sidecar_layout_audit.md").exists():
        (out / "label_sidecar_layout_audit.md").write_text(
            "# Early-PQ Label Sidecar Layout Audit\n\n"
            "No layout audit directory was provided. Reuse the reviewed v100 label-sidecar audit before making a layout claim.\n",
            encoding="utf-8",
        )

    baseline_metrics = summarize_selected(baseline_rows, args.latency_ms, args.recall_floor)
    targeted_metrics = summarize_selected(targeted_rows, args.latency_ms, args.recall_floor)
    chain_metrics = summarize_selected(chain_rows, args.latency_ms, args.recall_floor)
    smoke_metrics = summarize_selected(smoke_rows, args.latency_ms, args.recall_floor)
    maintenance_metrics = summarize_maintenance(maintenance_rows)

    completed_cycles = int(chain_summary.get("completed_cycles") or len(maintenance_metrics["cycles"]))
    expected_cycle_set = set(range(1, args.expected_chain_cycles + 1))
    observed_cycle_set = set(chain_metrics["cycles"])
    chain_full_pass = (
        completed_cycles >= args.expected_chain_cycles
        and expected_cycle_set.issubset(observed_cycle_set)
        and chain_metrics["rows"] >= args.expected_chain_rows
        and not chain_metrics["fail_rows"]
    )
    chain_status = "PASS" if chain_full_pass else ("IN_PROGRESS" if args.allow_partial else "NEEDS_EVIDENCE")
    target_pass = (
        baseline_metrics["rows"] > 0
        and baseline_metrics["max_avg_latency_ms"] >= args.latency_ms
        and targeted_metrics["rows"] > 0
        and not targeted_metrics["fail_rows"]
    )

    figures: list[str] = []
    for path in [
        plot_latency_recall(out, baseline_rows, targeted_rows, args.latency_ms, args.recall_floor),
        plot_chain_latency(out, chain_rows, args.latency_ms),
        plot_maintenance(out, maintenance_rows),
    ]:
        if path:
            figures.append(rel(path))

    maintenance_timing_status = "PASS" if maintenance_rows else "NEEDS_EVIDENCE"
    early_pq_overlap_status = str(background_interference_metrics.get("status") or "NEEDS_EVIDENCE")
    prior_interference_rows = [
        row for row in interference_rows
        if str(row.get("evidence_source") or "").startswith("targeted_prior")
    ]
    if early_pq_overlap_status in {"PASS", "PASS_WITH_CAVEAT"}:
        foreground_interference_status = "PASS_EARLY_PQ_OVERLAP"
        if maintenance_timing_status != "PASS":
            e4_status = "NEEDS_EVIDENCE"
            e4_caveat = ""
        elif early_pq_overlap_status == "PASS_WITH_CAVEAT":
            e4_status = "PASS_WITH_CAVEAT"
            e4_caveat = (
                "Foreground overlap/timing passes for the early-PQ run, but the source background "
                "summary also contains unrelated layout/4KB metadata gate failures. Keep layout/4KB "
                "claims scoped to the reused v100 layout audit."
            )
        else:
            e4_status = "PASS"
            e4_caveat = ""
    elif prior_interference_rows:
        foreground_interference_status = "PASS_BY_PRIOR_EVIDENCE"
        e4_status = (
            "PASS_WITH_CAVEAT"
            if maintenance_timing_status == "PASS"
            else "NEEDS_EVIDENCE"
        )
        e4_caveat = "Chain timing is measured on the early-PQ triggered run, but foreground interference is reused prior v100 low-core background evidence, not an overlapped early-PQ chain rerun."
    else:
        foreground_interference_status = "NEEDS_EVIDENCE"
        e4_status = "NEEDS_EVIDENCE"
        e4_caveat = ""

    claims = {
        "format": "pipeann.aris.v100_early_pq_triggered.v1",
        "created_utc": utc_stamp(),
        "inputs": {
            "targeted_dir": rel(targeted_dir),
            "chain_dir": rel(chain_dir) if chain_dir else "",
            "smoke_dir": rel(smoke_dir) if smoke_dir else "",
            "space_artifacts_dir": rel(space_dir) if space_dir else "",
            "background_interference_dir": rel(background_interference_dir) if background_interference_dir else "",
        },
        "claims": [
            {
                "id": "E1_EARLY_PQ_NO_RETRAIN_FAST_FAIL_BASELINE",
                "status": "PASS" if baseline_metrics["rows"] and baseline_metrics["max_avg_latency_ms"] >= args.latency_ms else "NEEDS_EVIDENCE",
                "metrics": baseline_metrics,
                "evidence": ["targeted_latency_profile.jsonl"],
            },
            {
                "id": "E2_TRIGGERED_RETRAIN_TARGETED_SENTINELS",
                "status": "PASS" if target_pass else "NEEDS_EVIDENCE",
                "metrics": targeted_metrics,
                "evidence": ["optimized_dynamic_update_results.jsonl", "pq_drift_strategy_compare.jsonl"],
                "caveat": "Targeted post-processing covers high-risk sentinel buckets/cycles from the stopped baseline; it is not the final five-cycle chain by itself.",
            },
            {
                "id": "E3_TRIGGERED_RETRAIN_1M_5CYCLE_CHAIN",
                "status": chain_status,
                "metrics": {
                    **chain_metrics,
                    "completed_cycles": completed_cycles,
                    "expected_cycles": args.expected_chain_cycles,
                    "expected_rows": args.expected_chain_rows,
                    "expected_cycle_set": sorted(expected_cycle_set),
                    "missing_cycles": sorted(expected_cycle_set - observed_cycle_set),
                },
                "evidence": ["optimized_dynamic_update_results.jsonl", "early_pq_delete_insert_maintenance.jsonl", "summary.json"],
            },
            {
                "id": "E4_MAINTENANCE_TIMING_AND_LOW_CORE_INTERFERENCE",
                "status": e4_status,
                "metrics": {
                    "maintenance": maintenance_metrics,
                    "maintenance_timing_status": maintenance_timing_status,
                    "foreground_interference_status": foreground_interference_status,
                    "early_pq_background_interference": background_interference_metrics,
                    "interference_rows": len(interference_rows),
                },
                "evidence": ["pq_retrain_interference_profile.jsonl", "early_pq_delete_insert_maintenance.jsonl"],
                **({"caveat": e4_caveat} if e4_caveat else {}),
            },
            {
                "id": "E5_SPACE_AND_LABEL_LAYOUT_REUSED",
                "status": "PASS" if copied_audits else "NEEDS_EVIDENCE",
                "metrics": {"copied_audits": copied_audits},
                "evidence": ["index_space_audit.md", "label_sidecar_layout_audit.md"],
            },
        ],
    }
    write_json(out / "optimized_claim_registry.json", claims)
    write_json(out / "summary.json", {
        "created_utc": utc_stamp(),
        "phase": "early_pq_triggered_artifact_synthesis",
        "baseline_summary": targeted_summary.get("baseline_summary", baseline_metrics),
        "targeted_summary": targeted_summary.get("strategy_summary", targeted_metrics),
        "chain_summary": chain_summary,
        "smoke_summary": smoke_summary,
        "baseline_metrics": baseline_metrics,
        "targeted_metrics": targeted_metrics,
        "chain_metrics": chain_metrics,
        "maintenance_metrics": maintenance_metrics,
        "background_interference_metrics": background_interference_metrics,
        "figures": figures,
        "claim_statuses": {claim["id"]: claim["status"] for claim in claims["claims"]},
    })

    ppt_rows = [
        {"metric": "baseline_no_retrain_rows", "value": baseline_metrics["rows"], "note": "fast-fail evidence rows"},
        {"metric": "baseline_no_retrain_max_avg_latency_ms", "value": f'{baseline_metrics["max_avg_latency_ms"]:.6f}', "note": f"limit <{args.latency_ms}"},
        {"metric": "baseline_no_retrain_max_p95_latency_ms", "value": f'{baseline_metrics["max_p95_latency_ms"]:.6f}', "note": f"limit <{args.latency_ms}"},
        {"metric": "baseline_no_retrain_p95_pass_rows", "value": baseline_metrics["p95_lt_latency_ms"], "note": "p95 latency rows under limit"},
        {"metric": "triggered_targeted_rows", "value": targeted_metrics["rows"], "note": "sentinel rows"},
        {"metric": "triggered_targeted_min_recall", "value": f'{targeted_metrics["min_recall"]:.6f}', "note": f"target >={args.recall_floor}"},
        {"metric": "triggered_targeted_max_avg_latency_ms", "value": f'{targeted_metrics["max_avg_latency_ms"]:.6f}', "note": f"target <{args.latency_ms}"},
        {"metric": "triggered_targeted_max_p95_latency_ms", "value": f'{targeted_metrics["max_p95_latency_ms"]:.6f}', "note": f"target <{args.latency_ms}"},
        {"metric": "triggered_targeted_p95_pass_rows", "value": targeted_metrics["p95_lt_latency_ms"], "note": "p95 latency rows under limit"},
        {"metric": "chain_completed_cycles", "value": completed_cycles, "note": f"expected {args.expected_chain_cycles}"},
        {"metric": "chain_rows", "value": chain_metrics["rows"], "note": f"selected rows generated so far; expected {args.expected_chain_rows}"},
        {"metric": "chain_max_avg_latency_ms", "value": f'{chain_metrics["max_avg_latency_ms"]:.6f}', "note": f"target <{args.latency_ms}"},
        {"metric": "chain_max_p95_latency_ms", "value": f'{chain_metrics["max_p95_latency_ms"]:.6f}', "note": f"target <{args.latency_ms}"},
        {"metric": "chain_p95_pass_rows", "value": chain_metrics["p95_lt_latency_ms"], "note": "p95 latency rows under limit"},
        {"metric": "maintenance_max_build_wall_s", "value": f'{maintenance_metrics["max_maintenance_build_wall_s"]:.6f}', "note": "triggered maintenance build"},
        {"metric": "maintenance_max_pq_train_wall_s", "value": f'{maintenance_metrics["max_pq_train_wall_s"]:.6f}', "note": "PQ train component"},
        {"metric": "maintenance_max_pq_recode_wall_s", "value": f'{maintenance_metrics["max_pq_recode_wall_s"]:.6f}', "note": "PQ recode component"},
        {"metric": "background_interference_status", "value": background_interference_metrics["status"], "note": "early-PQ foreground while low-core background build runs"},
        {"metric": "background_interference_rows", "value": background_interference_metrics.get("rows", 0), "note": "overlapping foreground rows"},
        {"metric": "background_interference_max_avg_latency_ms", "value": f'{fnum(background_interference_metrics, "max_avg_latency_ms"):.6f}', "note": f"target <{args.latency_ms}"},
        {"metric": "background_interference_max_p95_latency_ms", "value": f'{fnum(background_interference_metrics, "max_p95_latency_ms"):.6f}', "note": f"target <{args.latency_ms}"},
        {"metric": "background_interference_cpu_cap", "value": background_interference_metrics.get("background_cpu_cap", ""), "note": "background maintenance CPU cap"},
    ]
    write_csv(out / "ppt_ready_metrics.csv", ppt_rows)

    overall = overall_status([claim["status"] for claim in claims["claims"]])
    (out / "ppt_ready_conclusion_summary.md").write_text(
        "# Early-PQ Triggered Maintenance PPT Summary\n\n"
        f"- No-retrain early-PQ baseline fast-fails: `{baseline_metrics['rows']}` rows, worst avg `{baseline_metrics['max_avg_latency_ms']:.3f} ms`.\n"
        f"- Triggered retrain targeted sentinels: `{targeted_metrics['rows']}` rows, min recall `{targeted_metrics['min_recall']:.2f}`, worst avg `{targeted_metrics['max_avg_latency_ms']:.3f} ms`, worst p95 `{targeted_metrics['max_p95_latency_ms']:.3f} ms`.\n"
        f"- Chain progress: `{completed_cycles}/{args.expected_chain_cycles}` cycles, rows `{chain_metrics['rows']}/{args.expected_chain_rows}`, status `{chain_status}`.\n"
        f"- Maintenance max: build `{maintenance_metrics['max_maintenance_build_wall_s']:.3f}s`, PQ train `{maintenance_metrics['max_pq_train_wall_s']:.3f}s`, PQ recode `{maintenance_metrics['max_pq_recode_wall_s']:.3f}s`.\n"
        f"- Background interference: `{background_interference_metrics.get('foreground_overlap_status', background_interference_metrics['status'])}` for early-PQ foreground overlap/timing, rows `{background_interference_metrics.get('rows', 0)}`, worst avg `{fnum(background_interference_metrics, 'max_avg_latency_ms'):.3f} ms`, worst p95 `{fnum(background_interference_metrics, 'max_p95_latency_ms'):.3f} ms`; layout/4KB status is scoped separately by the reused v100 layout audit.\n"
        f"- Figures: `{figures}`\n",
        encoding="utf-8",
    )
    (out / "aris_final_review.md").write_text(
        "# Early-PQ Triggered Maintenance ARIS Review\n\n"
        f"Overall status: `{overall}`.\n\n"
        "## Evidence Checks\n"
        f"- Baseline no-retrain fast-fail: `{claims['claims'][0]['status']}` with {baseline_metrics}.\n"
        f"- Targeted triggered sentinels: `{claims['claims'][1]['status']}` with {targeted_metrics}.\n"
        f"- 1M 5-cycle chain: `{claims['claims'][2]['status']}` with {chain_metrics}; completed cycles `{completed_cycles}`.\n"
        f"- Maintenance timing/interference: `{claims['claims'][3]['status']}` with {maintenance_metrics}, background {background_interference_metrics}, and `{len(interference_rows)}` interference rows.\n"
        f"- Space/layout reuse: `{claims['claims'][4]['status']}` with copied audits `{copied_audits}`.\n\n"
        "## Guardrails\n"
        "- Use this as early-PQ/PQ-maintenance evidence. Keep the v100 Supersector32K space/4KB/label-sidecar audit as a separate reused claim unless rerun directly on the early-PQ chain outputs.\n"
        "- Selected route/L retuning is the accepted recall/latency口径; do not claim fixed-parameter graph quality from these rows.\n"
        f"- Partial chain outputs are marked `IN_PROGRESS`; final PASS requires at least {args.expected_chain_cycles} completed 60% delete/insert cycles and {args.expected_chain_rows} selected rows with no recall/latency failures.\n",
        encoding="utf-8",
    )
    validate_output_size(out)
    print(rel(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
