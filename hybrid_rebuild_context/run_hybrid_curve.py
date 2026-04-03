#!/usr/bin/env python3
"""Canonical runner for the latest hybrid combined curve."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from exact_hybrid_common import (
    DEFAULT_COARSE_NQ,
    DEFAULT_RSS_NQ,
    artifacts_results_v2_dir,
    cache_gt_path,
    cache_query_bin_path,
    cache_query_labels_path,
    canonical_selectivities,
    ensure_query_subset_cache,
    format_sel,
    pq16_exact_prefix,
    resolve_path,
    run_search_binary,
    run_single_query_peak_rss,
)

RECALL_TARGET = 98.0
COARSE_LS = [10, 20, 50, 100, 200, 500, 1000, 2000]


def pick_best_effort(rows: list[dict[str, float]]) -> dict[str, float] | None:
    if not rows:
        return None
    return max(rows, key=lambda row: (row["recall"], -row["latency_us"]))


def bisect_min_l(index_prefix: Path, query_bin: Path, gt_bin: Path, qlabel: Path, threshold: float,
                 recall_target: float) -> dict[str, float] | None:
    coarse_rows, _, _, _ = run_search_binary(index_prefix, query_bin, gt_bin, qlabel, threshold, COARSE_LS, threads=1)
    if not coarse_rows:
        return None

    passing = [row for row in coarse_rows if row["recall"] >= recall_target]
    if not passing:
        return pick_best_effort(coarse_rows)

    failing = [row for row in coarse_rows if row["recall"] < recall_target]
    lo = max((int(row["L"]) for row in failing), default=10)
    hi = min(int(row["L"]) for row in passing)
    best = min(passing, key=lambda row: int(row["L"]))

    while hi - lo > 5:
        mid = (hi + lo) // 2
        rows, _, _, _ = run_search_binary(index_prefix, query_bin, gt_bin, qlabel, threshold, [mid], threads=1)
        if not rows:
            lo = mid
            continue
        current = rows[0]
        if current["recall"] >= recall_target:
            hi = mid
            best = current
        else:
            lo = mid

    rows, _, _, _ = run_search_binary(index_prefix, query_bin, gt_bin, qlabel, threshold, [hi], threads=1)
    if rows:
        best = rows[0]
    return best


def measure_peak_rss(index_prefix: Path, query_bin: Path, qlabel: Path, threshold: float,
                     l_value: int) -> dict[str, float | str | None]:
    result = run_single_query_peak_rss(
        index_prefix=index_prefix,
        query_bin=query_bin,
        query_label_file=qlabel,
        prefilter_threshold=threshold,
        l_value=l_value,
        threads=1,
        timeout=180,
    )
    if result["returncode"] != 0:
        raise RuntimeError(f"single-query RSS measurement failed:\n{result['output']}")
    if result["peak_rss_mb"] is None:
        raise RuntimeError(f"missing /usr/bin/time peak RSS output:\n{result['output']}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selectivities", default=None,
                        help="Comma-separated selectivities. Defaults to the canonical 9-point exact set.")
    parser.add_argument("--coarse-nq", type=int, default=DEFAULT_COARSE_NQ)
    parser.add_argument("--rss-nq", type=int, default=DEFAULT_RSS_NQ)
    parser.add_argument("--outdir", default=str(artifacts_results_v2_dir()))
    parser.add_argument("--recall-target", type=float, default=RECALL_TARGET)
    parser.add_argument("--force-rebuild-cache", action="store_true")
    args = parser.parse_args()

    selectivities = canonical_selectivities(args.selectivities)
    ensure_query_subset_cache(
        selectivities,
        coarse_nq=args.coarse_nq,
        rss_nq=args.rss_nq,
        force=args.force_rebuild_cache,
    )

    outdir = resolve_path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "hybrid_results.csv"

    coarse_query = cache_query_bin_path(args.coarse_nq)
    rss_query = cache_query_bin_path(args.rss_nq)
    pf_index = pq16_exact_prefix()
    gr_index = pq16_exact_prefix()

    rows = []
    for sel in selectivities:
        sel_str = format_sel(sel)
        gt_bin = cache_gt_path(sel, args.coarse_nq)
        coarse_qlabel = cache_query_labels_path(sel, args.coarse_nq)
        rss_qlabel = cache_query_labels_path(sel, args.rss_nq)

        if not gt_bin.exists():
            print(f"SKIP sel={sel_str}: missing coarse GT {gt_bin}")
            continue

        print(f"[sel={sel_str}] prefilter search")
        pf = bisect_min_l(pf_index, coarse_query, gt_bin, coarse_qlabel, threshold=1.0, recall_target=args.recall_target)
        pf_rss = measure_peak_rss(pf_index, rss_query, rss_qlabel, 1.0, int(pf["L"])) if pf else None
        if pf_rss and pf_rss.get("warning"):
            print(f"[sel={sel_str}] warning: {pf_rss['warning']}")

        print(f"[sel={sel_str}] graph-only search")
        gr = bisect_min_l(gr_index, coarse_query, gt_bin, coarse_qlabel, threshold=0.0, recall_target=args.recall_target)
        gr_rss = measure_peak_rss(gr_index, rss_query, rss_qlabel, 0.0, int(gr["L"])) if gr else None
        if gr_rss and gr_rss.get("warning"):
            print(f"[sel={sel_str}] warning: {gr_rss['warning']}")

        rows.append({
            "selectivity": format_sel(sel),
            "pf_min_L": int(pf["L"]) if pf else "",
            "pf_recall": pf["recall"] if pf else "",
            "pf_latency_us": pf["latency_us"] if pf else "",
            "pf_rss_mb": pf_rss["peak_rss_mb"] if pf_rss is not None else "",
            "pf_rss_delta_mb": pf_rss["rss_delta_mb"] if pf_rss is not None else "",
            "pf_process_peak_rss_mb": pf_rss["process_peak_rss_mb"] if pf_rss is not None else "",
            "gr_min_L": int(gr["L"]) if gr else "",
            "gr_recall": gr["recall"] if gr else "",
            "gr_latency_us": gr["latency_us"] if gr else "",
            "gr_rss_mb": gr_rss["peak_rss_mb"] if gr_rss is not None else "",
            "gr_rss_delta_mb": gr_rss["rss_delta_mb"] if gr_rss is not None else "",
            "gr_process_peak_rss_mb": gr_rss["process_peak_rss_mb"] if gr_rss is not None else "",
        })

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "selectivity",
            "pf_min_L",
            "pf_recall",
            "pf_latency_us",
            "pf_rss_mb",
            "pf_rss_delta_mb",
            "pf_process_peak_rss_mb",
            "gr_min_L",
            "gr_recall",
            "gr_latency_us",
            "gr_rss_mb",
            "gr_rss_delta_mb",
            "gr_process_peak_rss_mb",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved -> {csv_path}")


if __name__ == "__main__":
    main()
