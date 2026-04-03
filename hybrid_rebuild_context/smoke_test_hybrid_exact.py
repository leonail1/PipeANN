#!/usr/bin/env python3
"""Run two smoke searches against the rebuilt exact assets."""

from __future__ import annotations

import argparse

from exact_hybrid_common import (
    DEFAULT_COARSE_NQ,
    cache_gt_path,
    cache_query_bin_path,
    cache_query_labels_path,
    canonical_selectivities,
    ensure_query_subset_cache,
    format_sel,
    parse_search_output,
    pq16_exact_prefix,
    pq32_exact_prefix,
    run_search_binary,
)


def run_one(name: str, index_prefix, sel: float, threshold: float, coarse_nq: int, l_value: int) -> None:
    query_bin = cache_query_bin_path(coarse_nq)
    gt_bin = cache_gt_path(sel, coarse_nq)
    qlabel = cache_query_labels_path(sel, coarse_nq)
    rows, metrics, output, code = run_search_binary(index_prefix, query_bin, gt_bin, qlabel, threshold, [l_value])
    if not rows:
        raise RuntimeError(f"{name} failed\n{output}")
    parsed_rows, _ = parse_search_output(output)
    if not parsed_rows:
        raise RuntimeError(f"{name} did not emit parseable output\n{output}")
    suffix = f" rc={code}" if code != 0 else ""
    print(f"[ok] {name} sel={format_sel(sel)} L={l_value} recall={parsed_rows[0]['recall']:.2f}{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selectivities", default="0.001,1.000")
    parser.add_argument("--coarse-nq", type=int, default=DEFAULT_COARSE_NQ)
    parser.add_argument("--prefilter-l", type=int, default=100)
    parser.add_argument("--graph-l", type=int, default=100)
    parser.add_argument("--force-rebuild-cache", action="store_true")
    args = parser.parse_args()

    selectivities = canonical_selectivities(args.selectivities)
    ensure_query_subset_cache(selectivities, coarse_nq=args.coarse_nq, rss_nq=1, force=args.force_rebuild_cache)

    prefilter_sel = min(selectivities)
    graph_sel = max(selectivities)
    run_one("prefilter", pq16_exact_prefix(), prefilter_sel, 1.0, args.coarse_nq, args.prefilter_l)
    run_one("graph-only", pq16_exact_prefix(), graph_sel, 0.0, args.coarse_nq, args.graph_l)


if __name__ == "__main__":
    main()
