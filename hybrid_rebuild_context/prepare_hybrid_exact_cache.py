#!/usr/bin/env python3
"""Build persistent query/query-label/GT subsets for exact hybrid runners."""

from __future__ import annotations

import argparse

from exact_hybrid_common import (
    DEFAULT_COARSE_NQ,
    DEFAULT_RSS_NQ,
    cache_dir,
    cache_gt_path,
    cache_query_bin_path,
    cache_query_labels_path,
    canonical_selectivities,
    ensure_query_subset_cache,
    format_sel,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selectivities", default=None,
                        help="Comma-separated selectivities. Defaults to the canonical 9-point exact set.")
    parser.add_argument("--coarse-nq", type=int, default=DEFAULT_COARSE_NQ,
                        help="Number of queries used for coarse L search.")
    parser.add_argument("--rss-nq", type=int, default=DEFAULT_RSS_NQ,
                        help="Number of queries used for single-query RSS measurement.")
    parser.add_argument("--force", action="store_true", help="Rebuild cache files even if they already exist.")
    args = parser.parse_args()

    selectivities = canonical_selectivities(args.selectivities)
    ensure_query_subset_cache(selectivities, coarse_nq=args.coarse_nq, rss_nq=args.rss_nq, force=args.force)

    print(f"Cache directory: {cache_dir()}")
    for nq in sorted({args.coarse_nq, args.rss_nq}):
        print(f"  query[{nq}] -> {cache_query_bin_path(nq)}")
    for sel in selectivities:
        print(f"  sel={format_sel(sel)} qlabel[{args.coarse_nq}] -> {cache_query_labels_path(sel, args.coarse_nq)}")
        print(f"  sel={format_sel(sel)} qlabel[{args.rss_nq}] -> {cache_query_labels_path(sel, args.rss_nq)}")
        gt_file = cache_gt_path(sel, args.coarse_nq)
        if gt_file.exists():
            print(f"  sel={format_sel(sel)} gt[{args.coarse_nq}] -> {gt_file}")


if __name__ == "__main__":
    main()
