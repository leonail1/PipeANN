#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


SELECTIVITY_SPECS = (
    ("u1e-03", 0.001),
    ("u3e-03", 0.003),
    ("u1e-02", 0.01),
    ("u5e-02", 0.05),
    ("u1e-01", 0.1),
    ("u25", 0.25),
    ("u30", 0.3),
    ("u50", 0.5),
    ("u75", 0.75),
    ("u100", 1.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the staged SIFT1M prefilter PQ compare experiment for a single visible-corpus stage."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--reuse-workloads", action="store_true")
    parser.add_argument("--plot-l", type=int, default=100)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--beamwidth", type=int, default=4)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--search-l", type=int, default=100)
    parser.add_argument("--target-recall", type=int, default=98)
    parser.add_argument("--selector-type", default="intersect")
    parser.add_argument("--metric", default="l2")
    parser.add_argument("--nbr-type", default="pq")
    parser.add_argument("--recalibration-sample-limit", type=int, default=200)
    parser.add_argument("--recalibration-timeout-s", type=int, default=900)
    parser.add_argument("--search-timeout-s", type=int, default=21600)
    parser.add_argument("--full-compare-queries-per-bucket", type=int, default=1000)
    parser.add_argument("--transition-probe-queries-per-bucket", type=int, default=5000)
    return parser.parse_args()


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(line_buffering=True, write_through=True)


def build_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def log(message: str) -> None:
    print(message, flush=True)


def load_manifest(manifest_path: Path) -> dict:
    with manifest_path.open("r", encoding="utf-8") as reader:
        return json.load(reader)


def get_stage(manifest: dict, stage_name: str) -> dict:
    for stage in manifest["stages"]:
        if stage["name"] == stage_name:
            return stage
    raise KeyError(f"stage not found in manifest: {stage_name}")


def run_checked(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True, env=build_subprocess_env())


def ensure_stage_workloads(repo_root: Path, manifest: dict, stage: dict, args: argparse.Namespace) -> None:
    workload_dir = Path(stage["workload_dir"])
    summary_json = workload_dir / "uniform_exact_selectivity_summary.json"
    if args.reuse_workloads and summary_json.exists():
        compare_query_count, probe_query_count = read_workload_query_counts(summary_json)
        if (
            compare_query_count == args.full_compare_queries_per_bucket
            and probe_query_count == args.transition_probe_queries_per_bucket
        ):
            return
        log(
            f"[compare:{stage['name']}] regenerating workloads because query counts changed "
            f"(compare={compare_query_count}, probe={probe_query_count}) -> "
            f"(compare={args.full_compare_queries_per_bucket}, probe={args.transition_probe_queries_per_bucket})"
        )

    if workload_dir.exists():
        shutil.rmtree(workload_dir)
    workload_dir.mkdir(parents=True, exist_ok=True)

    command = [
        str(repo_root / ".venv" / "bin" / "python"),
        "scripts/pipeann_hybrid_experiment.py",
        "generate-uniform-exact-selectivity-workloads",
        "--base-bin",
        stage["base_bin"],
        "--query-bin",
        manifest["query_bin"],
        "--index-type",
        "float",
        "--selector-type",
        args.selector_type,
        "--out-dir",
        str(workload_dir),
        "--queries-per-bucket",
        str(args.full_compare_queries_per_bucket),
        "--probe-queries-per-bucket",
        str(args.transition_probe_queries_per_bucket),
    ]
    for bucket_name, selectivity in SELECTIVITY_SPECS:
        command.extend(["--selectivity-spec", f"{bucket_name}:{selectivity}"])
    run_checked(command, repo_root)


def read_bucket_names(summary_json: Path) -> list[str]:
    with summary_json.open("r", encoding="utf-8") as reader:
        payload = json.load(reader)
    return [entry["bucket_name"] for entry in payload["workloads"]]


def read_max_selectivity(summary_json: Path) -> str:
    with summary_json.open("r", encoding="utf-8") as reader:
        payload = json.load(reader)
    return str(max(float(entry["selectivity"]) for entry in payload["workloads"]))


def read_workload_query_counts(summary_json: Path) -> tuple[int | None, int | None]:
    with summary_json.open("r", encoding="utf-8") as reader:
        payload = json.load(reader)
    compare_query_count = payload.get("queries_per_selectivity")
    if compare_query_count is None:
        compare_query_count = payload.get("queries_per_bucket")
    probe_query_count = payload.get("probe_queries_per_selectivity")
    return (
        None if compare_query_count is None else int(compare_query_count),
        None if probe_query_count is None else int(probe_query_count),
    )


def runtime_prefix(stage: dict, pq_bits: int) -> Path:
    return Path(stage["runtime_dir"]) / f"sift1m_uniform_prefilter_compare_{stage['short_name']}_pq{pq_bits}"


def stage_probe_query_bin(stage: dict) -> Path:
    return Path(stage["workload_dir"]) / "probe_queries.bin"


def bootstrap_auto_route(repo_root: Path, manifest: dict, stage: dict, pq_bits: int,
                         args: argparse.Namespace, run_dir: Path, index_prefix: Path) -> None:
    command = [
        str(repo_root / "build" / "tests" / "dynamic_prefilter_stage_driver"),
        "--source-prefix",
        str(index_prefix),
        "--dest-prefix",
    str(index_prefix),
        "--label-file",
        str(Path(stage["workload_dir"]) / "base.uniform_exact_selectivity.spmat"),
        "--query-bin",
        str(stage_probe_query_bin(stage)),
        "--workload-dir",
        stage["workload_dir"],
        "--insert-start",
        "0",
        "--insert-count",
        "0",
        "--insert-threads",
        "1",
        "--search-threads",
        str(args.threads),
        "--materialize-threads",
        "1",
        "--beamwidth",
        str(args.beamwidth),
        "--k",
        str(args.k),
        "--search-l",
        str(args.search_l),
        "--mem-l",
        "0",
        "--selector-type",
        args.selector_type,
        "--metric",
        args.metric,
        "--recalibration-sample-limit",
        str(args.recalibration_sample_limit),
        "--recalibration-timeout-s",
        str(args.recalibration_timeout_s),
    ]
    for bucket_name, selectivity in SELECTIVITY_SPECS:
        command.extend(["--bucket-spec", f"{bucket_name}:{selectivity}"])
    with (run_dir / "bootstrap_auto_route.log").open("w", encoding="utf-8", buffering=1) as writer:
        subprocess.run(
            command,
            cwd=repo_root,
            check=True,
            stdout=writer,
            stderr=subprocess.STDOUT,
            env=build_subprocess_env(),
        )


def run_variant(repo_root: Path, manifest: dict, stage: dict, pq_bits: int, args: argparse.Namespace) -> Path:
    workload_dir = Path(stage["workload_dir"])
    runtime_dir = Path(stage["runtime_dir"])
    results_dir = Path(stage["results_dir"])
    summary_json = workload_dir / "uniform_exact_selectivity_summary.json"
    if not summary_json.exists():
        raise FileNotFoundError(f"missing workload summary for stage {stage['name']}: {summary_json}")

    run_dir = results_dir / f"pq{pq_bits}"
    dest_prefix = runtime_prefix(stage, pq_bits)
    source_prefix = stage["source_prefixes"][str(pq_bits)]
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    query_bin = manifest["query_bin"]
    bucket_names = read_bucket_names(summary_json)
    max_selectivity = read_max_selectivity(summary_json)

    log(f"[compare:{stage['name']}] pq{pq_bits} prepare runtime index")
    prepare_cmd = [
        str(repo_root / ".venv" / "bin" / "python"),
        "scripts/pipeann_hybrid_experiment.py",
        "prepare-index-prefix-for-labels",
        "--source-prefix",
        source_prefix,
        "--dest-prefix",
        str(dest_prefix),
        "--label-file",
        str(workload_dir / "base.uniform_exact_selectivity.spmat"),
        "--summary-json",
        str(run_dir / "index_runtime.json"),
    ]
    with (run_dir / "prepare_index.log").open("w", encoding="utf-8", buffering=1) as writer:
        subprocess.run(
            prepare_cmd,
            cwd=repo_root,
            check=True,
            stdout=writer,
            stderr=subprocess.STDOUT,
            env=build_subprocess_env(),
        )

    log(f"[compare:{stage['name']}] pq{pq_bits} bootstrap auto route")
    bootstrap_auto_route(repo_root, manifest, stage, pq_bits, args, run_dir, dest_prefix)

    log(f"[compare:{stage['name']}] pq{pq_bits} build manifest")
    manifest_cmd = [
        str(repo_root / ".venv" / "bin" / "python"),
        "scripts/pipeann_hybrid_experiment.py",
        "build-manifest-from-summary",
        "--summary-json",
        str(summary_json),
        "--index-prefix",
        str(dest_prefix),
        "--index-type",
        "float",
        "--selector-type",
        args.selector_type,
        "--manifest",
        str(run_dir / "manifest.json"),
    ]
    with (run_dir / "build_manifest.log").open("w", encoding="utf-8", buffering=1) as writer:
        subprocess.run(
            manifest_cmd,
            cwd=repo_root,
            check=True,
            stdout=writer,
            stderr=subprocess.STDOUT,
            env=build_subprocess_env(),
        )

    log(f"[compare:{stage['name']}] pq{pq_bits} calibrate rerank for {len(bucket_names)} buckets")
    rerank_cmd = [
        str(repo_root / ".venv" / "bin" / "python"),
        "scripts/pipeann_hybrid_experiment.py",
        "calibrate-rerank",
        "--summary-json",
        str(summary_json),
        "--index-prefix",
        str(dest_prefix),
        "--out-dir",
        str(run_dir / "calibration"),
        "--threads",
        str(args.threads),
        "--beamwidth",
        str(args.beamwidth),
        "--k",
        str(args.k),
        "--similarity",
        args.metric,
        "--nbr-type",
        args.nbr_type,
        "--search-l",
        str(args.search_l),
        "--target-recall",
        str(args.target_recall),
        "--timeout",
        str(args.search_timeout_s),
        "--max-selectivity",
        max_selectivity,
    ]
    with (run_dir / "calibrate_rerank.log").open("w", encoding="utf-8", buffering=1) as writer:
        subprocess.run(
            rerank_cmd,
            cwd=repo_root,
            check=True,
            stdout=writer,
            stderr=subprocess.STDOUT,
            env=build_subprocess_env(),
        )

    log(f"[compare:{stage['name']}] pq{pq_bits} run search sweep")
    run_cmd = [
        str(repo_root / ".venv" / "bin" / "python"),
        "scripts/pipeann_hybrid_experiment.py",
        "run",
        "--manifest",
        str(run_dir / "manifest.json"),
        "--out-dir",
        str(run_dir / "run"),
        "--dataset-name",
        f"sift1m_{stage['short_name']}_prefilter_pq{pq_bits}_compare",
        "--threads",
        str(args.threads),
        "--beamwidth",
        str(args.beamwidth),
        "--k",
        str(args.k),
        "--similarity",
        args.metric,
        "--nbr-type",
        args.nbr_type,
        "--mem-l",
        "0",
        "--routes",
        "auto",
        "--l-values",
        str(args.search_l),
        "--timeout",
        str(args.search_timeout_s),
        "--prefilter-rerank-json",
        str(run_dir / "calibration" / "prefilter_rerank_calibration.json"),
    ]
    with (run_dir / "run.log").open("w", encoding="utf-8", buffering=1) as writer:
        subprocess.run(
            run_cmd,
            cwd=repo_root,
            check=True,
            stdout=writer,
            stderr=subprocess.STDOUT,
            env=build_subprocess_env(),
        )
    log(f"[compare:{stage['name']}] pq{pq_bits} completed")
    return run_dir / "run" / "results.jsonl"


def plot_stage(repo_root: Path, stage: dict, result_paths: list[tuple[str, Path]], args: argparse.Namespace) -> None:
    command = [
        str(repo_root / ".venv" / "bin" / "python"),
        "scripts/plot_prefilter_pq_compare.py",
        "--route",
        "auto",
    ]
    for series_name, result_path in result_paths:
        command.extend(["--series", series_name, str(result_path)])
    command.extend([
        "--output",
        stage["compare_png"],
        "--plot-l",
        str(args.plot_l),
        "--title",
        f"PipeANN sift1m hybrid auto-route comparison ({stage['short_name']}, 1 thread, L={args.plot_l})",
    ])
    plot_log = Path(stage["results_dir"]).parent / "plot.log"
    with plot_log.open("w", encoding="utf-8", buffering=1) as writer:
        subprocess.run(
            command,
            cwd=repo_root,
            check=True,
            stdout=writer,
            stderr=subprocess.STDOUT,
            env=build_subprocess_env(),
        )


def main() -> int:
    configure_stdio()
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    manifest = load_manifest(args.manifest)
    stage = get_stage(manifest, args.stage)

    log(f"[compare] start {stage['name']}")
    ensure_stage_workloads(repo_root, manifest, stage, args)
    if args.prepare_only:
        log(f"[ok] prepared workloads for {stage['name']}")
        return 0

    result_paths: list[tuple[str, Path]] = []
    for pq_bits in manifest["pq_bits"]:
        result_paths.append((f"PQ{pq_bits}", run_variant(repo_root, manifest, stage, pq_bits, args)))
    log(f"[compare] plotting {stage['name']}")
    plot_stage(repo_root, stage, result_paths, args)
    log(f"[ok] finished staged compare for {stage['name']}")
    log(f"[ok] figure: {stage['compare_png']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())