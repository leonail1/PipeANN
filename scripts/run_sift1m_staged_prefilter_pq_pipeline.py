#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the staged SIFT1M dynamic prefilter PQ experiment pipeline end-to-end."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/sift1m_uniform_prefilter_pq_staged_compare/stage_manifest.json"),
    )
    parser.add_argument("--prepare-if-missing", action="store_true")
    parser.add_argument("--force-rebuild-initial", action="store_true")
    parser.add_argument("--plot-l", type=int, default=100)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--beamwidth", type=int, default=4)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--search-l", type=int, default=100)
    parser.add_argument("--target-recall", type=int, default=98)
    parser.add_argument("--selector-type", default="intersect")
    parser.add_argument("--metric", default="l2")
    parser.add_argument("--nbr-type", default="pq")
    parser.add_argument("--insert-threads", type=int, default=1)
    parser.add_argument("--materialize-threads", type=int, default=1)
    parser.add_argument("--recalibration-sample-limit", type=int, default=200)
    parser.add_argument("--recalibration-timeout-s", type=int, default=900)
    parser.add_argument("--search-timeout-s", type=int, default=21600)
    parser.add_argument("--full-compare-queries-per-bucket", type=int, default=1000)
    parser.add_argument("--transition-probe-queries-per-bucket", type=int, default=5000)
    parser.add_argument(
        "--start-stage",
        help="Resume from a specific stage name or short name, such as stage_500k or 500k.",
    )
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


def run_checked(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True, env=build_subprocess_env())


def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as reader:
        return json.load(reader)


def ensure_manifest(repo_root: Path, args: argparse.Namespace) -> Path:
    manifest_path = args.manifest.resolve()
    if manifest_path.exists():
        return manifest_path
    if not args.prepare_if_missing:
        raise FileNotFoundError(f"manifest does not exist: {manifest_path}")
    command = [
        str(repo_root / ".venv" / "bin" / "python"),
        "scripts/prepare_sift1m_staged_prefilter.py",
        "--out-dir",
        str(manifest_path.parent),
    ]
    run_checked(command, repo_root)
    return manifest_path


def build_initial_indexes(repo_root: Path, manifest: dict, force_rebuild: bool) -> None:
    build_cfg = manifest["build_config"]
    stage0 = manifest["stages"][0]
    base_bin = stage0["base_bin"]
    for pq_bits in manifest["pq_bits"]:
        prefix = Path(stage0["source_prefixes"][str(pq_bits)])
        disk_index = Path(str(prefix) + "_disk.index")
        if force_rebuild and prefix.parent.exists():
            for sibling in prefix.parent.glob(prefix.name + "*"):
                if sibling.is_file() or sibling.is_symlink():
                    sibling.unlink()
        if disk_index.exists() and not force_rebuild:
            continue
        prefix.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(repo_root / "build" / "tests" / "build_disk_index"),
            "float",
            base_bin,
            str(prefix),
            str(build_cfg["r"]),
            str(build_cfg["l"]),
            str(pq_bits),
            str(build_cfg["memory_gb"]),
            str(build_cfg["threads"]),
            build_cfg["metric"],
            build_cfg["nbr_type"],
        ]
        run_checked(command, repo_root)


def runtime_prefix(stage: dict, pq_bits: int) -> str:
    return str(Path(stage["runtime_dir"]) / f"sift1m_uniform_prefilter_compare_{stage['short_name']}_pq{pq_bits}")


def stage_probe_query_bin(stage: dict) -> str:
    return str(Path(stage["workload_dir"]) / "probe_queries.bin")


def find_stage_index(manifest: dict, requested_stage: str) -> int:
    normalized = requested_stage.strip()
    for index, stage in enumerate(manifest["stages"]):
        if normalized in {stage["name"], stage["short_name"]}:
            return index
    available = ", ".join(stage["name"] for stage in manifest["stages"])
    raise KeyError(f"unknown stage {requested_stage!r}; available stages: {available}")


def run_stage_compare(repo_root: Path, manifest_path: Path, stage_name: str, reuse_workloads: bool,
                      args: argparse.Namespace) -> None:
    command = [
        str(repo_root / ".venv" / "bin" / "python"),
        "scripts/run_sift1m_staged_prefilter_pq_compare.py",
        "--manifest",
        str(manifest_path),
        "--stage",
        stage_name,
        "--plot-l",
        str(args.plot_l),
        "--threads",
        str(args.threads),
        "--beamwidth",
        str(args.beamwidth),
        "--k",
        str(args.k),
        "--search-l",
        str(args.search_l),
        "--target-recall",
        str(args.target_recall),
        "--selector-type",
        args.selector_type,
        "--metric",
        args.metric,
        "--nbr-type",
        args.nbr_type,
        "--recalibration-sample-limit",
        str(args.recalibration_sample_limit),
        "--recalibration-timeout-s",
        str(args.recalibration_timeout_s),
        "--search-timeout-s",
        str(args.search_timeout_s),
        "--full-compare-queries-per-bucket",
        str(args.full_compare_queries_per_bucket),
        "--transition-probe-queries-per-bucket",
        str(args.transition_probe_queries_per_bucket),
    ]
    if reuse_workloads:
        command.append("--reuse-workloads")
    run_checked(command, repo_root)


def prepare_stage_workloads_only(repo_root: Path, manifest_path: Path, stage_name: str,
                                 args: argparse.Namespace) -> None:
    command = [
        str(repo_root / ".venv" / "bin" / "python"),
        "scripts/run_sift1m_staged_prefilter_pq_compare.py",
        "--manifest",
        str(manifest_path),
        "--stage",
        stage_name,
        "--prepare-only",
        "--plot-l",
        str(args.plot_l),
        "--threads",
        str(args.threads),
        "--beamwidth",
        str(args.beamwidth),
        "--k",
        str(args.k),
        "--search-l",
        str(args.search_l),
        "--target-recall",
        str(args.target_recall),
        "--selector-type",
        args.selector_type,
        "--metric",
        args.metric,
        "--nbr-type",
        args.nbr_type,
        "--recalibration-sample-limit",
        str(args.recalibration_sample_limit),
        "--recalibration-timeout-s",
        str(args.recalibration_timeout_s),
        "--search-timeout-s",
        str(args.search_timeout_s),
        "--full-compare-queries-per-bucket",
        str(args.full_compare_queries_per_bucket),
        "--transition-probe-queries-per-bucket",
        str(args.transition_probe_queries_per_bucket),
    ]
    run_checked(command, repo_root)


def bucket_specs() -> list[str]:
    return [
        "u1e-03:0.001",
        "u3e-03:0.003",
        "u1e-02:0.01",
        "u5e-02:0.05",
        "u1e-01:0.1",
        "u25:0.25",
        "u30:0.3",
        "u50:0.5",
        "u75:0.75",
        "u100:1.0",
    ]


def run_transition(repo_root: Path, previous_stage: dict, next_stage: dict, pq_bits: int,
                   args: argparse.Namespace) -> None:
    source_prefix = runtime_prefix(previous_stage, pq_bits)
    dest_prefix = next_stage["source_prefixes"][str(pq_bits)]
    probe_dir = Path(next_stage["probe_dir"]) / f"pq{pq_bits}"
    probe_dir.mkdir(parents=True, exist_ok=True)
    probe_jsonl = probe_dir / "during_insert_probe.jsonl"
    if probe_jsonl.exists():
        probe_jsonl.unlink()

    command = [
        str(repo_root / "build" / "tests" / "dynamic_prefilter_stage_driver"),
        "--source-prefix",
        source_prefix,
        "--dest-prefix",
        dest_prefix,
        "--data-bin",
        next_stage["base_bin"],
        "--label-file",
        str(Path(next_stage["workload_dir"]) / "base.uniform_exact_selectivity.spmat"),
        "--query-bin",
        stage_probe_query_bin(next_stage),
        "--workload-dir",
        next_stage["workload_dir"],
        "--probe-jsonl",
        str(probe_jsonl),
        "--insert-summary-json",
        str(probe_dir / "insert_batch_summary.json"),
        "--selector-type",
        args.selector_type,
        "--metric",
        args.metric,
        "--insert-start",
        str(previous_stage["npoints"]),
        "--insert-count",
        str(next_stage["npoints"] - previous_stage["npoints"]),
        "--insert-threads",
        str(args.insert_threads),
        "--search-threads",
        str(args.threads),
        "--materialize-threads",
        str(args.materialize_threads),
        "--beamwidth",
        str(args.beamwidth),
        "--k",
        str(args.k),
        "--search-l",
        str(args.search_l),
        "--mem-l",
        "0",
        "--recalibration-sample-limit",
        str(args.recalibration_sample_limit),
        "--recalibration-timeout-s",
        str(args.recalibration_timeout_s),
    ]
    for spec in bucket_specs():
        command.extend(["--bucket-spec", spec])
    run_checked(command, repo_root)


def main() -> int:
    configure_stdio()
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = ensure_manifest(repo_root, args)
    manifest = load_manifest(manifest_path)
    stages = manifest["stages"]
    start_index = 0 if not args.start_stage else find_stage_index(manifest, args.start_stage)

    if start_index == 0:
        log(f"[pipeline] building initial indexes from {manifest_path}")
        build_initial_indexes(repo_root, manifest, args.force_rebuild_initial)
        log(f"[pipeline] running compare for {stages[0]['name']}")
        run_stage_compare(repo_root, manifest_path, stages[0]["name"], False, args)
        next_transition_index = 1
    else:
        resume_stage = stages[start_index]
        log(
            f"[pipeline] resuming from {resume_stage['name']} using existing artifacts under {manifest_path.parent}"
        )
        log(f"[pipeline] rerunning compare for {resume_stage['name']}")
        run_stage_compare(repo_root, manifest_path, resume_stage["name"], True, args)
        next_transition_index = start_index + 1

    for stage_index in range(next_transition_index, len(stages)):
        previous_stage = stages[stage_index - 1]
        next_stage = stages[stage_index]
        log(f"[pipeline] preparing workloads for {next_stage['name']}")
        prepare_stage_workloads_only(repo_root, manifest_path, next_stage["name"], args)
        for pq_bits in manifest["pq_bits"]:
            log(
                f"[pipeline] transition {previous_stage['short_name']} -> {next_stage['short_name']} for pq{pq_bits}"
            )
            run_transition(repo_root, previous_stage, next_stage, pq_bits, args)
        log(f"[pipeline] running compare for {next_stage['name']}")
        run_stage_compare(repo_root, manifest_path, next_stage["name"], True, args)

    log("[pipeline] aggregating staged results")
    aggregate_cmd = [
        str(repo_root / ".venv" / "bin" / "python"),
        "scripts/aggregate_sift1m_staged_prefilter_results.py",
        "--manifest",
        str(manifest_path),
        "--json-out",
        str(manifest_path.parent / "summary.json"),
        "--csv-out",
        str(manifest_path.parent / "summary.csv"),
    ]
    run_checked(aggregate_cmd, repo_root)

    log(f"[ok] finished staged prefilter PQ pipeline using {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())