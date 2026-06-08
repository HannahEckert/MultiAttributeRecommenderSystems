"""
Run post_processing.py for multiple seeds in parallel using CPU cores.
Each seed gets its own output subfolder. Iterates over model seeds 42, 43, 44
and builds the top_k path automatically from --model and --dataset.

Usage:
    python run_post_processing_parallel.py \
        --model BPR \
        --dataset lfm \
        --seeds 50 \
        --workers 10 \
        --l 0.0 0.25 0.5 0.75 1.0

Output is written to experiments/DATASET/outputMODELSEED/post_processing/...
"""

import argparse
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def run_single_seed(args):
    seed, model_seed, top_k_path, artist_metadata_path, listening_events_path, dataset, model, method, version, l, target_distribution, reranking_type = args
    output_base = Path(f"experiments/{dataset}/output{model}{model_seed}/post_processing")
    if method == "marras":
        output_path = str(output_base / target_distribution / "marras" / f"l{l}" / f"seed{seed}")
    elif method == "mitigation_continent":
        output_path = str(output_base / target_distribution / f"mitigation_continent_{reranking_type}" / f"l{l}" / f"seed{seed}")
    elif method == "nails":
        output_path = str(output_base / target_distribution / "nails" / f"l{l}" / f"seed{seed}")
    else:
        output_path = str(output_base / target_distribution / version / f"l{l}" / f"seed{seed}")
    cmd = [
        sys.executable, "post_processing.py",
        "--top-k-path", top_k_path,
        "--artist-metadata-path", artist_metadata_path,
        "--listening-events-path", listening_events_path,
        "--output-path", output_path,
        "--method", method,
        "--version", version,
        "--l", str(l),
        "--target-distribution", target_distribution,
        "--seed", str(seed),
        "--reranking-type", reranking_type,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return seed, result.returncode, result.stdout, result.stderr


def main():
    parser = argparse.ArgumentParser(description="Run post_processing.py for multiple seeds in parallel.")
    parser.add_argument("--model", type=str, default="BPR", help="Model name (e.g. BPR, LightGCN, NeuMF)")
    parser.add_argument("--dataset", type=str, default="lfm", help="Dataset name (e.g. lfm, lfm_small)")
    parser.add_argument("--artist_metadata_path", type=str, default="dataset/artists_metadata.csv")
    parser.add_argument("--listening_events_path", type=str, default="dataset/listening_events_filtered.csv")
    parser.add_argument("--method", type=str, nargs="+", default=["trade_off"], choices=["trade_off", "marras", "mitigation_continent", "nails"], help="One or more post-processing methods to run (e.g. --method trade_off marras nails)")
    parser.add_argument("--version", type=str, nargs="+", default=["sum"], choices=["sum", "product"], help="One or more versions to run (e.g. --version sum product)")
    parser.add_argument("--l", type=float, nargs="+", default=[0.1,0.01,0.005,0.001,0.0005,0.0001,0.00005], help="One or more trade-off values (e.g. --l 0.0 0.25 0.5)")
    parser.add_argument("--target_distribution", type=str, nargs="+", default=["interactions"], help="One or more target distributions (e.g. --target_distribution interactions population)")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds to run (0 to seeds-1)")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel workers (CPU cores to use)")
    parser.add_argument("--reranking_type", type=str, default="exposure", choices=["exposure", "visibility"], help="Reranking type for mitigation_continent")

    args = parser.parse_args()

    no_version_methods = {"marras", "mitigation_continent", "nails"}

    model_seeds = [42, 43, 44]
    seeds = list(range(args.seeds))
    task_args = [
        (
            seed,
            model_seed,
            f"experiments/{args.dataset}/output{args.model}{model_seed}/top_k_valid.tsv",
            args.artist_metadata_path,
            args.listening_events_path,
            args.dataset,
            args.model,
            method,
            version,
            l,
            target_distribution,
            args.reranking_type,
        )
        for model_seed in model_seeds
        for seed in seeds
        for method in args.method
        for l in args.l
        for version in (["-"] if method in no_version_methods else args.version)
        for target_distribution in args.target_distribution
    ]

    print(f"Running {len(model_seeds)} model seeds × {len(seeds)} seeds × {len(args.method)} methods × {len(args.l)} l values × {len(args.version)} versions × {len(args.target_distribution)} distributions = {len(task_args)} jobs with {args.workers} parallel workers...")
    failed = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_single_seed, a): (a[0], a[1], a[7], a[8], a[9], a[10]) for a in task_args}
        completed = 0
        for future in as_completed(futures):
            try:
                seed, returncode, stdout, stderr = future.result()
                completed += 1
                status = "✓" if returncode == 0 else "✗"
                _, model_seed_val, method_val, version_val, l_val, td_val = futures[future]
                label = f"method={method_val} l={l_val} target={td_val}" if method_val == "marras" else f"version={version_val} l={l_val} target={td_val}"
                print(f"[{completed}/{len(task_args)}] {status} modelseed={model_seed_val} seed={seed} {label}")
                if returncode != 0:
                    failed.append((seed, model_seed_val, method_val, version_val, l_val, td_val))
                    print(f"  STDERR: {stderr.strip()[-500:]}")
            except Exception as e:
                completed += 1
                failed.append(futures[future])
                print(f"[{completed}/{len(task_args)}] ✗ — exception: {e}")

    print(f"\nDone. {len(task_args) - len(failed)}/{len(task_args)} succeeded.")
    if failed:
        print(f"Failed (seed, l) pairs: {failed}")


if __name__ == "__main__":
    main()
