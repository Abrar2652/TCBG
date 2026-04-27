"""
experiments/eval_benchmark.py
Run 10-seed × 5-fold benchmark evaluation matching T3Former's protocol.

For each dataset, runs seeds 0-9. Each seed uses a different StratifiedKFold
random state, giving 10 × 5 = 50 evaluations per dataset.

Reports:
  mean ± std  (mean over 10 seed-means, std over 10 seed-means)

This matches the standard TUDataset benchmark protocol used by T3Former.

Usage:
  python experiments/eval_benchmark.py --datasets dblp tumblr highschool mit
  python experiments/eval_benchmark.py --datasets dblp --seeds 0 1 2 3 4 5 6 7 8 9
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPT = str(Path(__file__).parent / "train.py")


def run_seed(dataset: str, seed: int, result_dir: str, extra_args: list[str]) -> float | None:
    """Run train.py for one dataset/seed, return mean_acc (from saved JSON or stdout)."""
    result_file = os.path.join(result_dir, f"{dataset}_nc2_seed{seed}.json")
    # Use cached result if already exists
    if os.path.exists(result_file):
        with open(result_file) as f:
            r = json.load(f)
        print(f"  [cached] {dataset} seed={seed} -> {r['mean_acc']:.4f}")
        return float(r['mean_acc'])

    cmd = [
        sys.executable, SCRIPT,
        "--dataset", dataset,
        "--seed", str(seed),
        "--result_dir", result_dir,
        "--quiet",
    ] + extra_args

    print(f"  Running {dataset} seed={seed} ...", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ERROR seed={seed}:\n{result.stderr[-500:]}")
        return None

    # Parse from stdout as fallback
    for line in result.stdout.splitlines():
        if "Result:" in line and "acc =" in line:
            try:
                acc_str = line.split("acc =")[1].strip().split()[0]
                return float(acc_str)
            except Exception:
                pass

    # Try JSON
    if os.path.exists(result_file):
        with open(result_file) as f:
            r = json.load(f)
        return float(r['mean_acc'])

    print(f"  WARNING: could not parse result for {dataset} seed={seed}")
    print(result.stdout[-300:])
    return None


def aggregate(dataset: str, seeds: list[int], result_dir: str, extra_args: list[str]) -> dict:
    seed_means = []
    for seed in seeds:
        acc = run_seed(dataset, seed, result_dir, extra_args)
        if acc is not None:
            seed_means.append(acc)

    if not seed_means:
        return {'dataset': dataset, 'error': 'no results'}

    mean = float(np.mean(seed_means))
    std  = float(np.std(seed_means))
    print(f"\n{'='*40}")
    print(f"FINAL  {dataset:15s} | acc = {mean:.4f} +/- {std:.4f}  (n={len(seed_means)} seeds)")
    print(f"{'='*40}")
    return {
        'dataset': dataset,
        'seeds': seeds[:len(seed_means)],
        'seed_means': seed_means,
        'mean_acc': mean,
        'std_acc': std,
        'n_seeds': len(seed_means),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+',
                        default=['dblp', 'tumblr', 'highschool', 'mit', 'facebook'],
                        help='Datasets to evaluate')
    parser.add_argument('--seeds', nargs='+', type=int,
                        default=list(range(10)),
                        help='Seeds to run (default: 0-9)')
    parser.add_argument('--result_dir', type=str, default='./results')
    parser.add_argument('--summary_file', type=str, default='./results/benchmark_summary.json')
    # Pass-through args to train.py
    parser.add_argument('--cache_dir', type=str, default='./data/cache')
    parser.add_argument('--data_root', type=str, default='./data/raw')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    extra_args = [
        '--cache_dir', args.cache_dir,
        '--data_root', args.data_root,
        '--device', args.device,
    ]

    print(f"Benchmark: {len(args.datasets)} datasets × {len(args.seeds)} seeds")
    print(f"Seeds: {args.seeds}")
    print(f"Datasets: {args.datasets}\n")

    all_results = {}
    for ds in args.datasets:
        print(f"\n========== {ds.upper()} ==========")
        res = aggregate(ds, args.seeds, args.result_dir, extra_args)
        all_results[ds] = res

    # Summary table
    print("\n\n" + "="*60)
    print(f"{'Dataset':<15}  {'Mean Acc':>10}  {'Std':>8}  {'N seeds':>8}")
    print("-"*60)
    for ds, res in all_results.items():
        if 'error' not in res:
            print(f"{ds:<15}  {res['mean_acc']:>10.4f}  {res['std_acc']:>8.4f}  {res['n_seeds']:>8}")
        else:
            print(f"{ds:<15}  ERROR")

    # Save
    os.makedirs(os.path.dirname(args.summary_file), exist_ok=True)
    with open(args.summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to {args.summary_file}")


if __name__ == '__main__':
    main()
