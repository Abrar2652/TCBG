"""
experiments/run_pipeline.py
Master sequential pipeline — runs everything end to end.

Steps (each is skipped if output already exists):
  1. Grid search  — find best hyperparams for all 5 datasets (Phase 1+2)
  2. Ablation     — 5 variants × 5 datasets × 5 seeds
  3. Sensitivity  — param sweeps × 5 datasets × 3 seeds
  4. Convergence  — training curves × 5 datasets × 3 seeds
  5. Complexity   — timing/memory/params

Usage:
  python experiments/run_pipeline.py --device cuda
  python experiments/run_pipeline.py --device cuda --skip_sensitivity  (faster run)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
SCRIPTS = Path(__file__).parent

DATASETS = ['infectious', 'dblp', 'tumblr', 'mit', 'highschool']


def stamp():
    return datetime.now().strftime('%H:%M:%S')


def run_step(name: str, cmd: list[str], log_file: str) -> bool:
    print(f"\n{'='*70}")
    print(f"[{stamp()}] STEP: {name}")
    print(f"{'='*70}", flush=True)

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, 'w') as lf:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, cwd=str(ROOT)
        )
        for line in proc.stdout:
            print(line, end='', flush=True)
            lf.write(line)
        proc.wait()

    ok = proc.returncode == 0
    status = "DONE" if ok else f"FAILED (exit {proc.returncode})"
    print(f"\n[{stamp()}] {name}: {status}", flush=True)
    return ok


def all_gs_done(gs_dir):
    return all(
        os.path.exists(os.path.join(gs_dir, f"{ds}_gs_result.json"))
        for ds in DATASETS
    )


def all_ablation_done(abl_dir):
    return os.path.exists(os.path.join(abl_dir, 'ablation_summary.json'))


def all_sensitivity_done(sens_dir):
    return os.path.exists(os.path.join(sens_dir, 'sensitivity_summary.json'))


def all_convergence_done(conv_dir):
    return all(
        os.path.exists(os.path.join(conv_dir, f"convergence_{ds}.json"))
        for ds in DATASETS
    )


def all_complexity_done(cplx_dir):
    return os.path.exists(os.path.join(cplx_dir, 'complexity_results.json'))


def print_final_summary(gs_dir, t3former):
    print(f"\n\n{'='*70}")
    print(f"{'FINAL RESULTS SUMMARY':^70}")
    print(f"{'='*70}")
    print(f"{'Dataset':12s} {'TCBG (tuned)':>14s} {'T3Former':>12s} {'Delta':>10s}")
    print('-' * 55)
    for ds in DATASETS:
        gs_file = os.path.join(gs_dir, f"{ds}_gs_result.json")
        if os.path.exists(gs_file):
            d = json.load(open(gs_file))
            m = d['mean_acc'] * 100
            s = d['std_acc'] * 100
            t3 = t3former.get(ds, 0)
            print(f"{ds:12s} {m:>10.2f}% +/- {s:.2f}%  {t3:>10.2f}%  {m-t3:>+8.2f}%")
        else:
            print(f"{ds:12s} {'NOT DONE':>14s}")
    print('=' * 55)


def main():
    parser = argparse.ArgumentParser(description='TCBG Full Pipeline')
    parser.add_argument('--device',      type=str, default='cuda')
    parser.add_argument('--data_root',   type=str, default='./data/raw')
    parser.add_argument('--cache_dir',   type=str, default='./data/cache')
    parser.add_argument('--gs_dir',      type=str, default='./results_gs')
    parser.add_argument('--result_dir',  type=str, default='./results')
    parser.add_argument('--log_dir',     type=str, default='./logs')
    parser.add_argument('--skip_sensitivity', action='store_true')
    parser.add_argument('--skip_convergence', action='store_true')
    parser.add_argument('--skip_complexity',  action='store_true')
    args = parser.parse_args()

    t3former = {'infectious': 68.50, 'dblp': 60.90, 'tumblr': 63.20,
                'highschool': 67.20, 'mit': 73.16}

    os.makedirs(args.log_dir, exist_ok=True)
    py = sys.executable
    t_start = time.time()

    print(f"[{stamp()}] TCBG Pipeline Started")
    print(f"Device: {args.device}  Datasets: {DATASETS}\n")

    # ----------------------------------------------------------------
    # Step 1 — Grid Search (find best hyperparams + Phase 2 full eval)
    # ----------------------------------------------------------------
    if all_gs_done(args.gs_dir):
        print(f"[{stamp()}] Step 1 (Grid Search): ALL DONE — skipping")
    else:
        # Run datasets that are not yet done
        pending = [ds for ds in DATASETS
                   if not os.path.exists(
                       os.path.join(args.gs_dir, f"{ds}_gs_result.json"))]
        print(f"[{stamp()}] Step 1 (Grid Search): pending={pending}")
        run_step(
            "Grid Search",
            [py, str(SCRIPTS / 'grid_search_tuning.py'),
             '--datasets'] + pending + [
             '--device', args.device,
             '--data_root', args.data_root,
             '--cache_dir', args.cache_dir,
             '--result_dir', args.gs_dir],
            os.path.join(args.log_dir, 'grid_search.log'),
        )

    # ----------------------------------------------------------------
    # Step 2 — Ablation Study
    # ----------------------------------------------------------------
    abl_dir = os.path.join(args.result_dir, 'ablation')
    if all_ablation_done(abl_dir):
        print(f"\n[{stamp()}] Step 2 (Ablation): ALL DONE — skipping")
    else:
        run_step(
            "Ablation Study",
            [py, str(SCRIPTS / 'ablation_study.py'),
             '--device', args.device,
             '--data_root', args.data_root,
             '--cache_dir', args.cache_dir,
             '--result_dir', abl_dir,
             '--gs_dir', args.gs_dir],
            os.path.join(args.log_dir, 'ablation.log'),
        )

    # ----------------------------------------------------------------
    # Step 3 — Sensitivity Analysis
    # ----------------------------------------------------------------
    sens_dir = os.path.join(args.result_dir, 'sensitivity')
    if args.skip_sensitivity:
        print(f"\n[{stamp()}] Step 3 (Sensitivity): SKIPPED by flag")
    elif all_sensitivity_done(sens_dir):
        print(f"\n[{stamp()}] Step 3 (Sensitivity): ALL DONE — skipping")
    else:
        run_step(
            "Sensitivity Analysis",
            [py, str(SCRIPTS / 'sensitivity_analysis.py'),
             '--device', args.device,
             '--data_root', args.data_root,
             '--cache_dir', args.cache_dir,
             '--result_dir', sens_dir,
             '--gs_dir', args.gs_dir],
            os.path.join(args.log_dir, 'sensitivity.log'),
        )

    # ----------------------------------------------------------------
    # Step 4 — Convergence Analysis
    # ----------------------------------------------------------------
    conv_dir = os.path.join(args.result_dir, 'convergence')
    if args.skip_convergence:
        print(f"\n[{stamp()}] Step 4 (Convergence): SKIPPED by flag")
    elif all_convergence_done(conv_dir):
        print(f"\n[{stamp()}] Step 4 (Convergence): ALL DONE — skipping")
    else:
        run_step(
            "Convergence Analysis",
            [py, str(SCRIPTS / 'convergence_analysis.py'),
             '--device', args.device,
             '--cache_dir', args.cache_dir,
             '--result_dir', conv_dir,
             '--gs_dir', args.gs_dir],
            os.path.join(args.log_dir, 'convergence.log'),
        )

    # ----------------------------------------------------------------
    # Step 5 — Complexity Analysis
    # ----------------------------------------------------------------
    cplx_dir = os.path.join(args.result_dir, 'complexity')
    if args.skip_complexity:
        print(f"\n[{stamp()}] Step 5 (Complexity): SKIPPED by flag")
    elif all_complexity_done(cplx_dir):
        print(f"\n[{stamp()}] Step 5 (Complexity): ALL DONE — skipping")
    else:
        run_step(
            "Complexity Analysis",
            [py, str(SCRIPTS / 'complexity_analysis.py'),
             '--device', args.device,
             '--data_root', args.data_root,
             '--cache_dir', args.cache_dir,
             '--result_dir', cplx_dir,
             '--gs_dir', args.gs_dir,
             '--skip_preprocessing'],  # skip slow re-preprocessing
            os.path.join(args.log_dir, 'complexity.log'),
        )

    # ----------------------------------------------------------------
    # Step 6 — Runtime Analysis
    # ----------------------------------------------------------------
    rt_file = os.path.join(cplx_dir, 'runtime_results.json')
    if os.path.exists(rt_file):
        print(f"\n[{stamp()}] Step 6 (Runtime): ALL DONE — skipping")
    else:
        run_step(
            "Runtime Analysis",
            [py, str(SCRIPTS / 'runtime_analysis.py'),
             '--device', args.device,
             '--data_root', args.data_root,
             '--cache_dir', args.cache_dir,
             '--result_dir', cplx_dir,
             '--gs_dir', args.gs_dir,
             '--skip_preprocessing'],
            os.path.join(args.log_dir, 'runtime.log'),
        )

    # ----------------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------------
    elapsed = (time.time() - t_start) / 3600
    print(f"\n\n[{stamp()}] Pipeline complete in {elapsed:.1f} hours")
    print_final_summary(args.gs_dir, t3former)
    print(f"\nAll logs saved to {args.log_dir}/")
    print(f"Grid search results: {args.gs_dir}/")
    print(f"Analysis results: {args.result_dir}/")


if __name__ == '__main__':
    main()
