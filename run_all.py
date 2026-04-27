"""
run_all.py — Single entry point for fair baseline evaluation.

Usage:
    python run_all.py --device cuda
    python run_all.py --device cuda --t3former_dir /path/to/T3Former-3311

Steps performed automatically:
    1. Clone TempGNTK repo (if not already present)
    2. Run TempGNTK fair eval (all 5 datasets)
    3. Run T3Former fair eval (all 5 datasets)
    4. Print combined comparison table

Results written to:
    results/tempgntk_fair/
    results/t3former_fair/
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent.resolve()

def run(cmd):
    print(f"\n>>> {' '.join(str(c) for c in cmd)}\n")
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',       default='cuda')
    parser.add_argument('--t3former_dir', default=str(HERE / 'T3Former-3311'))
    parser.add_argument('--skip_tempgntk', action='store_true')
    parser.add_argument('--skip_t3former', action='store_true')
    args = parser.parse_args()

    # ── Step 1: Clone TempGNTK repo ──────────────────────────────────────────
    repo_dir = HERE / 'TempGNTK_repo'
    if not repo_dir.exists():
        print("Cloning TempGNTK repo...")
        run(['git', 'clone', 'https://github.com/kthrn22/TempGNTK.git',
             str(repo_dir)])
    else:
        print(f"TempGNTK_repo already present at {repo_dir}")

    # ── Step 2: TempGNTK fair eval ────────────────────────────────────────────
    if not args.skip_tempgntk:
        print("\n" + "="*60)
        print("RUNNING: TempGNTK Fair Eval")
        print("="*60)
        run([sys.executable, str(HERE / 'run_tempgntk_fair.py'),
             '--device', args.device])

    # ── Step 3: T3Former fair eval ────────────────────────────────────────────
    if not args.skip_t3former:
        if not Path(args.t3former_dir).exists():
            print(f"\nERROR: T3Former dir not found at {args.t3former_dir}")
            print("Place the T3Former-3311 folder here and re-run, or pass:")
            print("  python run_all.py --t3former_dir /path/to/T3Former-3311")
            sys.exit(1)

        print("\n" + "="*60)
        print("RUNNING: T3Former Fair Eval")
        print("="*60)
        run([sys.executable, str(HERE / 'run_t3former_fair.py'),
             '--device', args.device,
             '--t3former_dir', args.t3former_dir])

    print("\n" + "="*60)
    print("ALL DONE. Results in results/tempgntk_fair/ and results/t3former_fair/")
    print("="*60)

if __name__ == '__main__':
    main()
