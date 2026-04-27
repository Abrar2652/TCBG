"""
run_tcbg_fair.py — TCBG fair eval across social/brain/traffic with multiple seeds.

Wraps TCBG/experiments/train.py, runs each dataset × seed, aggregates.
Social/traffic: 5-fold CV × 10 seeds.
Brain: 70/10/20 × 10 seeds.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent.resolve()
TCBG_DIR = HERE / 'TCBG'

SOCIAL = ['infectious', 'dblp', 'tumblr', 'mit', 'highschool']
BRAIN  = ['dynhcp_task', 'dynhcp_gender', 'dynhcp_age']
TRAFFIC = ['pems04', 'pems08', 'pemsbay']

BRAIN_NUM_CLASSES = {'dynhcp_task': 7, 'dynhcp_gender': 2, 'dynhcp_age': 3}


def run_one(dataset, seed, device, data_root, cache_dir, num_classes,
            log_path, result_dir):
    cmd = [
        sys.executable, str(TCBG_DIR / 'experiments' / 'train.py'),
        '--dataset', dataset, '--seed', str(seed), '--device', device,
        '--num_classes', str(num_classes),
        '--data_root', str(data_root), '--cache_dir', str(cache_dir),
        '--result_dir', str(result_dir),
        '--quiet',
    ]
    t0 = time.time()
    with open(log_path, 'a') as f:
        f.write(f'\n=== seed={seed} {dataset} @ {time.strftime("%H:%M:%S")} ===\n')
        r = subprocess.run(cmd, cwd=str(TCBG_DIR), stdout=f, stderr=subprocess.STDOUT)
    return r.returncode, time.time() - t0


def extract_result(dataset, seed, result_dir, num_classes):
    """Parse {dataset}_nc{nc}_seed{seed}.json from TCBG results dir."""
    fname = f'{dataset}_nc{num_classes}_seed{seed}.json'
    p = Path(result_dir) / fname
    if p.exists():
        d = json.load(open(p))
        if 'mean_acc' in d:
            return float(d['mean_acc'])
        if 'fold_accs' in d:
            return float(np.mean(d['fold_accs']))
        if 'test_acc' in d:
            return float(d['test_acc'])
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets',   nargs='+', default=SOCIAL,
                   help='datasets to run')
    p.add_argument('--seeds',      type=int, default=10)
    p.add_argument('--device',     default='cuda')
    p.add_argument('--data_root',  default=str(TCBG_DIR / 'data' / 'raw'))
    p.add_argument('--cache_dir',  default=str(TCBG_DIR / 'data' / 'cache'))
    p.add_argument('--result_dir', default=str(HERE / 'results' / 'tcbg_fair'))
    p.add_argument('--num_classes_override', type=int, default=None,
                   help='Override num_classes (for traffic binary vs multi)')
    p.add_argument('--skip_existing', action='store_true')
    args = p.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)
    log_file = Path(args.result_dir) / 'run.log'

    seeds_to_run = list(range(args.seeds))
    print(f'TCBG Fair Eval | device={args.device} | seeds={seeds_to_run}')
    print(f'Datasets: {args.datasets}', flush=True)

    all_res = {}
    for ds in args.datasets:
        nc = args.num_classes_override
        if nc is None:
            nc = BRAIN_NUM_CLASSES.get(ds, 2)
        out_file = Path(args.result_dir) / f'{ds}_nc{nc}_fair.json'
        if out_file.exists() and args.skip_existing:
            all_res[f'{ds}_nc{nc}'] = json.load(open(out_file))
            print(f'[skip] {ds} nc={nc}')
            continue

        print(f'\n{"="*60}  {ds}  (num_classes={nc})')
        seed_accs = []
        for s in seeds_to_run:
            rc, elapsed = run_one(ds, s, args.device, args.data_root,
                                  args.cache_dir, nc, log_file, args.result_dir)
            acc = extract_result(ds, s, args.result_dir, nc)
            if rc != 0:
                print(f'  seed={s}  FAILED rc={rc}  (log: {log_file})', flush=True)
                continue
            if acc is None:
                print(f'  seed={s}  no result extracted', flush=True)
                continue
            seed_accs.append(acc)
            print(f'  seed={s}  acc={acc*100:.2f}%  ({elapsed:.0f}s)', flush=True)

        if not seed_accs:
            print(f'  no successful seeds for {ds}')
            continue

        mean = float(np.mean(seed_accs)); std = float(np.std(seed_accs))
        res = {
            'dataset': ds, 'num_classes': nc,
            'seeds': seeds_to_run[:len(seed_accs)],
            'seed_accs': seed_accs,
            'mean_acc': mean, 'std_acc': std,
        }
        all_res[f'{ds}_nc{nc}'] = res
        json.dump(res, open(out_file, 'w'), indent=2)
        print(f'  TCBG Fair: {mean*100:.2f}% ± {std*100:.2f}%')

    summary = Path(args.result_dir) / 'fair_comparison.json'
    json.dump(all_res, open(summary, 'w'), indent=2)
    print(f'\nSaved -> {summary}')


if __name__ == '__main__':
    main()
