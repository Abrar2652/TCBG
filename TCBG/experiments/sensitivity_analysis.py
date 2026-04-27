"""
experiments/sensitivity_analysis.py  — COMPREHENSIVE VERSION
Sensitivity analysis for TCBG. Varies one parameter at a time while
holding all others at the best found values (from grid search).

Protocol: StratifiedKFold(5), 3 seeds per value — same structure as main.

Parameters swept
  Bifiltration (require new Graphcode cache)
    T_grid          [10, 15, 20, 25, 30, 35, 40]
    K_grid          [1,  5,  10, 15, 20, 25, 30]
    min_persistence [0.00, 0.01, 0.03, 0.05, 0.07, 0.10, 0.15]

  Architecture (reuse existing cache)
    gin_layers      [1, 2, 3, 4, 5]
    gin_hidden      [16, 32, 64, 128, 256]
    dropout         [0.0, 0.1, 0.2, 0.3, 0.5]

  Training (reuse existing cache)
    lr              [0.01, 0.005, 0.001, 0.0005, 0.0001]
    weight_decay    [0.0, 1e-5, 1e-4, 1e-3]
    label_smoothing [0.0, 0.05, 0.1, 0.15, 0.2]

Note: T_grid / K_grid / min_persistence sweeps rebuild Graphcodes per value.
      For speed, these run on 2 representative datasets (dblp, infectious)
      unless --all_datasets is passed.
      Architecture and training sweeps run on all 5 datasets.

Usage:
  python experiments/sensitivity_analysis.py --device cuda
  python experiments/sensitivity_analysis.py --device cuda --params gin_hidden gin_layers
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gin_classifier import build_gin
from src.pipeline import TCBGPipeline
from data.social_loader import load_social_dataset, get_social_folds

warnings.filterwarnings('ignore')

ALL_DATASETS   = ['infectious', 'dblp', 'tumblr', 'mit', 'highschool']
PIPE_DATASETS  = ['dblp', 'infectious']   # for pipeline params (T/K/mp)
SENS_SEEDS     = [0, 1, 2]

# Parameter sweep definitions
# 'group' controls which datasets: 'pipeline' (2) or 'arch' (all 5)
SWEEP_PARAMS = {
    # ── Pipeline bifiltration params (new cache per value) ────────
    'T_grid': {
        'group':   'pipeline',
        'values':  [10, 15, 20, 25, 30, 35, 40],
        'default': 30,
    },
    'K_grid': {
        'group':   'pipeline',
        'values':  [1, 5, 10, 15, 20, 25, 30],
        'default': 20,
    },
    'min_persistence': {
        'group':   'pipeline',
        'values':  [0.00, 0.01, 0.03, 0.05, 0.07, 0.10, 0.15],
        'default': 0.05,
    },
    # ── Architecture params (reuse cache) ─────────────────────────
    'gin_layers': {
        'group':   'arch',
        'values':  [1, 2, 3, 4, 5],
        'default': 3,
    },
    'gin_hidden': {
        'group':   'arch',
        'values':  [16, 32, 64, 128, 256],
        'default': 64,
    },
    'dropout': {
        'group':   'arch',
        'values':  [0.0, 0.1, 0.2, 0.3, 0.5],
        'default': 0.3,
    },
    # ── Training params (reuse cache) ─────────────────────────────
    'lr': {
        'group':   'arch',
        'values':  [0.01, 0.005, 0.001, 0.0005, 0.0001],
        'default': 0.001,
    },
    'weight_decay': {
        'group':   'arch',
        'values':  [0.0, 1e-5, 1e-4, 1e-3],
        'default': 1e-4,
    },
    'label_smoothing': {
        'group':   'arch',
        'values':  [0.0, 0.05, 0.1, 0.15, 0.2],
        'default': 0.1,
    },
}

# Defaults used when NOT sweeping a param
DEFAULTS = {
    'T_grid':          30,
    'K_grid':          20,
    'min_persistence': 0.05,
    'gin_layers':      3,
    'gin_hidden':      64,
    'dropout':         0.3,
    'lr':              0.001,
    'weight_decay':    1e-4,
    'label_smoothing': 0.1,
}


# ------------------------------------------------------------------

def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, opt, crit, device):
    model.train()
    for batch in loader:
        batch = batch.to(device)
        opt.zero_grad()
        loss = crit(model(batch), batch.y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


@torch.no_grad()
def evaluate(model, loader, crit, device):
    model.eval()
    tl = 0; correct = 0; total = 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        tl += float(crit(logits, batch.y.view(-1)).item()) * batch.num_graphs
        correct += int((logits.argmax(-1) == batch.y.view(-1)).sum())
        total += batch.num_graphs
    return tl / max(total, 1), correct / max(total, 1)


def train_one(tr, va, te, num_classes, gin_layers, gin_hidden, dropout,
              lr, weight_decay, label_smoothing, device):
    config = {
        'gin_layers': gin_layers, 'gin_hidden': gin_hidden,
        'gin_dropout': dropout, 'node_feat_dim': 8,
        'global_feat_dim': 21, 'use_jk': True,
    }
    model = build_gin(config, num_classes).to(device)
    opt   = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = ReduceLROnPlateau(opt, mode='min', patience=10, factor=0.5)
    crit  = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    tr_l  = DataLoader(tr, batch_size=32, shuffle=True)
    va_l  = DataLoader(va, batch_size=32)
    te_l  = DataLoader(te, batch_size=32)
    best_val = float('inf'); best_acc = 0.0; pat = 0
    for _ in range(200):
        train_epoch(model, tr_l, opt, crit, device)
        vl, _ = evaluate(model, va_l, crit, device)
        sched.step(vl)
        if vl < best_val:
            best_val = vl; pat = 0
            _, acc = evaluate(model, te_l, crit, device)
            best_acc = acc
        else:
            pat += 1
        if pat >= 30:
            break
    return best_acc


def get_or_build_cache(dataset: str, T_grid: int, K_grid: int,
                        min_persistence: float, cache_dir: str,
                        data_root: str):
    key = (f"{dataset}_nc2_T{T_grid}_K{K_grid}"
           f"_mp{min_persistence}_mb20_nf8_gf21_epssocial")
    # Reuse standard cache when params match defaults
    if T_grid == 30 and K_grid == 20 and min_persistence == 0.05:
        key = f"{dataset}_nc2_T30_K20_mp0.05_mb20_nf8_gf21_epssocial"
    cache_file = os.path.join(cache_dir, f"{key}.pt")
    if os.path.exists(cache_file):
        saved = torch.load(cache_file, weights_only=False)
        return saved['data_list'], saved['labels']

    cfg = {
        'T_grid': T_grid, 'K_grid': K_grid,
        'epsilon': 'auto', 'hom_dim': [0, 1],
        'min_persistence': min_persistence, 'max_bars_per_level': 20,
        'node_feat_dim': 8, 'global_feat_dim': 21,
        'dataset_type': 'social', 'num_timesteps': None,
    }
    pipeline = TCBGPipeline(cfg)
    graphs, nodes, labels = load_social_dataset(
        dataset, root=data_root, verbose=False)
    data_list = pipeline.process_dataset(
        list(zip(graphs, nodes, labels)), verbose=False)
    os.makedirs(cache_dir, exist_ok=True)
    torch.save({'data_list': data_list, 'labels': labels}, cache_file)
    return data_list, labels


def load_best_params(dataset: str, gs_dir: str) -> dict:
    f = os.path.join(gs_dir, f"{dataset}_gs_result.json")
    if os.path.exists(f):
        d = json.load(open(f))
        return {'lr': d['best_lr'], 'dropout': d['best_dropout'],
                'gin_hidden': d['best_hidden']}
    return {'lr': 0.001, 'dropout': 0.3, 'gin_hidden': 64}


def run_value(dataset, T_grid, K_grid, min_persistence,
              gin_layers, gin_hidden, dropout, lr, weight_decay,
              label_smoothing, cache_dir, data_root, device):
    data_list, labels = get_or_build_cache(
        dataset, T_grid, K_grid, min_persistence, cache_dir, data_root)
    num_classes = len(set(labels))
    seed_means = []
    for seed in SENS_SEEDS:
        set_seed(seed)
        folds = get_social_folds(labels, n_splits=5, seed=seed)
        fold_accs = []
        for tv_idx, te_idx in folds:
            nv = max(1, int(len(tv_idx) * 0.2))
            tr = [data_list[i] for i in tv_idx[:-nv]]
            va = [data_list[i] for i in tv_idx[-nv:]]
            te = [data_list[i] for i in te_idx]
            fold_accs.append(train_one(
                tr, va, te, num_classes,
                gin_layers, gin_hidden, dropout, lr,
                weight_decay, label_smoothing, device))
        seed_means.append(float(np.mean(fold_accs)))
    return float(np.mean(seed_means)), float(np.std(seed_means))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets',     nargs='+', default=ALL_DATASETS)
    parser.add_argument('--params',       nargs='+', default=list(SWEEP_PARAMS.keys()),
                        choices=list(SWEEP_PARAMS.keys()))
    parser.add_argument('--all_datasets', action='store_true',
                        help='Run pipeline params on all 5 datasets (slow)')
    parser.add_argument('--device',       type=str, default='cuda')
    parser.add_argument('--data_root',    type=str, default='./data/raw')
    parser.add_argument('--cache_dir',    type=str, default='./data/cache')
    parser.add_argument('--result_dir',   type=str, default='./results/sensitivity')
    parser.add_argument('--gs_dir',       type=str, default='./results_gs')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.result_dir, exist_ok=True)
    print(f"Device: {device} | Seeds: {SENS_SEEDS}")
    print(f"Params: {args.params}\n")

    all_results = {}

    for param_name in args.params:
        spec   = SWEEP_PARAMS[param_name]
        values = spec['values']

        # Decide which datasets to use
        if spec['group'] == 'pipeline' and not args.all_datasets:
            datasets = [d for d in PIPE_DATASETS if d in args.datasets]
        else:
            datasets = args.datasets

        print(f"\n{'='*65}")
        print(f"PARAMETER: {param_name}  values={values}  datasets={datasets}")
        print(f"{'='*65}")

        all_results[param_name] = {}

        for dataset in datasets:
            bp = load_best_params(dataset, args.gs_dir)
            # Start from best params, override DEFAULTS for non-swept params
            d = {
                'T_grid':          DEFAULTS['T_grid'],
                'K_grid':          DEFAULTS['K_grid'],
                'min_persistence': DEFAULTS['min_persistence'],
                'gin_layers':      DEFAULTS['gin_layers'],
                'gin_hidden':      bp.get('gin_hidden', DEFAULTS['gin_hidden']),
                'dropout':         bp.get('dropout',    DEFAULTS['dropout']),
                'lr':              bp.get('lr',         DEFAULTS['lr']),
                'weight_decay':    DEFAULTS['weight_decay'],
                'label_smoothing': DEFAULTS['label_smoothing'],
            }
            means = []; stds = []
            for val in values:
                d[param_name] = val
                t0 = time.time()
                m, s = run_value(
                    dataset,
                    T_grid=d['T_grid'], K_grid=d['K_grid'],
                    min_persistence=d['min_persistence'],
                    gin_layers=d['gin_layers'], gin_hidden=d['gin_hidden'],
                    dropout=d['dropout'], lr=d['lr'],
                    weight_decay=d['weight_decay'],
                    label_smoothing=d['label_smoothing'],
                    cache_dir=args.cache_dir, data_root=args.data_root,
                    device=device)
                means.append(m); stds.append(s)
                print(f"  {dataset:12s}  {param_name}={str(val):8s}  "
                      f"=> {m*100:.2f}% ± {s*100:.2f}%  "
                      f"({(time.time()-t0)/60:.1f}m)", flush=True)
                # restore non-swept value
                d[param_name] = DEFAULTS.get(param_name,
                                             bp.get(param_name, val))

            all_results[param_name][dataset] = {
                'values': values, 'means': means, 'stds': stds,
                'default_val': spec['default'],
            }

        # Save per-parameter
        out = os.path.join(args.result_dir, f"sensitivity_{param_name}.json")
        with open(out, 'w') as f:
            json.dump({'param': param_name, 'values': values,
                       'datasets': all_results[param_name]}, f, indent=2)
        print(f"  → {out}")

    summary = os.path.join(args.result_dir, 'sensitivity_summary.json')
    with open(summary, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSensitivity summary → {summary}")


if __name__ == '__main__':
    main()
