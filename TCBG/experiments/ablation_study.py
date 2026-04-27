"""
experiments/ablation_study.py  — COMPREHENSIVE VERSION
Ablation study for TCBG across all 5 social network datasets.

Protocol: identical to main results
  StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
  5 seeds (0–4)  — same fold structure, half of main for speed
  Best hyperparams loaded from results_gs/{ds}_gs_result.json

Ablation groups
  Group 1 — Core components
    full            Full TCBG (all components active)
    no_global       w/o Global Features      (global_feat_dim=0)
    no_jk           w/o JK-Max               (use_jk=False, 3 GIN layers kept)
    no_curvature    w/o Forman-Ricci κ        (K_grid=1 → pure time filtration)
    no_global_no_jk w/o Global Features AND JK-Max

  Group 2 — 2-parameter vs reduced/no curvature resolution
    k_grid_3        K_grid=3  (minimal 2D — nearly 1D)
    k_grid_10       K_grid=10 (half curvature resolution)

  Group 3 — Homology dimension
    h0_only         H0 only  (connected components, birth/death of nodes)
    h1_only         H1 only  (loops / cycles)

  Group 4 — GIN architecture depth
    gin_1layer      GIN depth=1  (no message propagation depth)
    gin_2layers     GIN depth=2
    gin_4layers     GIN depth=4  (deeper than default)

  Group 5 — Training protocol
    no_label_smooth No label smoothing  (vanilla CrossEntropy)
    no_scheduler    No ReduceLROnPlateau (fixed lr throughout)

Usage:
  python experiments/ablation_study.py --device cuda
  python experiments/ablation_study.py --device cuda --datasets dblp infectious
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gin_classifier import build_gin
from src.pipeline import TCBGPipeline
from data.social_loader import load_social_dataset, get_social_folds

warnings.filterwarnings('ignore')

DATASETS       = ['infectious', 'dblp', 'tumblr', 'mit', 'highschool', 'facebook']
ABLATION_SEEDS = list(range(5))

# ------------------------------------------------------------------
# Ablation variant registry
# key   → (label, config_overrides, pipeline_overrides)
#   config_overrides  : merged into GIN/training config
#   pipeline_overrides: merged into TCBGPipeline config (T/K/hom_dim)
# ------------------------------------------------------------------
VARIANTS = {
    # ── Group 1: Core components ──────────────────────────────────
    'full': {
        'label': 'Full TCBG',
        'cfg': {},
        'pipe': {},
    },
    'no_global': {
        'label': 'w/o Global Features',
        'cfg': {'global_feat_dim': 0},
        'pipe': {},
    },
    'no_jk': {
        'label': 'w/o JK-Max',
        'cfg': {'use_jk': False},
        'pipe': {},
    },
    'no_curvature': {
        'label': 'w/o Curvature κ (K=1)',
        'cfg': {},
        'pipe': {'K_grid': 1},
    },
    'no_global_no_jk': {
        'label': 'w/o Global + JK-Max',
        'cfg': {'global_feat_dim': 0, 'use_jk': False},
        'pipe': {},
    },

    # ── Group 2: Curvature resolution ────────────────────────────
    'k_grid_3': {
        'label': 'K_grid=3 (near 1D)',
        'cfg': {},
        'pipe': {'K_grid': 3},
    },
    'k_grid_10': {
        'label': 'K_grid=10 (half)',
        'cfg': {},
        'pipe': {'K_grid': 10},
    },

    # ── Group 3: Homology dimension ───────────────────────────────
    'h0_only': {
        'label': 'H0 only (components)',
        'cfg': {},
        'pipe': {'hom_dim': [0]},
    },
    'h1_only': {
        'label': 'H1 only (loops)',
        'cfg': {},
        'pipe': {'hom_dim': [1]},
    },

    # ── Group 4: GIN depth ────────────────────────────────────────
    'gin_1layer': {
        'label': 'GIN depth=1',
        'cfg': {'gin_layers': 1},
        'pipe': {},
    },
    'gin_2layers': {
        'label': 'GIN depth=2',
        'cfg': {'gin_layers': 2},
        'pipe': {},
    },
    'gin_4layers': {
        'label': 'GIN depth=4',
        'cfg': {'gin_layers': 4},
        'pipe': {},
    },

    # ── Group 5: Training protocol ────────────────────────────────
    'no_label_smooth': {
        'label': 'No label smoothing',
        'cfg': {'label_smoothing': 0.0},
        'pipe': {},
    },
    'no_scheduler': {
        'label': 'No LR scheduler',
        'cfg': {'no_scheduler': True},
        'pipe': {},
    },
}


# ------------------------------------------------------------------
# Training helpers
# ------------------------------------------------------------------

def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    tl = 0; correct = 0; total = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, batch.y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tl += float(loss.item()) * batch.num_graphs
        correct += int((logits.argmax(-1) == batch.y.view(-1)).sum())
        total += batch.num_graphs
    return tl / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    tl = 0; correct = 0; total = 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        loss = criterion(logits, batch.y.view(-1))
        tl += float(loss.item()) * batch.num_graphs
        correct += int((logits.argmax(-1) == batch.y.view(-1)).sum())
        total += batch.num_graphs
    return tl / max(total, 1), correct / max(total, 1)


def train_fold(tr, va, te, num_classes, config, device):
    ls = config.get('label_smoothing', 0.1)
    criterion = nn.CrossEntropyLoss(label_smoothing=ls)
    tr_l = DataLoader(tr, batch_size=32, shuffle=True)
    va_l = DataLoader(va, batch_size=32)
    te_l = DataLoader(te, batch_size=32)
    model = build_gin(config, num_classes).to(device)
    opt   = Adam(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    sched = (None if config.get('no_scheduler', False)
             else ReduceLROnPlateau(opt, mode='min', patience=10, factor=0.5))
    best_val = float('inf'); best_acc = 0.0; pat = 0
    for _ in range(200):
        train_epoch(model, tr_l, opt, criterion, device)
        vl, _ = evaluate(model, va_l, criterion, device)
        if sched:
            sched.step(vl)
        if vl < best_val:
            best_val = vl; pat = 0
            _, acc = evaluate(model, te_l, criterion, device)
            best_acc = acc
        else:
            pat += 1
        if pat >= 30:
            break
    return best_acc


# ------------------------------------------------------------------
# Cache management (handles K_grid and hom_dim variants)
# ------------------------------------------------------------------

def get_cache_path(dataset: str, num_classes: int, K_grid: int,
                   hom_dim: list, cache_dir: str) -> str:
    hd_str = ''.join(str(d) for d in sorted(hom_dim))
    key = (f"{dataset}_nc{num_classes}_T30_K{K_grid}"
           f"_mp0.05_mb20_nf8_gf21_epssocial_hd{hd_str}")
    # For default (K=20, hd=01), reuse the standard cache produced by train.py
    if K_grid == 20 and sorted(hom_dim) == [0, 1]:
        key = f"{dataset}_nc{num_classes}_T30_K20_mp0.05_mb20_nf8_gf21_epssocial"
    return os.path.join(cache_dir, f"{key}.pt")


def load_or_build(dataset: str, num_classes: int, K_grid: int,
                  hom_dim: list, cache_dir: str, data_root: str):
    cache_file = get_cache_path(dataset, num_classes, K_grid, hom_dim, cache_dir)
    if os.path.exists(cache_file):
        saved = torch.load(cache_file, weights_only=False)
        return saved['data_list'], saved['labels']

    print(f"    [cache miss] Building K={K_grid} hom_dim={hom_dim} for {dataset}...")
    config = {
        'T_grid': 30, 'K_grid': K_grid,
        'epsilon': 'auto', 'hom_dim': hom_dim,
        'min_persistence': 0.05, 'max_bars_per_level': 20,
        'node_feat_dim': 8, 'global_feat_dim': 21,
        'dataset_type': 'social', 'num_timesteps': None,
    }
    pipeline = TCBGPipeline(config)
    graphs, nodes, labels = load_social_dataset(
        dataset, root=data_root, verbose=False)
    data_list = pipeline.process_dataset(
        list(zip(graphs, nodes, labels)), verbose=False)
    os.makedirs(cache_dir, exist_ok=True)
    torch.save({'data_list': data_list, 'labels': labels}, cache_file)
    return data_list, labels


# ------------------------------------------------------------------
# Load best hyperparams from grid search
# ------------------------------------------------------------------

def load_best_params(dataset: str, gs_dir: str) -> dict:
    f = os.path.join(gs_dir, f"{dataset}_gs_result.json")
    if os.path.exists(f):
        d = json.load(open(f))
        return {'lr': d['best_lr'], 'gin_dropout': d['best_dropout'],
                'gin_hidden': d['best_hidden']}
    return {'lr': 0.001, 'gin_dropout': 0.3, 'gin_hidden': 64}


# ------------------------------------------------------------------
# Run one variant across seeds/folds
# ------------------------------------------------------------------

def run_variant(dataset: str, vname: str, vdef: dict,
                base_params: dict, device: torch.device,
                num_classes: int, cache_dir: str, data_root: str) -> tuple:
    K_grid  = vdef['pipe'].get('K_grid', 20)
    hom_dim = vdef['pipe'].get('hom_dim', [0, 1])

    data_list, labels = load_or_build(
        dataset, num_classes, K_grid, hom_dim, cache_dir, data_root)

    # Merge base params + variant overrides
    config = {
        'gin_layers':      3,
        'gin_hidden':      base_params['gin_hidden'],
        'gin_dropout':     base_params['gin_dropout'],
        'node_feat_dim':   8,
        'global_feat_dim': 21,
        'use_jk':          True,
        'label_smoothing': 0.1,
        'no_scheduler':    False,
        'lr':              base_params['lr'],
    }
    config.update(vdef['cfg'])

    seed_means = []
    for seed in ABLATION_SEEDS:
        set_seed(seed)
        folds = get_social_folds(labels, n_splits=5, seed=seed)
        fold_accs = []
        for tv_idx, te_idx in folds:
            nv = max(1, int(len(tv_idx) * 0.2))
            tr = [data_list[i] for i in tv_idx[:-nv]]
            va = [data_list[i] for i in tv_idx[-nv:]]
            te = [data_list[i] for i in te_idx]
            fold_accs.append(train_fold(tr, va, te, num_classes, config, device))
        seed_means.append(float(np.mean(fold_accs)))

    return float(np.mean(seed_means)), float(np.std(seed_means)), seed_means


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='TCBG Comprehensive Ablation Study')
    parser.add_argument('--datasets',  nargs='+', default=DATASETS, choices=DATASETS)
    parser.add_argument('--variants',  nargs='+', default=list(VARIANTS.keys()),
                        choices=list(VARIANTS.keys()))
    parser.add_argument('--device',    type=str, default='cuda')
    parser.add_argument('--data_root', type=str, default='./data/raw')
    parser.add_argument('--cache_dir', type=str, default='./data/cache')
    parser.add_argument('--result_dir',type=str, default='./results/ablation')
    parser.add_argument('--gs_dir',    type=str, default='./results_gs')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.result_dir, exist_ok=True)
    print(f"Device: {device} | Seeds: {ABLATION_SEEDS}")
    print(f"Variants ({len(args.variants)}): {args.variants}\n")

    all_results = {}

    for dataset in args.datasets:
        print(f"\n{'='*65}")
        print(f"DATASET: {dataset.upper()}")
        print(f"{'='*65}")
        graphs, _, labels_raw = load_social_dataset(
            dataset, root=args.data_root, verbose=False)
        num_classes = len(set(labels_raw))
        base_params = load_best_params(dataset, args.gs_dir)
        print(f"Classes: {num_classes} | Best params: {base_params}")

        ds_results = {}
        for vname in args.variants:
            if vname not in VARIANTS:
                continue
            vdef = VARIANTS[vname]
            t0 = time.time()
            mean, std, seed_means = run_variant(
                dataset, vname, vdef, base_params, device,
                num_classes, args.cache_dir, args.data_root)
            elapsed = time.time() - t0
            print(f"  {vdef['label']:35s}  {mean*100:6.2f}% ± {std*100:.2f}%"
                  f"  ({elapsed/60:.1f}m)", flush=True)
            ds_results[vname] = {
                'label': vdef['label'],
                'mean':  mean,
                'std':   std,
                'seed_means': seed_means,
            }

        all_results[dataset] = ds_results

        # Per-dataset JSON
        out = os.path.join(args.result_dir, f"ablation_{dataset}.json")
        with open(out, 'w') as f:
            json.dump({'dataset': dataset, 'num_classes': num_classes,
                       'seeds': ABLATION_SEEDS, 'variants': ds_results}, f, indent=2)
        print(f"  Saved → {out}")

    # Summary table
    print(f"\n\n{'='*90}")
    print(f"{'ABLATION SUMMARY':^90}")
    print(f"{'='*90}")
    vnames_used = args.variants
    header = f"{'Variant':35s}" + "".join(f"{d:>11s}" for d in args.datasets)
    print(header); print('-' * len(header))

    # Group labels
    groups = [
        ('── Group 1: Core components ──', ['full','no_global','no_jk','no_curvature','no_global_no_jk']),
        ('── Group 2: Curvature resolution ──', ['k_grid_3','k_grid_10']),
        ('── Group 3: Homology dim ──', ['h0_only','h1_only']),
        ('── Group 4: GIN depth ──', ['gin_1layer','gin_2layers','gin_4layers']),
        ('── Group 5: Training ──', ['no_label_smooth','no_scheduler']),
    ]
    for grp_label, grp_variants in groups:
        any_in = any(v in vnames_used for v in grp_variants)
        if not any_in:
            continue
        print(f"\n{grp_label}")
        for vname in grp_variants:
            if vname not in vnames_used or vname not in VARIANTS:
                continue
            row = f"  {VARIANTS[vname]['label']:33s}"
            for ds in args.datasets:
                if ds in all_results and vname in all_results[ds]:
                    m = all_results[ds][vname]['mean'] * 100
                    row += f"  {m:7.2f}%"
                else:
                    row += f"  {'N/A':>7s}"
            print(row)

    summary_file = os.path.join(args.result_dir, 'ablation_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAblation summary → {summary_file}")


if __name__ == '__main__':
    main()
