"""
experiments/grid_search_tuning.py
Hyperparameter grid search for TCBG on Tumblr, Highschool, MIT datasets.

Mirrors T3Former's grid search exactly:
  lr          in {0.01, 0.005, 0.001}
  gin_dropout in {0.0,  0.3,   0.5}
  gin_hidden  in {16,   32,    64,   128}

Strategy (efficient — reuses cached Graphcodes):
  Phase 1 — Quick search:  5-fold CV x 3 seeds per combo  → pick best
  Phase 2 — Full eval:     5-fold CV x 10 seeds, best combo → results_gs/

Usage:
  python experiments/grid_search_tuning.py --datasets tumblr highschool mit --device cuda
  python experiments/grid_search_tuning.py --datasets tumblr --device cuda --result_dir ./results_gs
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from itertools import product
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gin_classifier import build_gin
from data.social_loader import load_social_dataset, get_social_folds

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Grid — same as T3Former Table 4 / Section 4.1
# ---------------------------------------------------------------------------

LR_GRID      = [0.01, 0.005, 0.001]
DROPOUT_GRID = [0.0, 0.3, 0.5]
HIDDEN_GRID  = [16, 32, 64, 128]

DATASET_GRID_DIMS = {
    'infectious': (30, 20),
    'dblp':       (30, 20),
    'tumblr':     (30, 20),
    'mit':        (30, 20),
    'highschool': (30, 20),
}
DATASET_TYPE = 'social'

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0; correct = 0; total = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, batch.y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += float(loss.item()) * batch.num_graphs
        preds = logits.argmax(dim=-1)
        correct += int((preds == batch.y.view(-1)).sum())
        total += batch.num_graphs
    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0; correct = 0; total = 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        loss = criterion(logits, batch.y.view(-1))
        total_loss += float(loss.item()) * batch.num_graphs
        preds = logits.argmax(dim=-1)
        correct += int((preds == batch.y.view(-1)).sum())
        total += batch.num_graphs
    return total_loss / max(total, 1), correct / max(total, 1)


def train_fold(train_data, val_data, test_data, num_classes, config, device):
    """Train one fold, return best-val-checkpoint test accuracy."""
    batch_size = config.get('batch_size', 32)
    epochs     = config.get('epochs', 200)
    patience   = config.get('patience', 30)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=batch_size)
    test_loader  = DataLoader(test_data,  batch_size=batch_size)

    model     = build_gin(config, num_classes=num_classes).to(device)
    optimizer = Adam(model.parameters(),
                     lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_loss = float('inf')
    best_test_acc = 0.0
    patience_ctr  = 0

    for epoch in range(1, epochs + 1):
        train_epoch(model, train_loader, optimizer, criterion, device)
        v_loss, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(v_loss)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            patience_ctr  = 0
            _, test_acc = evaluate(model, test_loader, criterion, device)
            best_test_acc = test_acc
        else:
            patience_ctr += 1

        if patience_ctr >= patience:
            break

    return best_test_acc


def run_cv(data_list, labels, num_classes, config, device, seeds):
    """Run 5-fold CV over multiple seeds. Returns list of per-seed mean accs."""
    seed_means = []
    for seed in seeds:
        set_seed(seed)
        folds = get_social_folds(labels, n_splits=5, seed=seed)
        fold_accs = []
        for train_val_idx, test_idx in folds:
            n_tv  = len(train_val_idx)
            n_val = max(1, int(n_tv * 0.2))
            val_idx   = train_val_idx[-n_val:]
            train_idx = train_val_idx[:-n_val]

            train_data = [data_list[i] for i in train_idx]
            val_data   = [data_list[i] for i in val_idx]
            test_data  = [data_list[i] for i in test_idx]

            acc = train_fold(train_data, val_data, test_data,
                             num_classes, config, device)
            fold_accs.append(acc)

        seed_means.append(float(np.mean(fold_accs)))

    return seed_means


# ---------------------------------------------------------------------------
# Graphcode loading (reuses existing cache from train.py run)
# ---------------------------------------------------------------------------

def load_graphcodes(dataset: str, num_classes: int, data_root: str,
                    cache_dir: str, device_str: str) -> Tuple[List[Data], List[int]]:
    """
    Load pre-computed Graphcodes from disk cache.
    If cache is missing, build it using the full TCBG pipeline.
    Cache key is identical to train.py so we never recompute.
    """
    # Fixed pipeline params (must match train.py exactly)
    mp  = 0.05
    mb  = 20
    nfd = 8
    gfd = 21
    t_grid, k_grid = DATASET_GRID_DIMS[dataset]
    dtype = DATASET_TYPE

    cache_key  = (f"{dataset}_nc{num_classes}_T{t_grid}_K{k_grid}"
                  f"_mp{mp}_mb{mb}_nf{nfd}_gf{gfd}_eps{dtype}")
    cache_file = os.path.join(cache_dir, f"{cache_key}.pt")

    if os.path.exists(cache_file):
        print(f"[cache] Loading Graphcodes from {cache_file}")
        saved = torch.load(cache_file, weights_only=False)
        return saved['data_list'], saved['labels']

    # Cache miss — run the full pipeline (same as train.py)
    print(f"[cache] Cache not found — computing Graphcodes for {dataset} ...")
    from src.pipeline import TCBGPipeline
    from data.utils import print_dataset_stats

    config = {
        'T_grid': t_grid, 'K_grid': k_grid,
        'epsilon': 'auto', 'hom_dim': [0, 1],
        'min_persistence': mp, 'max_bars_per_level': mb,
        'node_feat_dim': nfd, 'global_feat_dim': gfd,
        'dataset_type': dtype,
        'num_timesteps': None,
    }
    pipeline = TCBGPipeline(config)
    graphs, nodes, labels = load_social_dataset(dataset, root=data_root, verbose=True)
    graph_triples = list(zip(graphs, nodes, labels))
    data_list = pipeline.process_dataset(graph_triples, verbose=True)

    os.makedirs(cache_dir, exist_ok=True)
    torch.save({'data_list': data_list, 'labels': labels}, cache_file)
    print(f"[cache] Saved to {cache_file}")
    return data_list, labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='TCBG Hyperparameter Grid Search')
    parser.add_argument('--datasets', nargs='+',
                        default=['infectious', 'dblp', 'tumblr', 'mit', 'highschool'],
                        choices=['infectious', 'dblp', 'tumblr', 'mit', 'highschool'])
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--device',      type=str, default='cuda')
    parser.add_argument('--data_root',   type=str, default='./data/raw')
    parser.add_argument('--cache_dir',   type=str, default='./data/cache')
    parser.add_argument('--result_dir',  type=str, default='./results_gs')
    # Phase 1: how many seeds for the quick search
    parser.add_argument('--search_seeds', type=int, default=3,
                        help='Seeds used in Phase 1 quick grid search (default 3)')
    # Phase 2: full eval seeds
    parser.add_argument('--full_seeds',   type=int, default=10,
                        help='Seeds used in Phase 2 full evaluation (default 10)')
    args = parser.parse_args()

    device = torch.device(
        args.device if (torch.cuda.is_available() or args.device == 'cpu') else 'cpu'
    )
    print(f"Device: {device}")
    os.makedirs(args.result_dir, exist_ok=True)

    all_combos = list(product(LR_GRID, DROPOUT_GRID, HIDDEN_GRID))
    n_combos   = len(all_combos)
    print(f"Grid: {len(LR_GRID)} lr × {len(DROPOUT_GRID)} dropout × "
          f"{len(HIDDEN_GRID)} hidden = {n_combos} combos")

    summary = {}

    for dataset in args.datasets:
        print(f"\n{'='*60}")
        print(f"DATASET: {dataset.upper()}")
        print(f"{'='*60}")

        data_list, labels = load_graphcodes(
            dataset, args.num_classes, args.data_root, args.cache_dir, args.device
        )
        num_classes = len(set(labels))
        print(f"Graphs: {len(data_list)}  Classes: {num_classes}")

        # ----------------------------------------------------------------
        # Phase 1 — Quick grid search (search_seeds seeds per combo)
        # ----------------------------------------------------------------
        print(f"\n[Phase 1] Searching {n_combos} combos × {args.search_seeds} seeds ...")
        search_seeds = list(range(args.search_seeds))

        combo_results = []
        t0 = time.time()
        for ci, (lr, dropout, hidden) in enumerate(all_combos, 1):
            config = {
                'gin_layers':  3,
                'gin_hidden':  hidden,
                'gin_dropout': dropout,
                'node_feat_dim':   8,
                'global_feat_dim': 21,
                'batch_size':  32,
                'epochs':      200,
                'patience':    30,
                'lr':          lr,
                'weight_decay': 1e-4,
            }
            seed_means = run_cv(data_list, labels, num_classes, config,
                                device, search_seeds)
            mean_acc = float(np.mean(seed_means))
            combo_results.append({
                'lr': lr, 'dropout': dropout, 'hidden': hidden,
                'mean_acc': mean_acc, 'seed_means': seed_means,
            })
            elapsed = time.time() - t0
            eta = elapsed / ci * (n_combos - ci)
            print(f"  [{ci:2d}/{n_combos}] lr={lr}  drop={dropout}  h={hidden:3d}"
                  f"  => {mean_acc*100:.2f}%  "
                  f"(elapsed {elapsed/60:.1f}m  eta {eta/60:.1f}m)")

        # Sort and show top 5
        combo_results.sort(key=lambda x: x['mean_acc'], reverse=True)
        best = combo_results[0]
        print(f"\n[Phase 1] Top 5 combos for {dataset}:")
        for r in combo_results[:5]:
            marker = ' <<< BEST' if r is combo_results[0] else ''
            print(f"  lr={r['lr']}  drop={r['dropout']}  h={r['hidden']:3d}"
                  f"  => {r['mean_acc']*100:.2f}%{marker}")

        # Save Phase 1 search log
        gs_log_file = os.path.join(args.result_dir, f"{dataset}_gs_log.json")
        with open(gs_log_file, 'w') as f:
            json.dump({'dataset': dataset, 'search_seeds': search_seeds,
                       'combos': combo_results}, f, indent=2)
        print(f"[Phase 1] Search log saved to {gs_log_file}")

        # ----------------------------------------------------------------
        # Phase 2 — Full evaluation with best hyperparameters
        # ----------------------------------------------------------------
        full_seeds = list(range(args.full_seeds))
        print(f"\n[Phase 2] Full eval: lr={best['lr']}  dropout={best['dropout']}"
              f"  hidden={best['hidden']}  × {args.full_seeds} seeds ...")

        best_config = {
            'gin_layers':  3,
            'gin_hidden':  best['hidden'],
            'gin_dropout': best['dropout'],
            'node_feat_dim':   8,
            'global_feat_dim': 21,
            'batch_size':  32,
            'epochs':      200,
            'patience':    30,
            'lr':          best['lr'],
            'weight_decay': 1e-4,
        }

        all_seed_means = []
        for seed in full_seeds:
            set_seed(seed)
            folds = get_social_folds(labels, n_splits=5, seed=seed)
            fold_accs = []
            for fold_idx, (train_val_idx, test_idx) in enumerate(folds):
                n_tv  = len(train_val_idx)
                n_val = max(1, int(n_tv * 0.2))
                val_idx   = train_val_idx[-n_val:]
                train_idx = train_val_idx[:-n_val]

                train_data = [data_list[i] for i in train_idx]
                val_data   = [data_list[i] for i in val_idx]
                test_data  = [data_list[i] for i in test_idx]

                acc = train_fold(train_data, val_data, test_data,
                                 num_classes, best_config, device)
                fold_accs.append(acc)

            seed_mean = float(np.mean(fold_accs))
            all_seed_means.append(seed_mean)
            print(f"  Seed {seed}: {seed_mean*100:.2f}%  "
                  f"(running mean {np.mean(all_seed_means)*100:.2f}%)")

        final_mean = float(np.mean(all_seed_means))
        final_std  = float(np.std(all_seed_means))
        print(f"\n[Phase 2] {dataset.upper()} FINAL: {final_mean*100:.2f}% +/- {final_std*100:.2f}%")
        print(f"          Best config: lr={best['lr']}  dropout={best['dropout']}"
              f"  hidden={best['hidden']}")

        # Save final result
        result = {
            'dataset':      dataset,
            'num_classes':  num_classes,
            'best_lr':      best['lr'],
            'best_dropout': best['dropout'],
            'best_hidden':  best['hidden'],
            'seed_means':   all_seed_means,
            'mean_acc':     final_mean,
            'std_acc':      final_std,
            'search_seeds': args.search_seeds,
            'full_seeds':   args.full_seeds,
        }
        result_file = os.path.join(args.result_dir, f"{dataset}_gs_result.json")
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"[Phase 2] Result saved to {result_file}")

        summary[dataset] = result

    # ----------------------------------------------------------------
    # Final summary table
    # ----------------------------------------------------------------
    t3former = {'infectious': 68.50, 'dblp': 60.90, 'tumblr': 63.20,
                'highschool': 67.20, 'mit': 73.16}
    baseline = {'infectious': 68.75, 'dblp': 89.63, 'tumblr': 56.13,
                'highschool': 58.22, 'mit': 60.79}

    print(f"\n{'='*70}")
    print(f"{'FINAL GRID SEARCH RESULTS':^70}")
    print(f"{'='*70}")
    print(f"{'Dataset':12s} {'TCBG-fixed':>12s} {'TCBG-tuned':>12s} {'T3Former':>12s} {'Delta':>8s}")
    print(f"{'-'*70}")
    for ds, res in summary.items():
        m = res['mean_acc'] * 100
        s = res['std_acc']  * 100
        t3 = t3former.get(ds, 0)
        bl = baseline.get(ds, 0)
        print(f"{ds:12s} {bl:>10.2f}%  {m:>10.2f}%  {t3:>10.2f}%  {m-t3:>+7.2f}%")
    print(f"{'='*70}")

    summary_file = os.path.join(args.result_dir, 'gs_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull summary saved to {summary_file}")


if __name__ == '__main__':
    main()
