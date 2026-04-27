"""
experiments/convergence_analysis.py  — COMPREHENSIVE VERSION
Training convergence analysis for TCBG across all 5 datasets.

Records per-epoch:
  train_loss, val_loss, val_acc, test_acc (at best-val checkpoint),
  learning_rate

Averaged over 5 folds × 5 seeds for statistical stability.
Uses best hyperparams from grid search.

Also records:
  - Epoch at which best val loss was achieved (convergence speed)
  - Final test accuracy distribution (box-plot data)
  - Learning rate schedule trace

Usage:
  python experiments/convergence_analysis.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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
from data.social_loader import get_social_folds

warnings.filterwarnings('ignore')

DATASETS   = ['infectious', 'dblp', 'tumblr', 'mit', 'highschool']
CONV_SEEDS = list(range(5))    # 5 seeds for robust averaging
EPOCHS     = 200


def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def train_epoch(model, loader, opt, crit, device):
    model.train()
    tl = 0; total = 0
    for batch in loader:
        batch = batch.to(device)
        opt.zero_grad()
        logits = model(batch)
        loss = crit(logits, batch.y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tl += float(loss.item()) * batch.num_graphs
        total += batch.num_graphs
    return tl / max(total, 1)


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


def run_fold_with_history(tr, va, te, num_classes, config, device):
    """Run one fold and return full per-epoch history."""
    model = build_gin(config, num_classes).to(device)
    opt   = Adam(model.parameters(),
                 lr=config['lr'], weight_decay=config.get('weight_decay', 1e-4))
    sched = ReduceLROnPlateau(opt, mode='min', patience=10, factor=0.5)
    crit  = nn.CrossEntropyLoss(label_smoothing=0.1)
    tr_l  = DataLoader(tr, batch_size=32, shuffle=True)
    va_l  = DataLoader(va, batch_size=32)
    te_l  = DataLoader(te, batch_size=32)

    hist = {
        'train_loss': [], 'val_loss': [], 'val_acc': [],
        'test_acc_at_best': [], 'lr': [],
    }
    best_val  = float('inf'); best_epoch = 0; pat = 0
    best_test = 0.0

    for epoch in range(1, EPOCHS + 1):
        tl    = train_epoch(model, tr_l, opt, crit, device)
        vl, va_acc = evaluate(model, va_l, crit, device)
        sched.step(vl)
        current_lr = get_lr(opt)

        hist['train_loss'].append(tl)
        hist['val_loss'].append(vl)
        hist['val_acc'].append(va_acc)
        hist['lr'].append(current_lr)

        if vl < best_val:
            best_val = vl; best_epoch = epoch; pat = 0
            _, acc = evaluate(model, te_l, crit, device)
            best_test = acc
        else:
            pat += 1

        # Record test acc at best val checkpoint (carries forward)
        hist['test_acc_at_best'].append(best_test)

        if pat >= 30:
            break

    hist['best_epoch']    = best_epoch
    hist['best_test_acc'] = best_test
    hist['n_epochs']      = epoch
    return hist


def aggregate_histories(histories: list) -> dict:
    """Average per-epoch curves across folds/seeds. Handles early stopping."""
    min_len = min(len(h['train_loss']) for h in histories)
    keys = ['train_loss', 'val_loss', 'val_acc', 'test_acc_at_best', 'lr']
    result = {}
    for k in keys:
        arr = np.array([h[k][:min_len] for h in histories])
        result[f'{k}_mean'] = arr.mean(0).tolist()
        result[f'{k}_std']  = arr.std(0).tolist()

    result['epochs']        = list(range(1, min_len + 1))
    result['best_epoch_mean']  = float(np.mean([h['best_epoch'] for h in histories]))
    result['best_epoch_std']   = float(np.std([h['best_epoch'] for h in histories]))
    result['best_test_mean']   = float(np.mean([h['best_test_acc'] for h in histories]))
    result['best_test_std']    = float(np.std([h['best_test_acc'] for h in histories]))
    result['n_epochs_mean']    = float(np.mean([h['n_epochs'] for h in histories]))
    result['all_best_test']    = [h['best_test_acc'] for h in histories]
    return result


def load_best_params(dataset: str, gs_dir: str) -> dict:
    f = os.path.join(gs_dir, f"{dataset}_gs_result.json")
    if os.path.exists(f):
        d = json.load(open(f))
        return {'lr': d['best_lr'], 'gin_dropout': d['best_dropout'],
                'gin_hidden': d['best_hidden']}
    return {'lr': 0.001, 'gin_dropout': 0.3, 'gin_hidden': 64}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets',   nargs='+', default=DATASETS)
    parser.add_argument('--device',     type=str, default='cuda')
    parser.add_argument('--cache_dir',  type=str, default='./data/cache')
    parser.add_argument('--result_dir', type=str, default='./results/convergence')
    parser.add_argument('--gs_dir',     type=str, default='./results_gs')
    parser.add_argument('--force',      action='store_true',
                        help='Re-run even if output exists')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.result_dir, exist_ok=True)
    print(f"Device: {device} | Seeds: {CONV_SEEDS} | Epochs: {EPOCHS}\n")

    summary = {}

    for dataset in args.datasets:
        out_file = os.path.join(args.result_dir, f"convergence_{dataset}.json")
        if os.path.exists(out_file) and not args.force:
            print(f"[skip] {dataset} — already done")
            summary[dataset] = json.load(open(out_file)).get('best_test_mean', 0)
            continue

        print(f"\n{'='*55}  {dataset.upper()}  {'='*55}")

        # Load standard cache
        cache_key  = f"{dataset}_nc2_T30_K20_mp0.05_mb20_nf8_gf21_epssocial"
        cache_file = os.path.join(args.cache_dir, f"{cache_key}.pt")
        if not os.path.exists(cache_file):
            # try nc4 for mit
            cache_key  = f"{dataset}_nc4_T30_K20_mp0.05_mb20_nf8_gf21_epssocial"
            cache_file = os.path.join(args.cache_dir, f"{cache_key}.pt")
        if not os.path.exists(cache_file):
            print(f"  Cache not found — skipping {dataset}")
            continue

        saved     = torch.load(cache_file, weights_only=False)
        data_list = saved['data_list']
        labels    = saved['labels']
        num_classes = len(set(labels))

        bp = load_best_params(dataset, args.gs_dir)
        config = {
            'gin_layers': 3, 'gin_hidden': bp['gin_hidden'],
            'gin_dropout': bp['gin_dropout'], 'node_feat_dim': 8,
            'global_feat_dim': 21, 'use_jk': True,
            'lr': bp['lr'], 'weight_decay': 1e-4,
        }
        print(f"Config: lr={bp['lr']}  dropout={bp['gin_dropout']}"
              f"  hidden={bp['gin_hidden']}  nc={num_classes}")

        all_fold_histories = []

        for seed in CONV_SEEDS:
            set_seed(seed)
            folds = get_social_folds(labels, n_splits=5, seed=seed)
            for fold_idx, (tv_idx, te_idx) in enumerate(folds):
                nv = max(1, int(len(tv_idx) * 0.2))
                tr = [data_list[i] for i in tv_idx[:-nv]]
                va = [data_list[i] for i in tv_idx[-nv:]]
                te = [data_list[i] for i in te_idx]

                h = run_fold_with_history(tr, va, te, num_classes, config, device)
                all_fold_histories.append(h)
                print(f"  seed={seed} fold={fold_idx+1}  "
                      f"best_test={h['best_test_acc']*100:.2f}%  "
                      f"converged@ep{h['best_epoch']}/{h['n_epochs']}",
                      flush=True)

        # Aggregate
        agg = aggregate_histories(all_fold_histories)
        agg['dataset']  = dataset
        agg['config']   = config
        agg['n_runs']   = len(all_fold_histories)

        print(f"\n  SUMMARY: {agg['best_test_mean']*100:.2f}% ± {agg['best_test_std']*100:.2f}%"
              f"  converged @ ep {agg['best_epoch_mean']:.1f} ± {agg['best_epoch_std']:.1f}"
              f"  (avg {agg['n_epochs_mean']:.0f} total epochs)")

        with open(out_file, 'w') as f:
            json.dump(agg, f, indent=2)
        print(f"  Saved → {out_file}")
        summary[dataset] = agg['best_test_mean']

    # Final summary
    print(f"\n\n{'='*55}")
    print(f"{'CONVERGENCE SUMMARY':^55}")
    print(f"{'='*55}")
    for ds, acc in summary.items():
        if isinstance(acc, float):
            print(f"  {ds:12s}:  {acc*100:.2f}%")
    print(f"\nAll convergence data → {args.result_dir}/")


if __name__ == '__main__':
    main()
