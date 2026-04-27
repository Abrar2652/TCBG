"""
experiments/train.py
Main training and evaluation script for TCBG.

Usage:
  python experiments/train.py --dataset infectious --seed 42
  python experiments/train.py --dataset dynhcp_task --device cuda
  python experiments/train.py --dataset pems04 --num_classes 3

Supports:
  Social (5-fold CV): infectious, dblp, tumblr, mit, highschool
  Brain (fixed split): dynhcp_task, dynhcp_gender, dynhcp_age
  Traffic (5-fold CV): pems04, pems08, pemsbay  [--num_classes 2 or 3]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import TCBGPipeline
from src.gin_classifier import build_gin
from data.social_loader import load_social_dataset, get_social_folds
from data.brain_loader import load_brain_dataset, get_brain_splits
from data.traffic_loader import load_traffic_dataset, get_traffic_folds

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Dataset routing
# ---------------------------------------------------------------------------

SOCIAL_DATASETS = {'infectious', 'dblp', 'tumblr', 'mit', 'highschool', 'facebook'}
BRAIN_DATASETS  = {'dynhcp_task', 'dynhcp_gender', 'dynhcp_age'}
TRAFFIC_DATASETS = {'pems04', 'pems08', 'pemsbay'}

DATASET_TYPE = {
    **{d: 'social'  for d in SOCIAL_DATASETS},
    **{d: 'brain'   for d in BRAIN_DATASETS},
    **{d: 'traffic' for d in TRAFFIC_DATASETS},
}

DATASET_NUM_TIMESTEPS = {
    'infectious': 48, 'dblp': 46, 'tumblr': 89, 'mit': 5576, 'highschool': 203,
    'facebook': 200,
    'dynhcp_task': 34, 'dynhcp_gender': 34, 'dynhcp_age': 34,
    'pems04': 24, 'pems08': 24, 'pemsbay': 24,
}

DATASET_GRID = {
    'infectious': (30, 20), 'dblp': (30, 20), 'tumblr': (30, 20),
    'mit': (30, 20), 'highschool': (30, 20), 'facebook': (30, 20),
    'dynhcp_task': (34, 15), 'dynhcp_gender': (34, 15), 'dynhcp_age': (34, 15),
    'pems04': (24, 20), 'pems08': (24, 20), 'pemsbay': (24, 20),
}

# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float = 1.0,
) -> Tuple[float, float]:
    """One training epoch. Returns (loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

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
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model. Returns (loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        loss = criterion(logits, batch.y.view(-1))
        total_loss += float(loss.item()) * batch.num_graphs
        preds = logits.argmax(dim=-1)
        correct += int((preds == batch.y.view(-1)).sum())
        total += batch.num_graphs

    return total_loss / max(total, 1), correct / max(total, 1)


def train_fold(
    train_data: List[Data],
    val_data: List[Data],
    test_data: List[Data],
    num_classes: int,
    config: dict,
    device: torch.device,
    verbose: bool = True,
) -> Tuple[float, dict]:
    """
    Train and evaluate on one fold/split.

    Returns (test_accuracy, history_dict)
    """
    batch_size = config.get('batch_size', 32)
    epochs = config.get('epochs', 200)
    patience = config.get('patience', 30)
    lr = config.get('lr', 0.001)
    weight_decay = config.get('weight_decay', 1e-4)
    grad_clip = config.get('gradient_clip', 1.0)
    sched_patience = config.get('scheduler_patience', 10)
    sched_factor = config.get('scheduler_factor', 0.5)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=batch_size)
    test_loader  = DataLoader(test_data,  batch_size=batch_size)

    model = build_gin(config, num_classes=num_classes).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', patience=sched_patience, factor=sched_factor
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_loss = float('inf')
    best_test_acc = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'test_acc': []}

    for epoch in range(1, epochs + 1):
        t_loss, t_acc = train_epoch(model, train_loader, optimizer, criterion, device, grad_clip)
        v_loss, v_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(v_loss)

        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            patience_counter = 0
            # Evaluate on test set at best validation point
            _, test_acc = evaluate(model, test_loader, criterion, device)
            best_test_acc = test_acc
        else:
            patience_counter += 1

        if verbose and epoch % 20 == 0:
            print(f"  Ep {epoch:3d} | train_loss={t_loss:.4f} | "
                  f"val_loss={v_loss:.4f} val_acc={v_acc:.4f} | "
                  f"best_test={best_test_acc:.4f}")

        if patience_counter >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch}")
            break

    history['test_acc'].append(best_test_acc)
    return best_test_acc, history


# ---------------------------------------------------------------------------
# Dataset loading + Graphcode pre-computation
# ---------------------------------------------------------------------------

def load_and_process(
    dataset: str,
    num_classes: int,
    config: dict,
    data_root: str,
    cache_dir: str,
    verbose: bool = True,
) -> Tuple[List[Data], List[int]]:
    """
    Load dataset, run TCBG pipeline, return (pyg_data_list, labels).
    Uses disk cache when available.
    """
    # Resolve dataset-specific config FIRST (needed for cache key and pipeline)
    dtype   = DATASET_TYPE.get(dataset, 'social')
    t_grid, k_grid = DATASET_GRID.get(dataset, (30, 20))
    config.update({
        'T_grid':        config.get('T_grid', t_grid),
        'K_grid':        config.get('K_grid', k_grid),
        'dataset_type':  dtype,
        'num_timesteps': DATASET_NUM_TIMESTEPS.get(dataset),
    })

    mp  = config.get('min_persistence', 0.05)
    mb  = config.get('max_bars_per_level', 30)
    nfd = config.get('node_feat_dim', 8)
    gfd = config.get('global_feat_dim', 21)
    # Bake dataset_type into key so brain/traffic/social caches are separate
    cache_key = (f"{dataset}_nc{num_classes}_T{config['T_grid']}_K{config['K_grid']}"
                 f"_mp{mp}_mb{mb}_nf{nfd}_gf{gfd}_eps{dtype}")
    cache_file = os.path.join(cache_dir, f"{cache_key}.pt")

    if os.path.exists(cache_file):
        if verbose:
            print(f"Loading cached Graphcodes from {cache_file}")
        saved = torch.load(cache_file, weights_only=False)
        return saved['data_list'], saved['labels']

    pipeline = TCBGPipeline(config)  # uses the already-updated config

    if dtype == 'social':
        graphs, nodes, labels = load_social_dataset(dataset, root=data_root, verbose=verbose)
    elif dtype == 'brain':
        graphs, nodes, labels = load_brain_dataset(dataset, root=data_root, verbose=verbose)
    else:
        graphs, nodes, labels = load_traffic_dataset(
            dataset, root=data_root, n_classes=num_classes, verbose=verbose
        )

    if verbose:
        print(f"Computing Graphcodes for {len(graphs)} graphs ...")

    t0 = time.time()
    graph_triples = list(zip(graphs, nodes, labels))
    data_list = pipeline.process_dataset(graph_triples, verbose=verbose)
    elapsed = time.time() - t0

    if verbose:
        print(f"Graphcode computation: {elapsed:.1f}s ({elapsed/len(graphs):.3f}s/graph)")

    os.makedirs(cache_dir, exist_ok=True)
    torch.save({'data_list': data_list, 'labels': labels}, cache_file)
    if verbose:
        print(f"Cached to {cache_file}")

    return data_list, labels


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(args):
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu'
                          else 'cpu')
    print(f"Device: {device} | Dataset: {args.dataset} | Classes: {args.num_classes}")

    config = {
        'T_grid': args.T_grid,
        'K_grid': args.K_grid,
        'epsilon': 'auto',
        'hom_dim': [0, 1],
        'min_persistence': args.min_persistence if args.min_persistence is not None else 0.05,
        'max_bars_per_level': 20,
        'node_feat_dim': 8,        # 8-dim Graphcode node features
        'global_feat_dim': 21,   # 15 structural/spectral + 6 temporal spectral
        'gin_layers': args.gin_layers,
        'gin_hidden': args.gin_hidden,
        'gin_dropout': args.gin_dropout,
        'gin_eps': 0.0,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'patience': args.patience,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'gradient_clip': 1.0,
        'scheduler_patience': 10,
        'scheduler_factor': 0.5,
    }

    data_list, labels = load_and_process(
        dataset=args.dataset,
        num_classes=args.num_classes,
        config=config,
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        verbose=not args.quiet,
    )

    dtype = DATASET_TYPE.get(args.dataset, 'social')

    # ---- Brain: fixed split ----
    if dtype == 'brain':
        n = len(data_list)
        train_idx, val_idx, test_idx = get_brain_splits(n, seed=args.seed)
        train_data = [data_list[i] for i in train_idx]
        val_data   = [data_list[i] for i in val_idx]
        test_data  = [data_list[i] for i in test_idx]

        test_acc, history = train_fold(
            train_data, val_data, test_data,
            num_classes=args.num_classes,
            config=config,
            device=device,
            verbose=not args.quiet,
        )
        print(f"\nResult: {args.dataset} | test_acc = {test_acc:.4f}")
        result = {'dataset': args.dataset, 'test_acc': test_acc}

    # ---- Social / Traffic: 5-fold CV ----
    else:
        if dtype == 'social':
            folds = get_social_folds(labels, n_splits=5, seed=args.seed)
        else:
            folds = get_traffic_folds(labels, n_splits=5, seed=args.seed)

        fold_accs = []
        for fold_idx, (train_val_idx, test_idx) in enumerate(folds):
            # Use last 20% of train_val as validation
            n_tv = len(train_val_idx)
            n_val = max(1, int(n_tv * 0.2))
            val_idx   = train_val_idx[-n_val:]
            train_idx = train_val_idx[:-n_val]

            train_data = [data_list[i] for i in train_idx]
            val_data   = [data_list[i] for i in val_idx]
            test_data  = [data_list[i] for i in test_idx]

            if not args.quiet:
                print(f"\n--- Fold {fold_idx + 1}/5 ---")

            test_acc, _ = train_fold(
                train_data, val_data, test_data,
                num_classes=args.num_classes,
                config=config,
                device=device,
                verbose=not args.quiet,
            )
            fold_accs.append(test_acc)
            if not args.quiet:
                print(f"  Fold {fold_idx + 1} test_acc = {test_acc:.4f}")

        mean_acc = float(np.mean(fold_accs))
        std_acc  = float(np.std(fold_accs))
        print(f"\nResult: {args.dataset} | "
              f"acc = {mean_acc:.4f} +/- {std_acc:.4f}")
        result = {
            'dataset': args.dataset,
            'num_classes': args.num_classes,
            'fold_accs': fold_accs,
            'mean_acc': mean_acc,
            'std_acc': std_acc,
        }

    # Save results
    if args.result_dir:
        os.makedirs(args.result_dir, exist_ok=True)
        fname = f"{args.dataset}_nc{args.num_classes}_seed{args.seed}.json"
        with open(os.path.join(args.result_dir, fname), 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Saved results to {os.path.join(args.result_dir, fname)}")

    return result


def main():
    parser = argparse.ArgumentParser(description='TCBG Training Script')

    # Dataset
    parser.add_argument('--dataset', type=str, required=True,
                        choices=sorted(SOCIAL_DATASETS | BRAIN_DATASETS | TRAFFIC_DATASETS))
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of classes (2 or 3 for traffic)')

    # Bifiltration
    parser.add_argument('--T_grid', type=int, default=None,
                        help='Time grid resolution (default: dataset-specific)')
    parser.add_argument('--K_grid', type=int, default=None,
                        help='Curvature grid resolution (default: dataset-specific)')

    # Bifiltration (per-run overrides)
    parser.add_argument('--min_persistence', type=float, default=None,
                        help='Min bar persistence kept in Graphcode (default 0.05)')

    # GIN
    parser.add_argument('--gin_layers',  type=int,   default=3)
    parser.add_argument('--gin_hidden',  type=int,   default=64)
    parser.add_argument('--gin_dropout', type=float, default=0.3)

    # Training
    parser.add_argument('--lr',          type=float, default=0.001)
    parser.add_argument('--weight_decay',type=float, default=1e-4)
    parser.add_argument('--epochs',      type=int,   default=200)
    parser.add_argument('--patience',    type=int,   default=30)
    parser.add_argument('--batch_size',  type=int,   default=32)
    parser.add_argument('--seed',        type=int,   default=42)
    parser.add_argument('--device',      type=str,   default='cpu')

    # Paths
    parser.add_argument('--data_root',   type=str, default='./data/raw')
    parser.add_argument('--cache_dir',   type=str, default='./data/cache')
    parser.add_argument('--result_dir',  type=str, default='./results')

    parser.add_argument('--quiet', action='store_true')

    args = parser.parse_args()

    # Apply dataset-specific grid defaults if not overridden
    t_grid, k_grid = DATASET_GRID.get(args.dataset, (30, 20))
    if args.T_grid is None:
        args.T_grid = t_grid
    if args.K_grid is None:
        args.K_grid = k_grid

    run(args)


if __name__ == '__main__':
    main()
