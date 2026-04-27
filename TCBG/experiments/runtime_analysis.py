"""
experiments/runtime_analysis.py
Measure wall-clock training and preprocessing runtime for TCBG.
Matches T3Former Table 7 format: runtime per dataset in seconds.

Reports:
  - Preprocessing time (Graphcode computation): total + per graph
  - Training time: total for 200 epochs (5-fold, 1 seed)
  - Inference time: per graph (ms)
  - Trainable parameters

Usage:
  python experiments/runtime_analysis.py --device cuda
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
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gin_classifier import build_gin
from src.pipeline import TCBGPipeline
from data.social_loader import load_social_dataset, get_social_folds

warnings.filterwarnings('ignore')

DATASETS    = ['infectious', 'dblp', 'tumblr', 'mit', 'highschool']
NUM_CLASSES = {'infectious': 2, 'dblp': 2, 'tumblr': 2, 'mit': 2, 'highschool': 2}


def set_seed(seed=0):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def time_preprocessing(dataset, data_root):
    """Time Graphcode computation from raw data (no cache)."""
    config = {
        'T_grid': 30, 'K_grid': 20, 'epsilon': 'auto', 'hom_dim': [0, 1],
        'min_persistence': 0.05, 'max_bars_per_level': 20,
        'node_feat_dim': 8, 'global_feat_dim': 21,
        'dataset_type': 'social', 'num_timesteps': None,
    }
    pipeline = TCBGPipeline(config)
    graphs, nodes, labels = load_social_dataset(dataset, root=data_root, verbose=False)
    n = len(graphs)
    t0 = time.time()
    _ = pipeline.process_dataset(list(zip(graphs, nodes, labels)), verbose=False)
    return time.time() - t0, n


def time_training(dataset, cache_dir, device, nc, gs_dir):
    """Time one complete training run (1 seed, 5-fold, up to 200 epochs each fold)."""
    cache_key  = f"{dataset}_nc{nc}_T30_K20_mp0.05_mb20_nf8_gf21_epssocial"
    cache_file = os.path.join(cache_dir, f"{cache_key}.pt")
    if not os.path.exists(cache_file):
        return None

    saved = torch.load(cache_file, weights_only=False)
    data_list, labels = saved['data_list'], saved['labels']

    gs_file = os.path.join(gs_dir, f"{dataset}_gs_result.json")
    if os.path.exists(gs_file):
        d = json.load(open(gs_file))
        hidden, dropout, lr = d['best_hidden'], d['best_dropout'], d['best_lr']
    else:
        hidden, dropout, lr = 64, 0.3, 0.001

    config = {
        'gin_layers': 3, 'gin_hidden': hidden, 'gin_dropout': dropout,
        'node_feat_dim': 8, 'global_feat_dim': 21, 'use_jk': True,
    }
    n_params = count_params(build_gin(config, nc))

    set_seed(0)
    folds = get_social_folds(labels, n_splits=5, seed=0)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    total_train_time = 0.0
    fold_epoch_counts = []

    for tv_idx, te_idx in folds:
        n_val     = max(1, int(len(tv_idx) * 0.2))
        tr_data   = [data_list[i] for i in tv_idx[:-n_val]]
        va_data   = [data_list[i] for i in tv_idx[-n_val:]]
        te_data   = [data_list[i] for i in te_idx]

        tr_l = DataLoader(tr_data, batch_size=32, shuffle=True)
        va_l = DataLoader(va_data, batch_size=32)

        model = build_gin(config, nc).to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', patience=10, factor=0.5)

        best_val = float('inf'); pat = 0; n_epochs = 0
        t_fold = time.time()

        for epoch in range(200):
            model.train()
            for batch in tr_l:
                batch = batch.to(device)
                opt.zero_grad()
                logits = model(batch)
                loss = criterion(logits, batch.y.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            model.eval()
            vl = 0; vt = 0
            with torch.no_grad():
                for batch in va_l:
                    batch = batch.to(device)
                    loss = criterion(model(batch), batch.y.view(-1))
                    vl += float(loss.item()) * batch.num_graphs
                    vt += batch.num_graphs
            vl /= max(vt, 1)
            sched.step(vl)
            n_epochs += 1
            if vl < best_val:
                best_val = vl; pat = 0
            else:
                pat += 1
            if pat >= 30:
                break

        total_train_time += time.time() - t_fold
        fold_epoch_counts.append(n_epochs)

    # Inference time over full dataset
    inf_loader = DataLoader(data_list, batch_size=32, shuffle=False)
    t0 = time.time()
    model.eval()
    with torch.no_grad():
        for batch in inf_loader:
            batch = batch.to(device)
            _ = model(batch)
    inf_time_ms = (time.time() - t0) / len(data_list) * 1000

    peak_mb = 0
    if device.type == 'cuda':
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

    return {
        'n_graphs':           len(data_list),
        'n_params':           n_params,
        'train_total_s':      round(total_train_time, 2),
        'train_per_fold_s':   round(total_train_time / 5, 2),
        'avg_epochs':         round(float(np.mean(fold_epoch_counts)), 1),
        'inference_ms_graph': round(inf_time_ms, 4),
        'peak_gpu_mb':        round(peak_mb, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets',          nargs='+', default=DATASETS)
    parser.add_argument('--device',            type=str, default='cuda')
    parser.add_argument('--data_root',         type=str, default='./data/raw')
    parser.add_argument('--cache_dir',         type=str, default='./data/cache')
    parser.add_argument('--result_dir',        type=str, default='./results/complexity')
    parser.add_argument('--gs_dir',            type=str, default='./results_gs')
    parser.add_argument('--skip_preprocessing',action='store_true',
                        help='Skip Graphcode recomputation timing')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.result_dir, exist_ok=True)
    print(f"Device: {device}\n")

    results = {}

    for dataset in args.datasets:
        nc = NUM_CLASSES[dataset]
        print(f"--- {dataset.upper()} ---", flush=True)
        entry = {'dataset': dataset, 'num_classes': nc}

        if not args.skip_preprocessing:
            t, n = time_preprocessing(dataset, args.data_root)
            entry['preprocess_total_s']    = round(t, 2)
            entry['preprocess_per_graph_s']= round(t / n, 5)
            print(f"  Preprocessing: {t:.1f}s total ({t/n:.4f}s/graph)", flush=True)

        timing = time_training(dataset, args.cache_dir, device, nc, args.gs_dir)
        if timing:
            entry.update(timing)
            print(f"  Params:        {timing['n_params']:,}")
            print(f"  Train (5-fold):{timing['train_total_s']:.1f}s  "
                  f"({timing['avg_epochs']:.0f} avg epochs/fold)")
            print(f"  Inference:     {timing['inference_ms_graph']:.4f} ms/graph")
            print(f"  Peak GPU:      {timing['peak_gpu_mb']:.1f} MB", flush=True)

        results[dataset] = entry

    # Save JSON
    out_file = os.path.join(args.result_dir, 'runtime_results.json')
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print table (matches T3Former Table 7 style)
    print(f"\n\n{'='*85}")
    print(f"{'RUNTIME TABLE (matches T3Former Table 7 format)':^85}")
    print(f"{'='*85}")
    print(f"{'Dataset':12s} {'Graphs':>8s} {'Params':>10s} "
          f"{'Preproc(s)':>12s} {'Train(s)':>10s} "
          f"{'Infer(ms/g)':>13s} {'GPU(MB)':>9s}")
    print('-' * 85)
    for ds, r in results.items():
        g   = str(r.get('n_graphs', '-'))
        p   = f"{r.get('n_params', 0):,}" if 'n_params' in r else '-'
        pp  = f"{r.get('preprocess_total_s', 0):.1f}" if 'preprocess_total_s' in r else '-'
        tr  = f"{r.get('train_total_s', 0):.1f}"      if 'train_total_s'      in r else '-'
        inf = f"{r.get('inference_ms_graph', 0):.4f}" if 'inference_ms_graph' in r else '-'
        mem = f"{r.get('peak_gpu_mb', 0):.1f}"        if 'peak_gpu_mb'        in r else '-'
        print(f"{ds:12s} {g:>8s} {p:>10s} {pp:>12s} {tr:>10s} {inf:>13s} {mem:>9s}")

    print(f"\nSaved to {out_file}")


if __name__ == '__main__':
    main()
