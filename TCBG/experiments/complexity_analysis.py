"""
experiments/complexity_analysis.py  — COMPREHENSIVE VERSION
Runtime and memory complexity analysis for TCBG.

Reports (per dataset):
  Preprocessing breakdown
    1. Curvature computation time         (s/graph)
    2. Bifiltration construction time     (s/graph)
    3. Graphcode computation time         (s/graph)
    4. Total preprocessing time           (s total + s/graph)

  Training & inference
    5. Training time per epoch            (ms)
    6. Total training time (5-fold)       (s)
    7. Inference time                     (ms/graph)
    8. Peak GPU memory                    (MB)

  Model
    9.  Trainable parameters              (count)
    10. GIN layers / hidden dim

  Scalability
    11. Runtime vs #graphs trend          (measured sub-sampling)

Usage:
  python experiments/complexity_analysis.py --device cuda
  python experiments/complexity_analysis.py --device cuda --skip_preprocessing
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
from src.curvature import compute_forman_ricci, auto_epsilon
from src.bifiltration import build_bifiltration
from src.graphcode import compute_graphcode
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


# ------------------------------------------------------------------
# Stage-by-stage preprocessing timing
# ------------------------------------------------------------------

def time_preprocessing_stages(graphs, nodes, n_sample: int = 50):
    """
    Time each pipeline stage on a random sample of n_sample graphs.
    Returns dict of per-graph timing for each stage.
    """
    idx = np.random.choice(len(graphs), min(n_sample, len(graphs)), replace=False)
    sample_graphs = [graphs[i] for i in idx]
    sample_nodes  = [nodes[i]  for i in idx]

    t_curv = t_bflt = t_gc = 0.0
    n_valid = 0

    for g, nd in zip(sample_graphs, sample_nodes):
        if not g:
            continue

        # Stage 1: curvature
        ts = sorted(set(t for _, _, t in g))
        eps = auto_epsilon(ts, dataset_type='social') if len(ts) > 1 else 1.0
        t0 = time.perf_counter()
        edge_curv = compute_forman_ricci(g, eps)
        t_curv += time.perf_counter() - t0

        if not edge_curv:
            continue

        # Stage 2: bifiltration
        t0 = time.perf_counter()
        bifilt = build_bifiltration(edge_curv, T_grid=30, K_grid=20)
        t_bflt += time.perf_counter() - t0

        # Stage 3: graphcode
        t0 = time.perf_counter()
        _ = compute_graphcode(bifilt, T_grid=30, K_grid=20, hom_dim=[0, 1],
                              min_persistence=0.05, max_bars=20)
        t_gc += time.perf_counter() - t0
        n_valid += 1

    if n_valid == 0:
        return None
    return {
        'curvature_s_per_graph':     t_curv / n_valid,
        'bifiltration_s_per_graph':  t_bflt / n_valid,
        'graphcode_s_per_graph':     t_gc   / n_valid,
        'total_preproc_s_per_graph': (t_curv + t_bflt + t_gc) / n_valid,
        'n_sampled': n_valid,
    }


def time_full_preprocessing(dataset, data_root):
    """Time full pipeline over all graphs in dataset."""
    from src.pipeline import TCBGPipeline
    config = {
        'T_grid': 30, 'K_grid': 20, 'epsilon': 'auto', 'hom_dim': [0, 1],
        'min_persistence': 0.05, 'max_bars_per_level': 20,
        'node_feat_dim': 8, 'global_feat_dim': 21,
        'dataset_type': 'social', 'num_timesteps': None,
    }
    pipeline = TCBGPipeline(config)
    graphs, nodes, labels = load_social_dataset(
        dataset, root=data_root, verbose=False)
    n = len(graphs)
    t0 = time.perf_counter()
    _ = pipeline.process_dataset(list(zip(graphs, nodes, labels)), verbose=False)
    elapsed = time.perf_counter() - t0
    return elapsed, n, elapsed / n


# ------------------------------------------------------------------
# Training & inference timing
# ------------------------------------------------------------------

def time_training_and_inference(dataset, cache_dir, device, nc, gs_dir,
                                 n_timing_epochs: int = 10):
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
    model   = build_gin(config, num_classes=nc).to(device)
    n_params = count_params(model)

    set_seed(0)
    folds = get_social_folds(labels, n_splits=5, seed=0)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    # ── Per-epoch training time ──────────────────────────────────
    tv_idx, _ = folds[0]
    tr_data = [data_list[i] for i in tv_idx[:int(len(tv_idx)*0.8)]]
    tr_l = DataLoader(tr_data, batch_size=32, shuffle=False)
    opt  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    epoch_times = []
    model.train()
    for ep in range(n_timing_epochs + 2):        # 2 warm-up
        t0 = time.perf_counter()
        for batch in tr_l:
            batch = batch.to(device)
            opt.zero_grad()
            logits = model(batch)
            loss = criterion(logits, batch.y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        elapsed = time.perf_counter() - t0
        if ep >= 2:
            epoch_times.append(elapsed)

    # ── Full 5-fold training time ────────────────────────────────
    full_train_time = 0.0
    fold_epochs = []
    for tv_idx, te_idx in folds:
        nv = max(1, int(len(tv_idx) * 0.2))
        tr = [data_list[i] for i in tv_idx[:-nv]]
        va = [data_list[i] for i in tv_idx[-nv:]]

        mod2 = build_gin(config, nc).to(device)
        op2  = torch.optim.Adam(mod2.parameters(), lr=lr, weight_decay=1e-4)
        sc2  = torch.optim.lr_scheduler.ReduceLROnPlateau(op2, mode='min', patience=10, factor=0.5)
        tr_l2 = DataLoader(tr, batch_size=32, shuffle=True)
        va_l2 = DataLoader(va, batch_size=32)
        best_v = float('inf'); pat = 0; ep_cnt = 0
        t0 = time.perf_counter()
        for _ in range(200):
            mod2.train()
            for batch in tr_l2:
                batch = batch.to(device)
                op2.zero_grad()
                loss = criterion(mod2(batch), batch.y.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(mod2.parameters(), 1.0)
                op2.step()
            mod2.eval()
            vl = 0; vt = 0
            with torch.no_grad():
                for batch in va_l2:
                    batch = batch.to(device)
                    vl += float(criterion(mod2(batch), batch.y.view(-1)).item()) * batch.num_graphs
                    vt += batch.num_graphs
            vl /= max(vt, 1)
            sc2.step(vl)
            ep_cnt += 1
            if vl < best_v:
                best_v = vl; pat = 0
            else:
                pat += 1
            if pat >= 30:
                break
        full_train_time += time.perf_counter() - t0
        fold_epochs.append(ep_cnt)

    # ── Inference time ──────────────────────────────────────────
    inf_loader = DataLoader(data_list, batch_size=32, shuffle=False)
    t0 = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for batch in inf_loader:
            batch = batch.to(device)
            _ = model(batch)
    inf_ms = (time.perf_counter() - t0) / len(data_list) * 1000

    peak_mb = 0
    if device.type == 'cuda':
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

    return {
        'n_graphs':              len(data_list),
        'n_params':              n_params,
        'gin_hidden':            hidden,
        'epoch_time_ms_mean':    float(np.mean(epoch_times)) * 1000,
        'epoch_time_ms_std':     float(np.std(epoch_times)) * 1000,
        'full_train_5fold_s':    round(full_train_time, 2),
        'avg_epochs_per_fold':   round(float(np.mean(fold_epochs)), 1),
        'inference_ms_per_graph':round(inf_ms, 4),
        'peak_gpu_mb':           round(peak_mb, 1),
    }


# ------------------------------------------------------------------
# Scalability: runtime vs dataset size
# ------------------------------------------------------------------

def scalability_analysis(dataset, cache_dir, device, nc, gs_dir,
                          fractions=(0.25, 0.5, 0.75, 1.0)):
    cache_key  = f"{dataset}_nc{nc}_T30_K20_mp0.05_mb20_nf8_gf21_epssocial"
    cache_file = os.path.join(cache_dir, f"{cache_key}.pt")
    if not os.path.exists(cache_file):
        return None

    saved = torch.load(cache_file, weights_only=False)
    data_list = saved['data_list']
    n_total = len(data_list)

    gs_file = os.path.join(gs_dir, f"{dataset}_gs_result.json")
    if os.path.exists(gs_file):
        d = json.load(open(gs_file))
        hidden, dropout, lr = d['best_hidden'], d['best_dropout'], d['best_lr']
    else:
        hidden, dropout, lr = 64, 0.3, 0.001

    config = {'gin_layers': 3, 'gin_hidden': hidden, 'gin_dropout': dropout,
              'node_feat_dim': 8, 'global_feat_dim': 21, 'use_jk': True}
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    result = []

    for frac in fractions:
        n = max(32, int(n_total * frac))
        subset = data_list[:n]
        loader = DataLoader(subset, batch_size=32, shuffle=False, drop_last=True)
        model  = build_gin(config, nc).to(device)
        opt    = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        # Time 5 training epochs
        epoch_times = []
        model.train()
        for ep in range(7):
            t0 = time.perf_counter()
            for batch in loader:
                batch = batch.to(device)
                opt.zero_grad()
                loss = crit(model(batch), batch.y.view(-1))
                loss.backward()
                opt.step()
            if ep >= 2:
                epoch_times.append(time.perf_counter() - t0)
        result.append({
            'n_graphs': n,
            'fraction': frac,
            'epoch_ms': round(float(np.mean(epoch_times)) * 1000, 2),
        })

    return result


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets',           nargs='+', default=DATASETS)
    parser.add_argument('--device',             type=str, default='cuda')
    parser.add_argument('--data_root',          type=str, default='./data/raw')
    parser.add_argument('--cache_dir',          type=str, default='./data/cache')
    parser.add_argument('--result_dir',         type=str, default='./results/complexity')
    parser.add_argument('--gs_dir',             type=str, default='./results_gs')
    parser.add_argument('--skip_preprocessing', action='store_true')
    parser.add_argument('--skip_scalability',   action='store_true')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.result_dir, exist_ok=True)
    print(f"Device: {device}\n")

    results = {}

    for dataset in args.datasets:
        nc = NUM_CLASSES[dataset]
        print(f"\n{'─'*50}  {dataset.upper()}  {'─'*50}")
        entry = {'dataset': dataset}

        # ── Preprocessing ─────────────────────────────────────────
        if not args.skip_preprocessing:
            graphs, nodes, _ = load_social_dataset(
                dataset, root=args.data_root, verbose=False)

            print(f"  [stage timing on sample]")
            stages = time_preprocessing_stages(graphs, nodes, n_sample=50)
            if stages:
                for k, v in stages.items():
                    if k != 'n_sampled':
                        print(f"    {k:35s}: {v:.5f}s")
                entry.update(stages)

            print(f"  [full preprocessing]")
            total_t, n_g, per_g = time_full_preprocessing(dataset, args.data_root)
            entry['preprocess_total_s']    = round(total_t, 2)
            entry['preprocess_per_graph_s']= round(per_g, 5)
            print(f"    Total: {total_t:.1f}s  Per graph: {per_g:.5f}s  "
                  f"({n_g} graphs)", flush=True)

        # ── Training & inference ──────────────────────────────────
        print(f"  [training & inference timing]")
        timing = time_training_and_inference(
            dataset, args.cache_dir, device, nc, args.gs_dir)
        if timing:
            entry.update(timing)
            print(f"    Params:            {timing['n_params']:,}")
            print(f"    Epoch time:        {timing['epoch_time_ms_mean']:.1f}"
                  f" ± {timing['epoch_time_ms_std']:.1f} ms")
            print(f"    Full 5-fold train: {timing['full_train_5fold_s']:.1f}s"
                  f"  (avg {timing['avg_epochs_per_fold']:.0f} ep/fold)")
            print(f"    Inference:         {timing['inference_ms_per_graph']:.4f} ms/graph")
            print(f"    Peak GPU:          {timing['peak_gpu_mb']:.1f} MB", flush=True)

        # ── Scalability ───────────────────────────────────────────
        if not args.skip_scalability:
            print(f"  [scalability analysis]")
            scale = scalability_analysis(
                dataset, args.cache_dir, device, nc, args.gs_dir)
            if scale:
                entry['scalability'] = scale
                for s in scale:
                    print(f"    n={s['n_graphs']:4d} ({s['fraction']*100:.0f}%)"
                          f"  epoch={s['epoch_ms']:.1f}ms")

        results[dataset] = entry

    # Save
    out_file = os.path.join(args.result_dir, 'complexity_results.json')
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Summary table
    print(f"\n\n{'='*100}")
    print(f"{'COMPLEXITY SUMMARY':^100}")
    print(f"{'='*100}")
    print(f"{'Dataset':12s} {'Graphs':>8s} {'Params':>10s} {'Preproc(s/g)':>14s} "
          f"{'Ep(ms)':>8s} {'5-fold(s)':>10s} {'Infer(ms/g)':>13s} {'GPU(MB)':>9s}")
    print('-' * 100)
    for ds, r in results.items():
        g   = str(r.get('n_graphs', '-'))
        p   = f"{r.get('n_params', 0):,}"      if 'n_params' in r else '-'
        pp  = f"{r.get('preprocess_per_graph_s', 0):.5f}" if 'preprocess_per_graph_s' in r else '-'
        ep  = f"{r.get('epoch_time_ms_mean', 0):.1f}"     if 'epoch_time_ms_mean' in r else '-'
        tr  = f"{r.get('full_train_5fold_s', 0):.1f}"     if 'full_train_5fold_s' in r else '-'
        inf = f"{r.get('inference_ms_per_graph', 0):.4f}" if 'inference_ms_per_graph' in r else '-'
        mem = f"{r.get('peak_gpu_mb', 0):.1f}"            if 'peak_gpu_mb' in r else '-'
        print(f"{ds:12s} {g:>8s} {p:>10s} {pp:>14s} {ep:>8s} {tr:>10s} {inf:>13s} {mem:>9s}")

    print(f"\nSaved → {out_file}")


if __name__ == '__main__':
    main()
