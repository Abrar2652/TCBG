"""
experiments/run_t3former_fair.py

Runs T3Former's OWN model and code on the 5 social datasets using
the SAME evaluation protocol as our TCBG:
  - StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
  - 10 seeds (0-9) x 5 folds = 50 runs per dataset
  - Phase 1: grid search (36 combos x 3 seeds) to find best hyperparams
  - Phase 2: best hyperparams x 10 seeds x 5 folds = final result
  - Report: mean +/- std over 10 per-seed fold-means

This gives a FAIR comparison — same model, same seeds, same folds.

Usage:
  python experiments/run_t3former_fair.py --device cuda
  python experiments/run_t3former_fair.py --device cuda --datasets dblp_ct1 infectious_ct1
  python experiments/run_t3former_fair.py --device cuda --t3former_dir /path/to/T3Former-3311
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
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch

warnings.filterwarnings('ignore')

# ── Paths ────────────────────────────────────────────────────────────────────
HERE          = Path(__file__).parent.resolve()
DEFAULT_T3DIR = HERE / 'T3Former-3311'
DATA_DIR      = HERE / 'data'

RAW_DATA_PATHS = {
    'infectious_ct1': DATA_DIR / 'infectious_ct1' / 'infectious_ct1',
    'dblp_ct1':       DATA_DIR / 'dblp_ct1'       / 'dblp_ct1',
    'tumblr_ct1':     DATA_DIR / 'tumblr_ct1'      / 'tumblr_ct1',
    'mit_ct1':        DATA_DIR / 'mit_ct1'          / 'mit_ct1',
    'highschool_ct1': DATA_DIR / 'highschool_ct1'  / 'highschool_ct1',
    'facebook_ct1':   DATA_DIR / 'facebook_ct1'    / 'facebook_ct1',
}

DATASETS = ['infectious_ct1', 'dblp_ct1', 'tumblr_ct1', 'mit_ct1', 'highschool_ct1', 'facebook_ct1']

# T3Former paper reported numbers (single seed=42, KFold)
T3FORMER_PAPER = {
    'infectious_ct1': 68.50,
    'dblp_ct1':       60.90,
    'tumblr_ct1':     63.20,
    'mit_ct1':        73.16,
    'highschool_ct1': 67.20,
}

# Our TCBG results (10 seeds, StratifiedKFold)
TCBG_RESULTS = {
    'infectious_ct1': (69.55, 2.22),
    'dblp_ct1':       (93.60, 0.48),
    'tumblr_ct1':     (57.99, 1.41),
    'mit_ct1':        (70.42, 4.28),
    'highschool_ct1': (56.83, 3.99),
}

# Grid search space — identical to T3Former paper Section 4.1
LEARNING_RATES  = [0.01, 0.005, 0.001]
HIDDEN_DIMS     = [16, 32, 64, 128]
DROPOUT_RATES   = [0.5, 0.3, 0.0]
GS_SEEDS        = [0, 1, 2]          # 3 seeds for Phase 1 (speed)
FULL_SEEDS      = list(range(10))    # seeds 0-9 for Phase 2


# ── Custom dataset ────────────────────────────────────────────────────────────
class CustomGraphDataset(Dataset):
    def __init__(self, X, X1, graphs, y):
        self.X = X; self.X1 = X1
        self.graphs = graphs; self.y = y
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.X1[idx], self.graphs[idx], self.y[idx]


def custom_collate(batch):
    X_batch, X1_batch, graph_batch, y_batch = zip(*batch)
    return (
        torch.stack(X_batch),
        torch.stack(X1_batch),
        Batch.from_data_list(graph_batch),
        torch.stack(y_batch),
    )


# ── Load T3Former data for one dataset ───────────────────────────────────────
def load_dataset(dataset_name: str, t3former_dir: Path):
    """Load T3Former features using their exact data loader,
       but pointing to our TCBG raw data files."""
    # Temporarily patch the file_path_template in their data_loader
    import data_loader as t3_loader
    original_template = t3_loader.file_path_template

    data_path = RAW_DATA_PATHS[dataset_name]
    # Monkeypatch: readTUds receives the full prefix, not the template
    t3_loader.file_path_template = str(data_path.parent / data_path.stem)

    # Also patch the pickle paths to use T3Former's social_fe/
    betti_pkl = str(t3former_dir / 'social_fe' / 'sw_betti_3_2.pkl')
    dos_pkl   = str(t3former_dir / 'social_fe' / 'dos_vec_3_2.pkl')

    # Call T3Former's loader directly with the patched prefix
    from modules import readTUds, temporal_graph_from_TUds
    from torch_geometric.data import Data
    import pickle
    import networkx as nx
    from collections import defaultdict

    file_path = str(data_path)
    num_graphs, graphs_label, graphs_node, node_mapping, graphs_edge = readTUds(file_path)
    temporal_graphs, temp_edge_idx = temporal_graph_from_TUds(
        num_graphs, graphs_label, graphs_node, node_mapping, graphs_edge)

    with open(betti_pkl, 'rb') as f:
        sw_betti = pickle.load(f)
    X0 = torch.tensor(np.array(sw_betti[dataset_name]), dtype=torch.float32)

    with open(dos_pkl, 'rb') as f:
        dos_vec = pickle.load(f)
    X1 = torch.tensor(np.array(dos_vec[dataset_name]), dtype=torch.float32)

    y0 = np.array(graphs_label)

    # Build data_list with temporal degree node features (T3Former's exact method)
    def initialize_graph(tempG):
        G = nx.Graph()
        edge_time_map = defaultdict(list)
        for u, v, t in zip(tempG.edge_index[0].tolist(),
                           tempG.edge_index[1].tolist(),
                           tempG.t.tolist()):
            edge_time_map[(u, v)].append(t)
        for (u, v), times in edge_time_map.items():
            G.add_edge(u, v, time=sorted(times))
        return G

    max_t = max(g.t.max().item() for g in temporal_graphs)
    data_list = []
    for i in range(num_graphs):
        G = initialize_graph(temporal_graphs[i])
        edges_data = list(G.edges(data=True))
        nodes = sorted(G.nodes())
        node_features = []
        for node in nodes:
            degree_list = []
            for t in range(max_t):
                temp_edges = [(u, v) for u, v, attrs in edges_data if t in attrs['time']]
                tmp_g = nx.Graph()
                tmp_g.add_edges_from(temp_edges)
                degree_list.append(tmp_g.degree(node) if node in tmp_g else 0)
            node_features.append(degree_list)
        src, dst = temp_edge_idx[i]
        edge_index = torch.stack([src, dst], dim=0)
        x = torch.tensor(node_features, dtype=torch.float)
        y = torch.tensor(graphs_label[i])
        data_list.append(Data(x=x, edge_index=edge_index, y=y))

    t3_loader.file_path_template = original_template
    return data_list, X0, X1, y0


# ── One fold of T3Former training ────────────────────────────────────────────
def train_one_fold(train_loader, val_loader, sage_input_dim,
                   num_features, num_features2,
                   num_timesteps, num_timesteps2,
                   num_classes, lr, hidden_dim, dropout,
                   epochs, device):
    from model import T3Former as T3FormerModel
    model = T3FormerModel(
        sage_input_dim=sage_input_dim,
        transformer_input_dim=num_features,
        transformer2_input_dim=num_features2,
        hidden_dim=hidden_dim,
        output_dim=num_classes,
        n_heads=2,
        n_layers=2,
        num_timesteps1=num_timesteps,
        num_timesteps2=num_timesteps2,
        dropout_p=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    train_accs = []
    val_accs   = []

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        correct = 0; total = 0
        for xb, xb1, gb, yb in train_loader:
            gb = gb.to(device); xb = xb.to(device)
            xb1 = xb1.to(device); yb = yb.to(device)
            optimizer.zero_grad()
            out, _ = model(gb.x, gb.edge_index, gb.batch, xb, xb1)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            correct += (out.argmax(1) == yb).sum().item()
            total   += yb.size(0)
        train_accs.append(correct / total)

        # Validate
        model.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for xb, xb1, gb, yb in val_loader:
                gb = gb.to(device); xb = xb.to(device)
                xb1 = xb1.to(device); yb = yb.to(device)
                out, _ = model(gb.x, gb.edge_index, gb.batch, xb, xb1)
                correct += (out.argmax(1) == yb).sum().item()
                total   += yb.size(0)
        val_accs.append(correct / total)

    # T3Former's exact metric: val acc at epoch with best train acc
    best_epoch = int(np.argmax(train_accs))
    fold_acc   = val_accs[best_epoch]
    return fold_acc


# ── Run one seed x 5-fold CV ─────────────────────────────────────────────────
def run_seed(X, X1, data_list, y, labels_for_split,
             sage_input_dim, num_features, num_features2,
             num_timesteps, num_timesteps2, num_classes,
             lr, hidden_dim, dropout, epochs, seed, device,
             verbose=False):

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_accs = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), labels_for_split)):
        train_ds = CustomGraphDataset(X[train_idx], X1[train_idx],
                                      [data_list[i] for i in train_idx], y[train_idx])
        val_ds   = CustomGraphDataset(X[val_idx],   X1[val_idx],
                                      [data_list[i] for i in val_idx],   y[val_idx])
        train_ld = DataLoader(train_ds, batch_size=32, shuffle=True,  collate_fn=custom_collate)
        val_ld   = DataLoader(val_ds,   batch_size=32, shuffle=False, collate_fn=custom_collate)

        acc = train_one_fold(
            train_ld, val_ld,
            sage_input_dim, num_features, num_features2,
            num_timesteps, num_timesteps2, num_classes,
            lr, hidden_dim, dropout, epochs, device,
        )
        fold_accs.append(acc)
        if verbose:
            print(f'    fold {fold_idx+1}/5  acc={acc*100:.2f}%', flush=True)

    return float(np.mean(fold_accs))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets',     nargs='+', default=DATASETS)
    parser.add_argument('--device',       type=str,  default='cuda')
    parser.add_argument('--epochs',       type=int,  default=100,
                        help='Epochs per fold (T3Former default = 100)')
    parser.add_argument('--result_dir',   type=str,  default=str(HERE / 'results' / 't3former_fair'))
    parser.add_argument('--t3former_dir', type=str,  default=str(DEFAULT_T3DIR))
    parser.add_argument('--force',        action='store_true')
    parser.add_argument('--skip_grid',    action='store_true',
                        help='Skip Phase 1 grid search, use T3Former paper defaults')
    parser.add_argument('--default_lr',   type=float, default=0.005)
    parser.add_argument('--default_h',    type=int,   default=32)
    parser.add_argument('--default_drop', type=float, default=0.3)
    parser.add_argument('--seeds',        type=int,   default=10)
    args = parser.parse_args()
    global FULL_SEEDS
    FULL_SEEDS = list(range(args.seeds))

    # Add T3Former to path
    t3_dir = Path(args.t3former_dir)
    if not t3_dir.exists():
        print(f'ERROR: T3Former dir not found: {t3_dir}')
        print('Pass --t3former_dir /path/to/T3Former-3311')
        sys.exit(1)
    sys.path.insert(0, str(t3_dir))

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.result_dir, exist_ok=True)

    print(f'T3Former Fair Eval | device={device} | epochs={args.epochs}')
    print(f'Protocol: StratifiedKFold(5) x {len(FULL_SEEDS)} seeds = '
          f'{5*len(FULL_SEEDS)} runs per dataset\n')

    all_results = {}

    for dataset_name in args.datasets:
        out_file = os.path.join(args.result_dir, f'{dataset_name}_fair.json')
        if os.path.exists(out_file) and not args.force:
            print(f'[skip] {dataset_name} — already done')
            all_results[dataset_name] = json.load(open(out_file))
            continue

        print(f"\n{'='*60}  {dataset_name.upper()}")
        print('Loading data...', flush=True)
        data_list, X0, X1, y0 = load_dataset(dataset_name, t3_dir)
        print(f'  {len(data_list)} graphs loaded', flush=True)

        X    = torch.tensor(X0, dtype=torch.float32) if not isinstance(X0, torch.Tensor) else X0
        X1t  = torch.tensor(X1, dtype=torch.float32) if not isinstance(X1, torch.Tensor) else X1
        y    = torch.tensor(y0, dtype=torch.long)
        labels_np = y0.astype(int)

        num_classes    = len(np.unique(y0))
        sage_input_dim = data_list[0].x.shape[-1]
        num_timesteps  = X.shape[1]
        num_timesteps2 = X1t.shape[1]
        num_features   = X.shape[2]
        num_features2  = X1t.shape[2]

        print(f'  classes={num_classes}  sage_dim={sage_input_dim}  '
              f'betti_shape={tuple(X.shape)}  dos_shape={tuple(X1t.shape)}', flush=True)

        # ── Phase 1: Grid search (3 seeds x 36 combos) ───────────────────
        print('\n--- Phase 1: Grid Search (3 seeds x 36 combos) ---', flush=True)
        gs_file = os.path.join(args.result_dir, f'{dataset_name}_gs.json')

        if args.skip_grid:
            best_lr, best_hidden, best_dropout = args.default_lr, args.default_h, args.default_drop
            print(f'  [skip grid] using defaults: lr={best_lr} h={best_hidden} drop={best_dropout}')
        elif os.path.exists(gs_file) and not args.force:
            gs_res = json.load(open(gs_file))
            best_lr      = gs_res['best_lr']
            best_hidden  = gs_res['best_hidden']
            best_dropout = gs_res['best_dropout']
            print(f'  [cached] best: lr={best_lr} h={best_hidden} drop={best_dropout}')
        else:
            best_val    = 0.0
            best_lr     = 0.001
            best_hidden = 64
            best_dropout = 0.3
            gs_log = []

            for lr in LEARNING_RATES:
                for hidden_dim in HIDDEN_DIMS:
                    for dropout in DROPOUT_RATES:
                        combo_means = []
                        for seed in GS_SEEDS:
                            m = run_seed(
                                X, X1t, data_list, y, labels_np,
                                sage_input_dim, num_features, num_features2,
                                num_timesteps, num_timesteps2, num_classes,
                                lr, hidden_dim, dropout, args.epochs, seed, device,
                                verbose=False,
                            )
                            combo_means.append(m)
                        combo_mean = float(np.mean(combo_means))
                        gs_log.append({'lr': lr, 'hidden': hidden_dim,
                                       'dropout': dropout, 'mean': combo_mean})
                        print(f'  lr={lr} h={hidden_dim} drop={dropout}  '
                              f'mean={combo_mean*100:.2f}%', flush=True)
                        if combo_mean > best_val:
                            best_val = combo_mean
                            best_lr = lr; best_hidden = hidden_dim; best_dropout = dropout

            print(f'\n  BEST: lr={best_lr} h={best_hidden} drop={best_dropout}  '
                  f'val={best_val*100:.2f}%')
            json.dump({'best_lr': best_lr, 'best_hidden': best_hidden,
                       'best_dropout': best_dropout, 'best_val': best_val,
                       'log': gs_log}, open(gs_file, 'w'), indent=2)

        # ── Phase 2: Full eval (10 seeds x 5 folds) ──────────────────────
        print(f'\n--- Phase 2: Full Eval (10 seeds x 5 folds) ---', flush=True)
        print(f'  Config: lr={best_lr}  h={best_hidden}  drop={best_dropout}', flush=True)

        seed_means = []
        for seed in FULL_SEEDS:
            print(f'\n  seed={seed}:', flush=True)
            m = run_seed(
                X, X1t, data_list, y, labels_np,
                sage_input_dim, num_features, num_features2,
                num_timesteps, num_timesteps2, num_classes,
                best_lr, best_hidden, best_dropout,
                args.epochs, seed, device, verbose=True,
            )
            seed_means.append(m)
            print(f'  --> seed={seed} mean: {m*100:.2f}%', flush=True)

        mean_acc = float(np.mean(seed_means))
        std_acc  = float(np.std(seed_means))

        t3paper = T3FORMER_PAPER.get(dataset_name, 0)
        tcbg_m, tcbg_s = TCBG_RESULTS.get(dataset_name, (0, 0))

        result = {
            'dataset':      dataset_name,
            'protocol':     'StratifiedKFold(5) x 10 seeds',
            'seeds':        FULL_SEEDS,
            'seed_means':   seed_means,
            'mean_acc':     mean_acc,
            'std_acc':      std_acc,
            'best_lr':      best_lr,
            'best_hidden':  best_hidden,
            'best_dropout': best_dropout,
            'epochs':       args.epochs,
            't3former_paper': t3paper,
            'tcbg_mean':    tcbg_m,
            'tcbg_std':     tcbg_s,
        }
        all_results[dataset_name] = result
        json.dump(result, open(out_file, 'w'), indent=2)

        print(f'\n  T3Former Fair:  {mean_acc*100:.2f}% +/- {std_acc*100:.2f}%')
        print(f'  T3Former Paper: {t3paper:.2f}%  (single seed=42)')
        print(f'  TCBG (ours):    {tcbg_m:.2f}% +/- {tcbg_s:.2f}%')

    # ── Final comparison table ────────────────────────────────────────────
    print(f"\n\n{'='*75}")
    print(f"{'FAIR COMPARISON  (same protocol: StratifiedKFold x 10 seeds)':^75}")
    print(f"{'='*75}")
    print(f"{'Dataset':15s}  {'T3Former (fair)':>20s}  {'T3Former (paper)':>18s}  {'TCBG (ours)':>18s}")
    print('-' * 75)
    for ds in DATASETS:
        if ds in all_results:
            r   = all_results[ds]
            m   = r['mean_acc'] * 100
            s   = r['std_acc']  * 100
            t3p = r['t3former_paper']
            tcm, tcs = r['tcbg_mean'], r['tcbg_std']
            winner = 'TCBG' if tcm > m else 'T3F '
            print(f'{ds:15s}  {m:7.2f}% +/- {s:4.2f}%  '
                  f'{t3p:>18.2f}%  '
                  f'{tcm:>7.2f}% +/- {tcs:.2f}%  <- {winner}')
    print('=' * 75)
    print('\nNote: "T3Former (fair)" = T3Former model, our 10-seed protocol')
    print('      "T3Former (paper)" = T3Former paper reported (single seed=42)')

    summary_file = os.path.join(args.result_dir, 'fair_comparison.json')
    json.dump(all_results, open(summary_file, 'w'), indent=2)
    print(f'\nSaved -> {summary_file}')


if __name__ == '__main__':
    main()
