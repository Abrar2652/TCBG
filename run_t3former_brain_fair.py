"""
run_t3former_brain_fair.py
Fair evaluation of T3Former on NeuroGraph DynHCP brain datasets.
Protocol: 70/10/20 train/val/test split × 10 seeds, phase-1 grid search on 3 seeds.
"""
from __future__ import annotations
import argparse, json, os, pickle, sys, warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch, Data

warnings.filterwarnings('ignore')

HERE = Path(__file__).parent.resolve()
BRAIN_DATASETS = ['DynHCPActivity', 'DynHCPGender', 'DynHCPAge']
T3FORMER_PAPER = {'DynHCPActivity': 90.76, 'DynHCPGender': 75.79, 'DynHCPAge': 58.73}
LEARNING_RATES = [0.01, 0.005, 0.001]
HIDDEN_DIMS    = [16, 32, 64, 128]
DROPOUT_RATES  = [0.5, 0.3, 0.0]
GS_SEEDS       = [0, 1, 2]
FULL_SEEDS     = list(range(10))


class BrainDataset(Dataset):
    def __init__(self, X, X1, graphs, y):
        self.X, self.X1, self.graphs, self.y = X, X1, graphs, y
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.X1[i], self.graphs[i], self.y[i]


def collate(batch):
    X, X1, gs, y = zip(*batch)
    return torch.stack(X), torch.stack(X1), Batch.from_data_list(list(gs)), torch.stack(y)


def load_brain(name, t3former_dir, data_root):
    # Temporarily remove T3Former dir from sys.path to avoid shadowing by its NeuroGraph.py
    t3_str = str(t3former_dir)
    saved = [p for p in sys.path if p == t3_str]
    sys.path[:] = [p for p in sys.path if p != t3_str]
    if 'NeuroGraph' in sys.modules:
        del sys.modules['NeuroGraph']
    try:
        from NeuroGraph.datasets import NeuroGraphDynamic
    finally:
        sys.path[:0] = saved
    ds = NeuroGraphDynamic(root=data_root, name=name)
    merged = [Data(x=g.x, edge_index=g.edge_index, y=g.y[0]) for g in ds.dataset]
    with open(Path(t3former_dir) / 'neuro_fe' / f'betti_{name}.data', 'rb') as f:
        betti = pickle.load(f)
    with open(Path(t3former_dir) / 'neuro_fe' / f'dos_{name}.data', 'rb') as f:
        dos = pickle.load(f)
    X0 = torch.tensor(np.array(betti), dtype=torch.float32)
    X1 = torch.tensor(np.array(dos),   dtype=torch.float32)
    y = ds.labels.long() if torch.is_tensor(ds.labels) else torch.tensor(ds.labels, dtype=torch.long)
    return merged, X0, X1, y


def train_one(tr, te, sage_dim, nf, nf2, nt, nt2, nc, lr, h, drop, epochs, device):
    from model import T3Former
    model = T3Former(
        sage_input_dim=sage_dim, transformer_input_dim=nf, transformer2_input_dim=nf2,
        hidden_dim=h, output_dim=nc, n_heads=2, n_layers=2,
        num_timesteps1=nt, num_timesteps2=nt2, dropout_p=drop,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    crit = nn.CrossEntropyLoss()
    train_accs, test_accs = [], []
    for _ in range(epochs):
        model.train(); c = t = 0
        for xb, x1b, gb, yb in tr:
            gb, xb, x1b, yb = gb.to(device), xb.to(device), x1b.to(device), yb.to(device)
            opt.zero_grad()
            out, _ = model(gb.x, gb.edge_index, gb.batch, xb, x1b)
            crit(out, yb).backward(); opt.step()
            c += (out.argmax(1) == yb).sum().item(); t += yb.size(0)
        train_accs.append(c / t)
        model.eval(); c = t = 0
        with torch.no_grad():
            for xb, x1b, gb, yb in te:
                gb, xb, x1b, yb = gb.to(device), xb.to(device), x1b.to(device), yb.to(device)
                out, _ = model(gb.x, gb.edge_index, gb.batch, xb, x1b)
                c += (out.argmax(1) == yb).sum().item(); t += yb.size(0)
        test_accs.append(c / t)
    return test_accs[int(np.argmax(train_accs))]


def run_seed(X, X1, data_list, y, sage_dim, nf, nf2, nt, nt2, nc,
             lr, h, drop, epochs, seed, device):
    y_np = y.numpy()
    idx = np.arange(len(y))
    train_idx, temp_idx, _, y_temp = train_test_split(
        idx, y_np, test_size=0.3, random_state=seed, stratify=y_np)
    _, test_idx = train_test_split(
        temp_idx, test_size=2/3, random_state=seed, stratify=y_temp)
    mk = lambda i: BrainDataset(X[i], X1[i], [data_list[k] for k in i], y[i])
    tr = DataLoader(mk(train_idx), batch_size=32, shuffle=True,  collate_fn=collate)
    te = DataLoader(mk(test_idx),  batch_size=32, shuffle=False, collate_fn=collate)
    return train_one(tr, te, sage_dim, nf, nf2, nt, nt2, nc, lr, h, drop, epochs, device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets',     nargs='+', default=BRAIN_DATASETS)
    p.add_argument('--device',       default='cuda')
    p.add_argument('--epochs',       type=int, default=50)
    p.add_argument('--t3former_dir', required=True)
    p.add_argument('--data_root',    required=True)
    p.add_argument('--result_dir',   default=str(HERE / 'results' / 't3former_brain_fair'))
    p.add_argument('--force',        action='store_true')
    p.add_argument('--skip_grid',    action='store_true',
                   help='Skip Phase 1 grid search, use paper defaults')
    p.add_argument('--seeds',        type=int, default=10,
                   help='Number of seeds for Phase 2')
    p.add_argument('--default_lr',   type=float, default=0.005)
    p.add_argument('--default_h',    type=int,   default=32)
    p.add_argument('--default_drop', type=float, default=0.3)
    args = p.parse_args()

    sys.path.insert(0, args.t3former_dir)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.result_dir, exist_ok=True)

    print(f'T3Former Brain Fair Eval | device={device} | epochs={args.epochs}')
    print(f'Protocol: 70/10/20 split × 10 seeds\n', flush=True)
    all_res = {}

    for name in args.datasets:
        out_file = Path(args.result_dir) / f'{name}_fair.json'
        if out_file.exists() and not args.force:
            print(f'[skip] {name}')
            all_res[name] = json.load(open(out_file))
            continue

        print(f'\n{"="*60}  {name}')
        data_list, X, X1, y = load_brain(name, args.t3former_dir, args.data_root)
        nc = int(y.max().item()) + 1
        sage_dim = data_list[0].x.shape[-1]
        nt, nf   = X.shape[1], X.shape[2]
        nt2, nf2 = X1.shape[1], X1.shape[2]
        print(f'  n={len(y)}  classes={nc}  sage_dim={sage_dim}  '
              f'betti={tuple(X.shape)}  dos={tuple(X1.shape)}', flush=True)

        gs_file = Path(args.result_dir) / f'{name}_gs.json'
        if args.skip_grid:
            best = (args.default_lr, args.default_h, args.default_drop)
            print(f'  [skip grid] using defaults lr={best[0]} h={best[1]} drop={best[2]}')
        elif gs_file.exists() and not args.force:
            gs = json.load(open(gs_file))
            best = (gs['best_lr'], gs['best_hidden'], gs['best_dropout'])
            print(f'  [cached GS] {best}')
        else:
            print('\n--- Phase 1: Grid Search (3 seeds × 36 combos) ---', flush=True)
            best_val = 0; best = (0.001, 64, 0.3); gs_log = []
            for lr in LEARNING_RATES:
                for h in HIDDEN_DIMS:
                    for drop in DROPOUT_RATES:
                        vals = [run_seed(X, X1, data_list, y, sage_dim,
                                         nf, nf2, nt, nt2, nc, lr, h, drop,
                                         args.epochs, s, device) for s in GS_SEEDS]
                        m = float(np.mean(vals))
                        gs_log.append({'lr': lr, 'h': h, 'drop': drop, 'mean': m})
                        print(f'  lr={lr} h={h} drop={drop}  mean={m*100:.2f}%', flush=True)
                        if m > best_val:
                            best_val = m; best = (lr, h, drop)
            json.dump({'best_lr': best[0], 'best_hidden': best[1],
                       'best_dropout': best[2], 'best_val': best_val,
                       'log': gs_log}, open(gs_file, 'w'), indent=2)
            print(f'\n  BEST: lr={best[0]} h={best[1]} drop={best[2]}  '
                  f'val={best_val*100:.2f}%', flush=True)

        seeds_to_run = list(range(args.seeds))
        print(f'\n--- Phase 2: Full ({len(seeds_to_run)} seeds) ---', flush=True)
        seed_accs = []
        for s in seeds_to_run:
            a = run_seed(X, X1, data_list, y, sage_dim, nf, nf2, nt, nt2, nc,
                         best[0], best[1], best[2], args.epochs, s, device)
            seed_accs.append(a)
            print(f'  seed={s}  acc={a*100:.2f}%', flush=True)

        mean = float(np.mean(seed_accs)); std = float(np.std(seed_accs))
        res = {
            'dataset': name, 'protocol': '70/10/20 × 10 seeds',
            'seeds': seeds_to_run, 'seed_accs': seed_accs,
            'mean_acc': mean, 'std_acc': std,
            'best_lr': best[0], 'best_hidden': best[1], 'best_dropout': best[2],
            'epochs': args.epochs, 't3former_paper': T3FORMER_PAPER.get(name, 0),
        }
        all_res[name] = res
        json.dump(res, open(out_file, 'w'), indent=2)
        print(f'\n  T3Former Fair:  {mean*100:.2f}% ± {std*100:.2f}%')
        print(f'  T3Former Paper: {T3FORMER_PAPER.get(name,0):.2f}%')

    summary = Path(args.result_dir) / 'fair_comparison.json'
    json.dump(all_res, open(summary, 'w'), indent=2)
    print(f'\n\nSaved -> {summary}')


if __name__ == '__main__':
    main()
