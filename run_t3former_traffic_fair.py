"""
run_t3former_traffic_fair.py — Fair T3Former eval on PEMS04/08/BAY (binary + multi).

Protocol: StratifiedKFold(5) × 10 seeds, skip grid, use T3Former paper defaults.
Loads features precomputed by gen_traffic_features.py.
"""
from __future__ import annotations
import argparse, json, os, pickle, sys, warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch

warnings.filterwarnings('ignore')
HERE = Path(__file__).parent.resolve()

DATASETS = ['pems04', 'pems08', 'pemsbay']

# T3Former paper Table 3
T3_PAPER = {
    ('pems04', 2): 96.76, ('pems04', 3): 92.66,
    ('pems08', 2): 95.16, ('pems08', 3): 89.65,
    ('pemsbay', 2): 96.68, ('pemsbay', 3): 92.35,
}

FULL_SEEDS = list(range(10))


class TrafficDataset(Dataset):
    def __init__(self, X, X1, graphs, y):
        self.X, self.X1, self.graphs, self.y = X, X1, graphs, y
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.X1[i], self.graphs[i], self.y[i]


def collate(batch):
    X, X1, gs, y = zip(*batch)
    return torch.stack(X), torch.stack(X1), Batch.from_data_list(list(gs)), torch.stack(y)


def load_traffic(name, task, t3former_dir):
    fdir = Path(t3former_dir) / 'traffic_features'
    with open(fdir / f'{name}_betti_binary.pkl', 'rb') as f:
        betti = pickle.load(f)
    with open(fdir / f'{name}_dos_binary.pkl', 'rb') as f:
        dos = pickle.load(f)
    with open(fdir / f'{name}_sage_{task}.pkl', 'rb') as f:
        data_list = pickle.load(f)
    X0 = torch.tensor(np.asarray(betti), dtype=torch.float32)
    X1 = torch.tensor(np.asarray(dos), dtype=torch.float32)
    y = np.array([int(d.y.item()) for d in data_list])
    return data_list, X0, X1, y


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


def run_seed(X, X1, data_list, y_np, sage_dim, nf, nf2, nt, nt2, nc,
             lr, h, drop, epochs, seed, device):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_accs = []
    y_t = torch.tensor(y_np, dtype=torch.long)
    for tr_idx, val_idx in skf.split(np.zeros(len(y_np)), y_np):
        mk = lambda idx: TrafficDataset(X[idx], X1[idx],
                                        [data_list[k] for k in idx], y_t[idx])
        tr = DataLoader(mk(tr_idx), batch_size=16, shuffle=True,  collate_fn=collate)
        te = DataLoader(mk(val_idx), batch_size=16, shuffle=False, collate_fn=collate)
        a = train_one(tr, te, sage_dim, nf, nf2, nt, nt2, nc,
                      lr, h, drop, epochs, device)
        fold_accs.append(a)
    return float(np.mean(fold_accs))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets',     nargs='+', default=DATASETS)
    p.add_argument('--tasks',        nargs='+', default=['binary', 'multi'],
                   choices=['binary', 'multi'])
    p.add_argument('--device',       default='cuda')
    p.add_argument('--epochs',       type=int, default=30)
    p.add_argument('--t3former_dir', required=True)
    p.add_argument('--result_dir',   default=str(HERE / 'results' / 't3former_traffic_fair'))
    p.add_argument('--seeds',        type=int, default=10)
    p.add_argument('--lr',           type=float, default=0.005)
    p.add_argument('--h',            type=int,   default=32)
    p.add_argument('--drop',         type=float, default=0.3)
    p.add_argument('--force',        action='store_true')
    args = p.parse_args()

    sys.path.insert(0, args.t3former_dir)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.result_dir, exist_ok=True)
    seeds = list(range(args.seeds))

    print(f'T3Former Traffic Fair | device={device} | epochs={args.epochs} | '
          f'seeds={seeds} | lr={args.lr} h={args.h} drop={args.drop}', flush=True)

    all_res = {}
    for ds in args.datasets:
        for task in args.tasks:
            nc = 2 if task == 'binary' else 3
            key = f'{ds}_{task}'
            out_file = Path(args.result_dir) / f'{key}_fair.json'
            if out_file.exists() and not args.force:
                all_res[key] = json.load(open(out_file))
                print(f'[skip] {key}')
                continue
            print(f'\n{"="*60}  {key}')
            data_list, X, X1, y = load_traffic(ds, task, args.t3former_dir)
            sage_dim = data_list[0].x.shape[-1]
            nt, nf   = X.shape[1], X.shape[2]
            nt2, nf2 = X1.shape[1], X1.shape[2]
            print(f'  n={len(y)}  nc={nc}  sage_dim={sage_dim}  '
                  f'betti={tuple(X.shape)}  dos={tuple(X1.shape)}', flush=True)

            seed_accs = []
            for s in seeds:
                a = run_seed(X, X1, data_list, y, sage_dim,
                             nf, nf2, nt, nt2, nc,
                             args.lr, args.h, args.drop, args.epochs, s, device)
                seed_accs.append(a)
                print(f'  seed={s}  acc={a*100:.2f}%', flush=True)

            mean = float(np.mean(seed_accs)); std = float(np.std(seed_accs))
            res = {
                'dataset': ds, 'task': task, 'num_classes': nc,
                'seeds': seeds, 'seed_accs': seed_accs,
                'mean_acc': mean, 'std_acc': std,
                'lr': args.lr, 'h': args.h, 'drop': args.drop,
                'epochs': args.epochs,
                't3former_paper': T3_PAPER.get((ds, nc), 0),
            }
            all_res[key] = res
            json.dump(res, open(out_file, 'w'), indent=2)
            print(f'  T3Former Fair: {mean*100:.2f}% ± {std*100:.2f}%  '
                  f'(paper {T3_PAPER.get((ds, nc), 0):.2f}%)', flush=True)

    summary = Path(args.result_dir) / 'fair_comparison.json'
    json.dump(all_res, open(summary, 'w'), indent=2)
    print(f'\nSaved → {summary}')


if __name__ == '__main__':
    main()