"""
gen_traffic_features.py — Generate T3Former traffic features for PEMS04/08/BAY.

For each dataset + task (binary | multi):
  - Partition raw data into 2-hour intervals (24 × 5-min steps)
  - Build TemporalData with node-level activity via speed-threshold filtering
  - Compute sliding-window Betti + DoS using quantile thresholds (T3Former protocol)
  - Build SAGE Data objects with temporal node features
Saves to T3Former-3311/traffic_features/{ds}_{feat}_{task}.pkl
"""
from __future__ import annotations
import argparse, csv, os, pickle, sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from scipy.linalg import eigh
from scipy.stats import gaussian_kde
from torch_geometric.data import Data
from tqdm import tqdm

HERE = Path(__file__).parent.resolve()
T3_DIR = HERE / 'T3Former-3311'
FEAT_DIR = T3_DIR / 'traffic_features'

STEPS_PER_INTERVAL = 24       # 2 hours of 5-min steps
THRESH_BINS = 20              # quantile thresholds
WINDOW = 3; JUMP = 2          # sliding window over thresholds → 9 windows

DATASETS = {
    'pems04': {
        'npz': 'TCBG/data/raw/PEMS04/PEMS04.npz',
        'csv': 'TCBG/data/raw/PEMS04/PEMS04.csv',
        'speed_idx': 1,
    },
    'pems08': {
        'npz': 'TCBG/data/raw/PEMS08/PEMS08.npz',
        'csv': 'TCBG/data/raw/PEMS08/PEMS08.csv',
        'speed_idx': 1,
    },
    'pemsbay': {
        'npz': 'TCBG/data/raw/PEMSBAY/pems-bay.npz',
        'pkl': 'TCBG/data/raw/PEMSBAY/adj_mx_bay.pkl',
        'speed_idx': 0,
    },
}


def load_adj_csv(csv_path, n_nodes):
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    with open(csv_path) as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            if len(row) >= 2:
                u, v = int(row[0]), int(row[1])
                adj[u, v] = 1
                adj[v, u] = 1
    return adj


def load_adj_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    if isinstance(data, list):
        adj = np.array(data[2])
    else:
        adj = np.array(data)
    adj = (adj > 0).astype(np.float32)
    return adj


def load_pems(name):
    info = DATASETS[name]
    npz = np.load(HERE / info['npz'])
    if 'data' in npz:
        arr = npz['data']
    else:
        arr = npz['x']
    arr = arr.astype(np.float32)
    T, N, F = arr.shape
    if 'csv' in info:
        adj = load_adj_csv(HERE / info['csv'], N)
    else:
        adj = load_adj_pkl(HERE / info['pkl'])
    return arr, adj, info['speed_idx']


def make_labels(mean_speeds, n_classes):
    pct = np.percentile(mean_speeds, [35, 60])
    if n_classes == 2:
        return np.where(mean_speeds <= pct[0], 0, 1).astype(int)
    y = np.zeros(len(mean_speeds), dtype=int)
    y[mean_speeds > pct[0]] = 1
    y[mean_speeds > pct[1]] = 2
    return y


def compute_dos(G, num_bins=4, bandwidth=0.05):
    if G.number_of_nodes() == 0:
        return np.zeros(num_bins)
    L = nx.normalized_laplacian_matrix(G).toarray()
    ev = eigh(L, eigvals_only=True)
    if len(ev) < 2 or float(np.ptp(ev)) < 1e-9:
        return np.zeros(num_bins)
    try:
        kde = gaussian_kde(ev, bw_method=bandwidth)
    except Exception:
        return np.zeros(num_bins)
    bins = np.linspace(ev.min(), ev.max(), num_bins)
    return kde(bins)


def sliding_windows_betti_dos(G, thresholds):
    """Returns (betti_arr (W,4), dos_arr (W,4))."""
    import pyflagser
    node_time = dict(G.nodes(data='time'))
    betti, dos = [], []
    W = (len(thresholds) - WINDOW) // JUMP + 1
    for i in range(0, W * JUMP, JUMP):
        t_s, t_e = thresholds[i], thresholds[i + WINDOW - 1]
        active = {n for n, times in node_time.items()
                  if times and any(t_s <= t <= t_e for t in times)}
        sub = G.subgraph(active).copy() if active else nx.Graph()
        if sub.number_of_nodes() == 0:
            betti.append((0.0, 0.0, 0.0, 0.0))
            dos.append(np.zeros(4))
        else:
            adj = nx.to_numpy_array(sub)
            hom = pyflagser.flagser_unweighted(
                adj, min_dimension=0, max_dimension=2, directed=False, coeff=2)
            b0, b1 = hom['betti'][0], hom['betti'][1]
            betti.append((float(b0), float(b1),
                          float(sub.number_of_nodes()), float(sub.number_of_edges())))
            dos.append(compute_dos(sub))
    return np.array(betti, dtype=np.float32), np.array(dos, dtype=np.float32)


def build_intervals(arr, adj, speed_idx):
    """Return list of (interval_speed_matrix (T, N), mean_speed)."""
    T_total, N, _ = arr.shape
    n_int = T_total // STEPS_PER_INTERVAL
    speed = arr[:, :, speed_idx]          # (T_total, N)
    intervals = []
    mean_speeds = []
    for i in range(n_int):
        s, e = i * STEPS_PER_INTERVAL, (i + 1) * STEPS_PER_INTERVAL
        iv = speed[s:e]
        intervals.append(iv)
        mean_speeds.append(float(iv.mean()))
    return intervals, np.array(mean_speeds), adj


def process_dataset(name, ckpt_dir):
    print(f'\n[{name}] loading raw data', flush=True)
    arr, adj, sidx = load_pems(name)
    intervals, mean_speeds, adj = build_intervals(arr, adj, sidx)
    n = len(intervals)
    n_nodes = adj.shape[0]
    speed_thresh = float(np.median(arr[..., sidx]))
    print(f'[{name}] n_intervals={n}  n_nodes={n_nodes}  '
          f'speed_thresh={speed_thresh:.2f}', flush=True)

    # Edge list (static)
    edges = []
    for u in range(n_nodes):
        for v in range(u + 1, n_nodes):
            if adj[u, v] > 0:
                edges.append((u, v))
    edge_index = torch.tensor(
        [[u for u, v in edges] + [v for u, v in edges],
         [v for u, v in edges] + [u for u, v in edges]], dtype=torch.long)

    # Compute global thresholds based on interval step indices (0..23)
    all_t = np.arange(STEPS_PER_INTERVAL, dtype=float)
    thresholds = np.quantile(all_t, np.linspace(0, 1, THRESH_BINS))

    # Per-interval features
    ckpt_file = ckpt_dir / f'{name}_betti_dos.ckpt.pkl'
    sw_betti = []; sw_dos = []; start = 0
    if ckpt_file.exists():
        with open(ckpt_file, 'rb') as f:
            saved = pickle.load(f)
        sw_betti, sw_dos = saved['betti'], saved['dos']
        start = len(sw_betti)
        print(f'[{name}] resume from interval {start}/{n}', flush=True)

    for i in tqdm(range(start, n), desc=name, initial=start, total=n, miniters=20):
        iv = intervals[i]  # (T=24, N)
        # Build graph with node-level active timestamps (speed > threshold)
        G = nx.Graph()
        G.add_edges_from(edges)
        for node in range(n_nodes):
            active_t = np.nonzero(iv[:, node] > speed_thresh)[0].tolist()
            G.add_node(node, time=active_t)
        b, d = sliding_windows_betti_dos(G, thresholds)
        sw_betti.append(b)
        sw_dos.append(d)
        if (i + 1) % 40 == 0:
            with open(ckpt_file, 'wb') as f:
                pickle.dump({'betti': sw_betti, 'dos': sw_dos}, f)

    sw_betti = np.stack(sw_betti)   # (n_int, W, 4)
    sw_dos   = np.stack(sw_dos)     # (n_int, W, 4)

    # Save betti/dos (shared across binary and multi since labels differ only)
    with open(FEAT_DIR / f'{name}_betti_binary.pkl', 'wb') as f:
        pickle.dump(sw_betti, f)
    with open(FEAT_DIR / f'{name}_dos_binary.pkl', 'wb') as f:
        pickle.dump(sw_dos, f)
    print(f'[{name}] saved betti/dos', flush=True)

    # SAGE data_list: one Data per interval, x = temporal speed feature (N, 24)
    for n_classes in [2, 3]:
        task = 'binary' if n_classes == 2 else 'multi'
        labels = make_labels(mean_speeds, n_classes)
        data_list = []
        for i in range(n):
            x = torch.tensor(intervals[i].T, dtype=torch.float)   # (N, 24)
            y = torch.tensor([int(labels[i])], dtype=torch.long)
            data_list.append(Data(x=x, edge_index=edge_index, y=y))
        with open(FEAT_DIR / f'{name}_sage_{task}.pkl', 'wb') as f:
            pickle.dump(data_list, f)
        print(f'[{name}] saved sage_{task}  label_dist={np.bincount(labels).tolist()}',
              flush=True)

    if ckpt_file.exists():
        ckpt_file.unlink()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', nargs='+', default=list(DATASETS.keys()))
    p.add_argument('--ckpt_dir', default='/nas/ckgfs/jaunts/jahin/tcbg_data/traffic_ckpt')
    args = p.parse_args()

    FEAT_DIR.mkdir(exist_ok=True)
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    for ds in args.datasets:
        done = all((FEAT_DIR / f'{ds}_{f}_{t}.pkl').exists()
                   for f in ['betti', 'dos'] for t in ['binary'])
        done = done and all((FEAT_DIR / f'{ds}_sage_{t}.pkl').exists()
                            for t in ['binary', 'multi'])
        if done:
            print(f'[skip] {ds} features already exist')
            continue
        process_dataset(ds, Path(args.ckpt_dir))


if __name__ == '__main__':
    main()