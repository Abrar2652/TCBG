"""
Generate missing DoS feature files for NeuroGraph DynHCP datasets.
Checkpointed: saves partial results every N subjects to ckpt_dir.
"""
from __future__ import annotations
import argparse
import os
import pickle
import gc
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.linalg import eigh
from scipy.stats import gaussian_kde
from tqdm import tqdm


def compute_dos(G, num_bins=4, bandwidth=0.05):
    if G.number_of_nodes() == 0:
        return np.zeros(num_bins)
    L = nx.normalized_laplacian_matrix(G).toarray()
    eigenvalues = eigh(L, eigvals_only=True)
    if len(eigenvalues) < 2 or float(np.ptp(eigenvalues)) < 1e-9:
        return np.zeros(num_bins)
    try:
        kde = gaussian_kde(eigenvalues, bw_method=bandwidth)
    except Exception:
        return np.zeros(num_bins)
    bin_centers = np.linspace(eigenvalues.min(), eigenvalues.max(), num_bins)
    return kde(bin_centers)


def dos_extraction(data):
    ptr = data.ptr
    edge_index = data.edge_index
    out = []
    for j in range(len(ptr) - 1):
        node_start, node_end = ptr[j].item(), ptr[j + 1].item()
        mask = (edge_index[0] >= node_start) & (edge_index[0] < node_end)
        edges = edge_index[:, mask] - node_start
        G = nx.Graph()
        G.add_nodes_from(range(node_end - node_start))
        G.add_edges_from(edges.t().tolist())
        out.append(compute_dos(G))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', required=True)
    p.add_argument('--data_root', required=True)
    p.add_argument('--ckpt_dir', required=True)
    p.add_argument('--datasets', nargs='+', default=['DynHCPActivity', 'DynHCPGender', 'DynHCPAge'])
    p.add_argument('--ckpt_every', type=int, default=500)
    args = p.parse_args()

    from NeuroGraph.datasets import NeuroGraphDynamic

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    for name in args.datasets:
        out_path = Path(args.out_dir) / f'dos_{name}.data'
        if out_path.exists():
            print(f'[skip] {out_path} exists', flush=True)
            continue
        ckpt_path = Path(args.ckpt_dir) / f'dos_{name}.ckpt.pkl'

        print(f'[load] NeuroGraphDynamic {name} root={args.data_root}', flush=True)
        ds = NeuroGraphDynamic(root=args.data_root, name=name)
        n = len(ds.dataset)
        print(f'[run ] DoS on {n} subjects', flush=True)

        if ckpt_path.exists():
            with open(ckpt_path, 'rb') as f:
                dos = pickle.load(f)
            start = len(dos)
            print(f'[resume] from subject {start}/{n}', flush=True)
        else:
            dos = []
            start = 0

        bar = tqdm(range(start, n), desc=name, initial=start, total=n, miniters=50)
        for i in bar:
            dos.append(dos_extraction(ds.dataset[i]))
            if (i + 1) % args.ckpt_every == 0:
                with open(ckpt_path, 'wb') as f:
                    pickle.dump(dos, f)
                gc.collect()
                bar.write(f'[ckpt] {name} {i+1}/{n} → {ckpt_path}')

        with open(out_path, 'wb') as f:
            pickle.dump(dos, f)
        print(f'[save] {out_path}  n={len(dos)}', flush=True)
        if ckpt_path.exists():
            ckpt_path.unlink()


if __name__ == '__main__':
    main()
