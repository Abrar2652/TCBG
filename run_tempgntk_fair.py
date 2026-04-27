"""
run_tempgntk_fair.py
Fair evaluation of TempGNTK on 5 temporal graph datasets.

Protocol (identical to TCBG):
    StratifiedKFold(5) x 10 seeds = 50 runs per dataset
    Gram matrix computed once per dataset and cached.

Usage:
    python run_tempgntk_fair.py --device cuda
    python run_tempgntk_fair.py --device cuda --datasets infectious_ct1 dblp_ct1
"""

from __future__ import annotations
import argparse, json, sys, time, warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings('ignore')

HERE       = Path(__file__).parent.resolve()
REPO_DIR   = HERE / 'TempGNTK_repo'
RESULT_DIR = HERE / 'results' / 'tempgntk_fair'
CACHE_DIR  = HERE / 'cache_tempgntk'
DATA_DIR   = HERE / 'data'

# path prefix passed to readTUds (appends _A.txt, _graph_labels.txt, etc.)
RAW_DATA_PATHS = {
    'infectious_ct1': DATA_DIR / 'infectious_ct1' / 'infectious_ct1',
    'dblp_ct1':       DATA_DIR / 'dblp_ct1'       / 'dblp_ct1',
    'tumblr_ct1':     DATA_DIR / 'tumblr_ct1'      / 'tumblr_ct1',
    'mit_ct1':        DATA_DIR / 'mit_ct1'          / 'mit_ct1',
    'highschool_ct1': DATA_DIR / 'highschool_ct1'  / 'highschool_ct1',
    'facebook_ct1':   DATA_DIR / 'facebook_ct1'    / 'facebook_ct1',
}

ALL_DATASETS = list(RAW_DATA_PATHS.keys())
SEEDS        = list(range(10))


class Args:
    time_dim=25; alpha=25**0.5; beta=25**0.5; num_sub_graphs=5
    k_recent=15; num_mlp_layers=1; device='cpu'; node_ntk=False
    encode_time=True; relative_difference=True; neighborhood_avg=False
    node_onehot=False; mean_graph_pooling=False; jumping_knowledge=False
    skip_connection=False


def setup_repo():
    if not REPO_DIR.exists():
        import subprocess
        print("Cloning TempGNTK_repo...")
        subprocess.run(['git','clone','https://github.com/kthrn22/TempGNTK.git',
                        str(REPO_DIR)], check=True)
    sys.path.insert(0, str(REPO_DIR))


def load_dataset(dataset: str):
    from utils_graph_classification import readTUds, temporal_graph_from_TUds
    path = str(RAW_DATA_PATHS[dataset])
    ng, labels, g_node, nmap, g_edge = readTUds(path)
    graphs = temporal_graph_from_TUds(ng, labels, g_node, nmap, g_edge)
    labels = np.array(labels)
    uniq = np.unique(labels)
    if uniq[0] != 0:
        m = {v: i for i, v in enumerate(uniq)}
        labels = np.array([m[l] for l in labels])
    return graphs, labels


def compute_full_gram(graphs, args, cache_path: Path):
    from utils_graph_classification import pre_t_gntk, compute_gram_matrix
    if cache_path.exists():
        print(f"  Loading cached Gram: {cache_path.name}")
        return np.load(str(cache_path))
    print(f"  Computing Gram matrix ({len(graphs)} graphs)...")
    t0 = time.time()
    ds   = pre_t_gntk(graphs, args)
    gram = compute_gram_matrix(ds, args, mode='train')
    print(f"  Shape: {gram.shape}  [{(time.time()-t0)/60:.1f} min]")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(cache_path), gram)
    return gram


def normalize_fold_kernels(gram, train_idx, test_idx, K):
    train_k = np.zeros((len(train_idx), len(train_idx)))
    test_k  = np.zeros((len(test_idx),  len(train_idx)))
    for k in range(K):
        tg = gram[np.ix_(train_idx, train_idx, [k])].squeeze(-1).copy()
        d  = np.sqrt(np.diag(tg)); d[d==0] = 1
        tg /= (d[:,None] * d[None,:]); train_k += tg

        tg2 = gram[np.ix_(test_idx, train_idx, [k])].squeeze(-1).copy()
        dte = np.sqrt(gram[test_idx,  test_idx,  k]); dte[dte==0] = 1
        dtr = np.sqrt(gram[train_idx, train_idx, k]); dtr[dtr==0] = 1
        tg2 /= (dte[:,None] * dtr[None,:]); test_k += tg2
    return train_k, test_k


def svm_eval(train_k, test_k, y_tr, y_te):
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    C_list = np.logspace(-2, 4, 30).tolist()
    clf = GridSearchCV(SVC(kernel='precomputed', max_iter=200000),
                       {'C': C_list}, cv=5, scoring='accuracy', n_jobs=-1)
    clf.fit(train_k, y_tr)
    return float((clf.best_estimator_.predict(test_k) == y_te).mean())


def run_dataset(dataset: str, device: str):
    from sklearn.model_selection import StratifiedKFold
    print(f"\n{'='*60}\nDATASET: {dataset}\n{'='*60}")
    args = Args(); args.device = device

    graphs, labels = load_dataset(dataset)
    print(f"  {len(graphs)} graphs, {len(np.unique(labels))} classes")

    cache = CACHE_DIR / f"gram_{dataset}_K{args.num_sub_graphs}_kr{args.k_recent}.npy"
    gram  = compute_full_gram(graphs, args, cache)

    seed_means = []
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        fold_accs = []
        for tr_idx, te_idx in skf.split(np.zeros(len(labels)), labels):
            tk, vk = normalize_fold_kernels(gram, tr_idx, te_idx, args.num_sub_graphs)
            fold_accs.append(svm_eval(tk, vk, labels[tr_idx], labels[te_idx]))
        sm = float(np.mean(fold_accs))
        seed_means.append(sm)
        print(f"  seed={seed}  mean={sm*100:.2f}%", flush=True)

    mean, std = float(np.mean(seed_means)), float(np.std(seed_means))
    print(f"\n  RESULT: {mean*100:.2f}% +/- {std*100:.2f}%")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULT_DIR / f'{dataset}_fair.json'
    json.dump({'dataset': dataset,
               'protocol': 'StratifiedKFold(5) x 10 seeds',
               'seeds': SEEDS, 'seed_means': seed_means,
               'mean_acc': mean, 'std_acc': std},
              open(out, 'w'), indent=2)
    print(f"  Saved: {out}")
    return mean, std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=ALL_DATASETS)
    parser.add_argument('--device',   default='cuda')
    args = parser.parse_args()

    setup_repo()
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"TempGNTK Fair Eval | device={device}")
    print(f"Protocol: StratifiedKFold(5) x {len(SEEDS)} seeds\n")

    summary = {}
    for ds in args.datasets:
        done = RESULT_DIR / f'{ds}_fair.json'
        if done.exists():
            d = json.load(open(done))
            print(f"SKIP {ds} — {d['mean_acc']*100:.2f}% +/- {d['std_acc']*100:.2f}%")
            summary[ds] = (d['mean_acc'], d['std_acc'])
            continue
        m, s = run_dataset(ds, device)
        summary[ds] = (m, s)

    print(f"\n{'='*50}\nSUMMARY\n{'='*50}")
    for ds, (m, s) in summary.items():
        print(f"  {ds:20s}  {m*100:.2f}% +/- {s*100:.2f}%")

    json.dump({k: {'mean': v[0], 'std': v[1]} for k, v in summary.items()},
              open(RESULT_DIR / 'summary.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
