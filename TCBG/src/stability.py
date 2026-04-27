"""
src/stability.py
Perturbation robustness utilities for the TCBG stability theorem.

Theorem (guide Section 4.6):
  Let G, G' differ by at most k edge insertions/deletions.
  Then the bottleneck distance between their Graphcodes is O(k/n).

This module provides:
  1. Empirical stability measurement (add/remove k edges, measure distance).
  2. Graphcode distance computation (using node-feature matching as proxy).
  3. Utilities to generate perturbed temporal graphs for experiments.
"""

from __future__ import annotations

import random
from typing import List, Tuple, Optional
import numpy as np


def perturb_temporal_graph(
    temporal_edges: List[Tuple[int, int, float]],
    k: int,
    mode: str = 'mixed',
    seed: int = 42,
) -> List[Tuple[int, int, float]]:
    """
    Perturb a temporal graph by k edge modifications.

    Parameters
    ----------
    temporal_edges : list of (u, v, t)
    k : number of modifications
    mode : 'insert' | 'delete' | 'mixed'
    seed : random seed

    Returns
    -------
    Perturbed edge list.
    """
    rng = random.Random(seed)
    edges = list(temporal_edges)

    if not edges:
        return edges

    timestamps = [e[2] for e in edges]
    t_min, t_max = min(timestamps), max(timestamps)

    all_nodes = set()
    for u, v, t in edges:
        all_nodes.add(u)
        all_nodes.add(v)
    nodes = sorted(all_nodes)

    for _ in range(k):
        if mode == 'insert' or (mode == 'mixed' and rng.random() < 0.5):
            # Insert a random edge
            u = rng.choice(nodes)
            v = rng.choice(nodes)
            if u == v:
                continue
            t = rng.uniform(t_min, t_max)
            edges.append((u, v, t))
        else:
            # Delete a random edge
            if edges:
                idx = rng.randrange(len(edges))
                edges.pop(idx)

    return edges


def graphcode_node_distance(
    nodes_a: List[List[float]],
    nodes_b: List[List[float]],
) -> float:
    """
    Approximate bottleneck distance between two Graphcodes using
    min-cost matching of node feature vectors (L-inf distance).

    Falls back to Wasserstein-style matching via scipy if available.

    Returns a scalar distance value.
    """
    if not nodes_a and not nodes_b:
        return 0.0
    if not nodes_a or not nodes_b:
        return 1.0  # maximum distance

    arr_a = np.array(nodes_a, dtype=np.float32)
    arr_b = np.array(nodes_b, dtype=np.float32)

    try:
        from scipy.spatial.distance import cdist
        from scipy.optimize import linear_sum_assignment

        # Pad to equal size (unmatched points are at distance 1.0)
        na, nb = len(arr_a), len(arr_b)
        if na < nb:
            pad = np.ones((nb - na, arr_a.shape[1]), dtype=np.float32)
            arr_a = np.vstack([arr_a, pad])
        elif nb < na:
            pad = np.ones((na - nb, arr_b.shape[1]), dtype=np.float32)
            arr_b = np.vstack([arr_b, pad])

        cost = cdist(arr_a, arr_b, metric='chebyshev')
        row_ind, col_ind = linear_sum_assignment(cost)
        return float(cost[row_ind, col_ind].max())

    except ImportError:
        # Fallback: greedy nearest-neighbour matching
        from scipy.spatial.distance import cdist
        cost = np.linalg.norm(
            arr_a[:, None, :] - arr_b[None, :, :], axis=-1, ord=np.inf
        )
        matched = set()
        total = 0.0
        for i in range(min(len(arr_a), len(arr_b))):
            cost_masked = cost.copy()
            cost_masked[:, list(matched)] = np.inf
            j = int(cost_masked[i].argmin())
            total = max(total, cost[i, j])
            matched.add(j)
        return total


def measure_stability(
    pipeline,
    temporal_edges: List[Tuple[int, int, float]],
    k_values: List[int] = None,
    n_trials: int = 5,
    seed: int = 42,
) -> dict:
    """
    Empirically measure Graphcode stability under perturbations.

    Parameters
    ----------
    pipeline : TCBGPipeline instance
    temporal_edges : original graph
    k_values : list of perturbation sizes to test
    n_trials : number of random perturbations per k
    seed : base random seed

    Returns
    -------
    dict mapping k -> list of distances (one per trial)
    """
    if k_values is None:
        k_values = [1, 2, 5, 10]

    from .pipeline import TCBGPipeline
    original_data = pipeline.process_graph(temporal_edges)
    original_nodes = original_data.x.tolist() if original_data.x is not None else []

    results = {}
    for k in k_values:
        distances = []
        for trial in range(n_trials):
            perturbed = perturb_temporal_graph(
                temporal_edges, k=k, mode='mixed', seed=seed + trial * 100
            )
            perturbed_data = pipeline.process_graph(perturbed)
            perturbed_nodes = perturbed_data.x.tolist() if perturbed_data.x is not None else []
            d = graphcode_node_distance(original_nodes, perturbed_nodes)
            distances.append(d)
        results[k] = distances

    return results
