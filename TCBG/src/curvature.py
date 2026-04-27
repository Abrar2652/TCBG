"""
src/curvature.py
Forman-Ricci curvature computation for temporal graphs.

For each temporal edge (u, v, t), compute curvature using the causal
subgraph G_t = all edges with timestamp in [t - epsilon, t].

  kappa(e) = 4 - deg_t(u) - deg_t(v) + 3 * triangles_t(e)

Algorithm uses a sliding-window data structure for O(|E|) amortized cost.
"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Tuple

import numpy as np


EdgeCurv = Tuple[int, int, float, float]  # (u, v, timestamp, kappa)


def compute_forman_ricci(
    temporal_edges: List[Tuple[int, int, float]],
    lookback_epsilon: float,
) -> List[EdgeCurv]:
    """
    Compute Forman-Ricci curvature for every temporal edge.

    Parameters
    ----------
    temporal_edges : list of (u, v, t)
        Temporal edges sorted by timestamp.  Duplicates (same u,v different t)
        are allowed and treated as separate events.
    lookback_epsilon : float
        Causal window size.  For edge (u,v,t) the curvature neighbourhood is
        all edges with timestamp in [t - epsilon, t].

    Returns
    -------
    list of (u, v, t, kappa)
    """
    if not temporal_edges:
        return []

    # Sort by timestamp (stable)
    edges = sorted(temporal_edges, key=lambda e: e[2])

    # Sliding window: maintain active edges in [t - epsilon, t]
    # deg[v]          : degree of v inside the window
    # adj[v]          : set of neighbours of v inside the window
    # edge_set        : set of (min(u,v), max(u,v)) inside the window
    # For triangle counting we need adjacency lists.

    deg: defaultdict[int, int] = defaultdict(int)
    adj: defaultdict[int, set] = defaultdict(set)
    edge_set: set = set()
    # Keep a deque of (t_i, u, v) so we can expire old edges.
    from collections import deque
    window: deque = deque()   # entries: (t, u, v)

    results: List[EdgeCurv] = []

    for u, v, t in edges:
        # ---- expire edges outside the lookback window ----
        while window and window[0][0] < t - lookback_epsilon:
            t_old, a, b = window.popleft()
            key = (min(a, b), max(a, b))
            if key in edge_set:
                edge_set.discard(key)
                deg[a] -= 1
                deg[b] -= 1
                adj[a].discard(b)
                adj[b].discard(a)

        # ---- count triangles before adding (u,v) ----
        # triangles_t(e) = |N(u) ∩ N(v)| in current window
        triangle_count = len(adj[u] & adj[v])

        # ---- read degrees *before* adding the new edge ----
        # The guide specifies deg_t(u) in G_t which contains edges up to t
        # including the current edge — so we add first then read.
        key = (min(u, v), max(u, v))
        if key not in edge_set:
            edge_set.add(key)
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
            window.append((t, u, v))

        kappa = 4.0 - deg[u] - deg[v] + 3.0 * triangle_count
        results.append((u, v, t, kappa))

    return results


def compute_forman_ricci_static(
    edges: List[Tuple[int, int]],
    edge_weights: dict | None = None,
) -> dict:
    """
    Compute Forman-Ricci curvature on a static graph (used in ablations).

    Parameters
    ----------
    edges : list of (u, v)
    edge_weights : optional dict (u,v) -> weight (ignored in formula)

    Returns
    -------
    dict mapping (u,v) -> kappa
    """
    deg: defaultdict[int, int] = defaultdict(int)
    adj: defaultdict[int, set] = defaultdict(set)
    edge_list = []

    for u, v in edges:
        key = (min(u, v), max(u, v))
        if key not in {(min(a, b), max(a, b)) for a, b in edge_list}:
            edge_list.append((u, v))
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)

    result = {}
    for u, v in edge_list:
        triangle_count = len(adj[u] & adj[v])
        kappa = 4.0 - deg[u] - deg[v] + 3.0 * triangle_count
        result[(u, v)] = kappa
        result[(v, u)] = kappa

    return result


def auto_epsilon(
    temporal_edges: List[Tuple[int, int, float]],
    dataset_type: str = "social",
    num_timesteps: int | None = None,
) -> float:
    """
    Compute the lookback epsilon automatically.

    social:  (t_max - t_min) / num_timesteps * 3   — 3-step sliding window
    brain:   t_max - t_min + 1                      — cumulative window (all windows)
             Brain connectivity graphs are dense; using the cumulative
             aggregated graph for curvature gives a global structural role
             for each edge rather than a noisy local snapshot.
    traffic: 3.0  (3 x 5-min intervals = 15-min lookback)
    """
    timestamps = [e[2] for e in temporal_edges]
    t_min, t_max = min(timestamps), max(timestamps)
    span = t_max - t_min

    if dataset_type == "traffic":
        return 3.0

    if dataset_type == "brain":
        # Use cumulative window so curvature at step t uses all edges seen so far.
        # epsilon > span ensures [t - epsilon, t] always covers [0, t].
        return float(span + 1.0)

    # social (and fallback)
    if span == 0:
        return 1.0
    n_ts = num_timesteps if num_timesteps else max(1, int(span) + 1)
    return span / n_ts * 3.0


def normalize_curvatures(edge_curvatures: List[EdgeCurv]) -> List[EdgeCurv]:
    """Min-max normalize kappa values to [0, 1] (used for bifiltration axis)."""
    if not edge_curvatures:
        return []
    kappas = np.array([e[3] for e in edge_curvatures], dtype=float)
    kmin, kmax = kappas.min(), kappas.max()
    if kmax == kmin:
        normalized = np.zeros_like(kappas)
    else:
        normalized = (kappas - kmin) / (kmax - kmin)
    return [(u, v, t, float(kn)) for (u, v, t, _), kn in
            zip(edge_curvatures, normalized)]
