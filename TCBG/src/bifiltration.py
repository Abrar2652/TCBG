"""
src/bifiltration.py
Build the 2-parameter (time, curvature) bifiltration grid.

For each grid point (tau_i, kappa_j) we record which edges (and implied
nodes) are present.  The nesting property is guaranteed by construction:
  E(tau_i, kappa_j) ⊆ E(tau_i', kappa_j')  whenever i<=i', j<=j'.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


EdgeCurv = Tuple[int, int, float, float]  # (u, v, t, kappa)
# Bifiltration cell: set of edges present at (i, j)
Bifiltration = Dict[Tuple[int, int], List[Tuple[int, int]]]


def build_bifiltration(
    edge_curvatures: List[EdgeCurv],
    T_grid: int = 30,
    K_grid: int = 20,
) -> dict:
    """
    Construct the (T_grid x K_grid) bifiltration.

    Parameters
    ----------
    edge_curvatures : list of (u, v, t, kappa)
    T_grid : int   – number of time axis thresholds
    K_grid : int   – number of curvature axis thresholds

    Returns
    -------
    dict with keys:
        'tau_thresholds'   : np.ndarray shape (T_grid,)
        'kappa_thresholds' : np.ndarray shape (K_grid,)
        'cells'            : dict (i,j) -> list of (u,v) edge tuples
                             (only edges that *first appear* at (i,j))
        'cumulative'       : dict (i,j) -> list of (u,v) all edges present
        'nodes'            : dict (i,j) -> set of node ids present
    """
    if not edge_curvatures:
        return {
            'tau_thresholds': np.array([]),
            'kappa_thresholds': np.array([]),
            'cells': {},
            'cumulative': {},
            'nodes': {},
        }

    timestamps = np.array([e[2] for e in edge_curvatures])
    curvatures = np.array([e[3] for e in edge_curvatures])

    t_min, t_max = timestamps.min(), timestamps.max()
    k_min, k_max = curvatures.min(), curvatures.max()

    # Add tiny margin so boundary edges are included
    tau_thresholds = np.linspace(t_min, t_max + 1e-9, T_grid)
    if k_min == k_max:
        kappa_thresholds = np.array([k_min] * K_grid, dtype=float)
    else:
        kappa_thresholds = np.linspace(k_min, k_max + 1e-9, K_grid)

    # For each edge record (i_first_tau, j_first_kappa) — the earliest cell
    # at which it becomes active.  We use searchsorted for speed.
    tau_idx = np.searchsorted(tau_thresholds, timestamps, side='left')
    kappa_idx = np.searchsorted(kappa_thresholds, curvatures, side='left')

    # Clamp to valid grid indices
    tau_idx = np.clip(tau_idx, 0, T_grid - 1)
    kappa_idx = np.clip(kappa_idx, 0, K_grid - 1)

    # cells[(i,j)] = edges that *appear for the first time* at (i,j)
    cells: dict = {}
    for idx, (u, v, t, k) in enumerate(edge_curvatures):
        i = int(tau_idx[idx])
        j = int(kappa_idx[idx])
        cells.setdefault((i, j), []).append((u, v))

    # cumulative[(i,j)] = ALL edges present at (tau_i, kappa_j)
    # Instead of iterating every (i,j) pair (expensive), we build it lazily.
    # Store as a pre-sorted list of (i_birth, j_birth, u, v) for slicing.
    birth_records = []
    for (i, j), edge_list in cells.items():
        for u, v in edge_list:
            birth_records.append((i, j, u, v))

    return {
        'tau_thresholds': tau_thresholds,
        'kappa_thresholds': kappa_thresholds,
        'cells': cells,                   # first-appearance only
        'birth_records': birth_records,   # (i_birth, j_birth, u, v)
        'T_grid': T_grid,
        'K_grid': K_grid,
    }


def get_edges_at(bifiltration: dict, i: int, j: int) -> List[Tuple[int, int]]:
    """
    Return all edges present at grid point (i, j)  (cumulative slice).

    An edge with birth (i0, j0) is present at (i, j) iff i0 <= i AND j0 <= j.
    """
    edges = []
    for (i0, j0, u, v) in bifiltration['birth_records']:
        if i0 <= i and j0 <= j:
            edges.append((u, v))
    return edges


def get_time_slice(bifiltration: dict, j: int) -> List[List[Tuple[int, int]]]:
    """
    Extract the 1-parameter filtration along the time axis at curvature
    level j.  Returns a list of length T_grid where element i contains
    edges active at (i, j).
    """
    T = bifiltration['T_grid']
    slices = [[] for _ in range(T)]
    for (i0, j0, u, v) in bifiltration['birth_records']:
        if j0 <= j:
            # This edge is active for all i >= i0
            for i in range(i0, T):
                slices[i].append((u, v))
    return slices


def get_node_set_at(bifiltration: dict, i: int, j: int) -> set:
    """Return the set of nodes present at grid point (i, j)."""
    nodes = set()
    for (i0, j0, u, v) in bifiltration['birth_records']:
        if i0 <= i and j0 <= j:
            nodes.add(u)
            nodes.add(v)
    return nodes
