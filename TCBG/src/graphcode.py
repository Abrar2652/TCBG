"""
src/graphcode.py
Graphcode construction from a 2-parameter bifiltration.

Implements Option A from the guide (Section 7.3):
  - Use GUDHI for 1-param PH at each curvature slice.
  - Connect bars across consecutive slices by maximum-overlap matching.

If GUDHI is unavailable we fall back to a pure-NumPy Vietoris-Rips style
boundary-matrix reduction for 0-dim and 1-dim persistence.

Node features per Graphcode node:
  [birth_time, death_time, curvature_level (normalised), persistence]
  All normalised to [0, 1] across the graph.
"""

from __future__ import annotations

from typing import List, Tuple, Optional
import numpy as np

# ---------------------------------------------------------------------------
# GUDHI import (optional)
# ---------------------------------------------------------------------------
try:
    import gudhi
    _GUDHI_AVAILABLE = True
except ImportError:
    _GUDHI_AVAILABLE = False


# ---------------------------------------------------------------------------
# Persistence computation helpers
# ---------------------------------------------------------------------------

def _edges_to_adj(edges: List[Tuple[int, int]], n_nodes: int) -> np.ndarray:
    """Build adjacency matrix from edge list."""
    A = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for u, v in edges:
        A[u, v] = 1.0
        A[v, u] = 1.0
    return A


def _compute_ph_gudhi(
    edge_sequence: List[List[Tuple[int, int]]],
    n_nodes: int,
    hom_dim: int = 1,
) -> List[Tuple[float, float]]:
    """
    Compute 1-parameter persistence barcode using GUDHI SimplexTree.

    edge_sequence[i] = list of edges present at filtration step i.
    Returns list of (birth_step, death_step) bars.
    """
    st = gudhi.SimplexTree()

    # Insert each vertex at the step it first appears in an edge.
    # This makes H0 birth times meaningful (time vertex enters the network)
    # rather than forcing all births to 0.
    vertex_first_seen: dict = {}
    for i, edges in enumerate(edge_sequence):
        for u, v in edges:
            if u not in vertex_first_seen:
                vertex_first_seen[u] = float(i)
            if v not in vertex_first_seen:
                vertex_first_seen[v] = float(i)
    for v in range(n_nodes):
        st.insert([v], filtration=vertex_first_seen.get(v, 0.0))

    # Insert edges in filtration order
    seen = set()
    for i, edges in enumerate(edge_sequence):
        for u, v in edges:
            key = (min(u, v), max(u, v))
            if key not in seen:
                seen.add(key)
                st.insert([u, v], filtration=float(i))
                # Add triangle closures (2-simplices) if possible
                # (improves H1 computation accuracy)
                for w in range(n_nodes):
                    if w != u and w != v:
                        if (min(u, w), max(u, w)) in seen and \
                           (min(v, w), max(v, w)) in seen:
                            st.insert([u, v, w], filtration=float(i))

    st.compute_persistence()
    pairs = st.persistence_pairs()

    bars = []
    for pair in pairs:
        if len(pair) == 2:
            birth_simplex, death_simplex = pair
            if len(birth_simplex) - 1 == hom_dim:
                birth_val = st.filtration(birth_simplex)
                death_val = st.filtration(death_simplex) if death_simplex else float(len(edge_sequence))
                bars.append((birth_val, death_val))

    return bars


def _compute_ph_numpy(
    edge_sequence: List[List[Tuple[int, int]]],
    n_nodes: int,
    hom_dim: int = 1,
) -> List[Tuple[float, float]]:
    """
    Fallback: compute 0-dim or 1-dim PH via Union-Find (H0) or
    naive boundary matrix reduction (H1).
    """
    if hom_dim == 0:
        return _compute_h0(edge_sequence, n_nodes)
    else:
        return _compute_h1_naive(edge_sequence, n_nodes)


def _compute_h0(
    edge_sequence: List[List[Tuple[int, int]]],
    n_nodes: int,
) -> List[Tuple[float, float]]:
    """H0 persistence via Union-Find on edge filtration.
    Each vertex is born at the step its first edge appears."""
    parent = list(range(n_nodes))
    rank = [0] * n_nodes
    # Vertices born when they first appear in an edge (not at step 0)
    vertex_birth: dict = {}
    for i, edges in enumerate(edge_sequence):
        for u, v in edges:
            if u not in vertex_birth:
                vertex_birth[u] = float(i)
            if v not in vertex_birth:
                vertex_birth[v] = float(i)
    birth = {v: vertex_birth.get(v, 0.0) for v in range(n_nodes)}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y, step):
        rx, ry = find(x), find(y)
        if rx == ry:
            return None  # creates a loop
        # Younger component dies
        bx, by = birth[rx], birth[ry]
        if bx <= by:
            # ry dies
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
                bx, by = by, bx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1
            return (by, float(step))
        else:
            if rank[ry] < rank[rx]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1
            return (bx, float(step))

    bars = []
    seen = set()
    for i, edges in enumerate(edge_sequence):
        for u, v in edges:
            key = (min(u, v), max(u, v))
            if key not in seen:
                seen.add(key)
                bar = union(u, v, i)
                if bar is not None:
                    bars.append(bar)

    # Surviving components get death = infinity (represented as T_grid)
    T = float(len(edge_sequence))
    n_components = len({find(v) for v in range(n_nodes)})
    # All but one surviving component reported with infinite death
    for _ in range(n_components - 1):
        bars.append((0.0, T))

    return bars


def _compute_h1_naive(
    edge_sequence: List[List[Tuple[int, int]]],
    n_nodes: int,
) -> List[Tuple[float, float]]:
    """
    Naive H1 computation: detect cycle-creating edges via Union-Find.
    Each cycle-creating edge creates a 1-bar born at that step.
    Death assigned by a simple pairing heuristic (matching with first
    later edge that destroys the loop — approximated as T_grid).
    """
    parent = list(range(n_nodes))
    rank = [0] * n_nodes

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return False  # cycle
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1
        return True

    bars = []
    T = float(len(edge_sequence))
    seen = set()
    for i, edges in enumerate(edge_sequence):
        for u, v in edges:
            key = (min(u, v), max(u, v))
            if key not in seen:
                seen.add(key)
                merged = union(u, v)
                if not merged:
                    # Cycle created: 1-bar born at step i
                    # Death heuristic: assign T (infinite persistence)
                    bars.append((float(i), T))
    return bars


# ---------------------------------------------------------------------------
# Bar matching across curvature levels
# ---------------------------------------------------------------------------

def _overlap(bar_a: Tuple[float, float], bar_b: Tuple[float, float]) -> float:
    """Compute overlap length between two intervals [b,d)."""
    b = max(bar_a[0], bar_b[0])
    d = min(bar_a[1], bar_b[1])
    return max(0.0, d - b)


def _match_bars(
    bars_j: List[Tuple[float, float]],
    bars_j1: List[Tuple[float, float]],
) -> List[Tuple[int, int]]:
    """
    Greedily match bars between level j and level j+1 by maximum overlap.
    Returns list of (idx_in_j, idx_in_j1) pairs.
    """
    if not bars_j or not bars_j1:
        return []

    matched_j: set = set()
    matched_j1: set = set()
    connections = []

    # Build overlap matrix
    overlaps = np.zeros((len(bars_j), len(bars_j1)))
    for a, ba in enumerate(bars_j):
        for b, bb in enumerate(bars_j1):
            overlaps[a, b] = _overlap(ba, bb)

    # Greedy matching: pick highest overlap pairs first
    while True:
        idx = np.unravel_index(np.argmax(overlaps), overlaps.shape)
        a, b = int(idx[0]), int(idx[1])
        if overlaps[a, b] <= 0:
            break
        if a not in matched_j and b not in matched_j1:
            connections.append((a, b))
            matched_j.add(a)
            matched_j1.add(b)
        overlaps[a, b] = -1.0  # suppress

    return connections


# ---------------------------------------------------------------------------
# Main Graphcode computation
# ---------------------------------------------------------------------------

def compute_graphcode(
    bifiltration: dict,
    T_grid: int,
    K_grid: int,
    hom_dim: int | list = 1,
    min_persistence: float = 0.05,
    max_bars_per_level: int = 30,
) -> Tuple[List[List[float]], List[Tuple[int, int]]]:
    """
    Compute Graphcode from a bifiltration.

    Parameters
    ----------
    bifiltration      : dict returned by build_bifiltration()
    T_grid, K_grid    : grid dimensions
    hom_dim           : 0, 1, or [0,1] for both dimensions concatenated
    min_persistence   : drop bars shorter than this (normalised to [0,1])
    max_bars_per_level: keep only the top-K most persistent bars per level
                        to bound Graphcode size (default 30)

    Returns
    -------
    gc_nodes : list of [birth, death, curv_level, persistence]
               birth/death in [0,1], curv_level in [0,1], persistence in [0,1]
               Features are NOT re-normalised per graph — absolute scale is
               preserved so different graphs produce distinguishable embeddings.
    gc_edges : list of (node_i, node_j) edges
    """
    dims = [hom_dim] if isinstance(hom_dim, int) else list(hom_dim)

    all_nodes: List[List[float]] = []
    all_edges: List[Tuple[int, int]] = []

    # Collect all node ids that appear
    all_node_ids: set = set()
    for (i0, j0, u, v) in bifiltration.get('birth_records', []):
        all_node_ids.add(u)
        all_node_ids.add(v)
    n_nodes = max(all_node_ids) + 1 if all_node_ids else 0

    if n_nodes == 0:
        return [], []

    for dim in dims:
        # level_bars[j] = list of (birth_norm, death_norm) kept after filtering
        level_bars: List[List[Tuple[float, float]]] = []

        for j in range(K_grid):
            # Build 1-param filtration along time axis at curvature level j
            edge_seq: List[List[Tuple[int, int]]] = [[] for _ in range(T_grid)]
            for (i0, j0, u, v) in bifiltration.get('birth_records', []):
                if j0 <= j:
                    edge_seq[i0].append((u, v))

            # Compute persistence
            if _GUDHI_AVAILABLE:
                bars_raw = _compute_ph_gudhi(edge_seq, n_nodes, hom_dim=dim)
            else:
                bars_raw = _compute_ph_numpy(edge_seq, n_nodes, hom_dim=dim)

            # Normalise step indices to [0,1] using grid size
            T_norm = max(T_grid - 1, 1)
            bars_norm = [
                (b / T_norm, min(d / T_norm, 1.0))
                for b, d in bars_raw
            ]

            # Filter by min_persistence
            bars_filtered = [(b, d) for b, d in bars_norm if d - b >= min_persistence]

            # Keep only top max_bars_per_level most persistent bars
            bars_filtered.sort(key=lambda x: x[1] - x[0], reverse=True)
            bars_filtered = bars_filtered[:max_bars_per_level]

            level_bars.append(bars_filtered)

        # Build Graphcode nodes: 8-dim feature vector per bar
        # [birth, death, curv_level, persistence,
        #  midpoint, log_pers, birth_ratio, curv_pers_interaction]
        # level_offset[j] = index in all_nodes where level j starts
        level_offset = []
        for j, bars in enumerate(level_bars):
            level_offset.append(len(all_nodes))
            curv_norm = j / max(K_grid - 1, 1)
            for birth, death in bars:
                pers = death - birth
                midpoint    = (birth + death) / 2.0
                log_pers    = float(np.log1p(pers * 10.0) / np.log1p(10.0))
                birth_ratio = birth / (death + 1e-8)   # 0=born early, →1=born late
                curv_pers   = curv_norm * pers          # interaction term
                all_nodes.append([birth, death, curv_norm, pers,
                                   midpoint, log_pers, birth_ratio, curv_pers])

        # Build Graphcode edges via bar matching across consecutive levels
        for j in range(K_grid - 1):
            bars_j  = level_bars[j]
            bars_j1 = level_bars[j + 1]
            if not bars_j or not bars_j1:
                continue

            connections = _match_bars(bars_j, bars_j1)

            offset_j  = level_offset[j]
            offset_j1 = level_offset[j + 1]
            for a_local, b_local in connections:
                node_a = offset_j  + a_local
                node_b = offset_j1 + b_local
                if node_a < len(all_nodes) and node_b < len(all_nodes):
                    all_edges.append((node_a, node_b))

    # Features are already in [0,1] by construction — no per-graph re-scaling.
    # Deduplicate edges
    edge_set = set()
    deduped_edges = []
    for a, b in all_edges:
        key = (min(a, b), max(a, b))
        if key not in edge_set:
            edge_set.add(key)
            deduped_edges.append((a, b))

    return all_nodes, deduped_edges


# ---------------------------------------------------------------------------
# Fallback: CROCKER plot vectorization (Section 17)
# ---------------------------------------------------------------------------

def compute_crocker(
    bifiltration: dict,
    T_grid: int,
    K_grid: int,
    hom_dim: int = 1,
) -> np.ndarray:
    """
    Fallback vectorization: CROCKER plot (2D Betti number matrix).
    Returns array of shape (K_grid, T_grid) where entry [j, i] =
    beta_{hom_dim} at grid point (i, j).
    """
    all_node_ids: set = set()
    for (i0, j0, u, v) in bifiltration.get('birth_records', []):
        all_node_ids.add(u)
        all_node_ids.add(v)
    n_nodes = max(all_node_ids) + 1 if all_node_ids else 0

    crocker = np.zeros((K_grid, T_grid), dtype=np.float32)

    for j in range(K_grid):
        edge_seq: List[List[Tuple[int, int]]] = [[] for _ in range(T_grid)]
        for (i0, j0, u, v) in bifiltration.get('birth_records', []):
            if j0 <= j:
                edge_seq[i0].append((u, v))

        # Compute cumulative edges at each step and count Betti number
        adj: dict = {}
        parent = list(range(n_nodes))
        rank = [0] * n_nodes

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        n_components = 0
        n_cycles = 0
        seen_edges = set()
        seen_nodes = set()

        for i in range(T_grid):
            for u, v in edge_seq[i]:
                if u not in seen_nodes:
                    seen_nodes.add(u)
                    n_components += 1
                if v not in seen_nodes:
                    seen_nodes.add(v)
                    n_components += 1

                key = (min(u, v), max(u, v))
                if key not in seen_edges:
                    seen_edges.add(key)
                    ru, rv = find(u), find(v)
                    if ru == rv:
                        n_cycles += 1
                    else:
                        n_components -= 1
                        if rank[ru] < rank[rv]:
                            ru, rv = rv, ru
                        parent[rv] = ru
                        if rank[ru] == rank[rv]:
                            rank[ru] += 1

            if hom_dim == 0:
                crocker[j, i] = n_components
            else:
                crocker[j, i] = n_cycles

    return crocker
