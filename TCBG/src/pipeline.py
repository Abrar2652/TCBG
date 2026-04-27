"""
src/pipeline.py
End-to-end pipeline: temporal graph -> Graphcode PyG Data object.

  temporal_edges (list of (u,v,t))
      |
      v  compute_forman_ricci()
  edge_curvatures (list of (u,v,t,kappa))
      |
      v  build_bifiltration()
  bifiltration grid
      |
      v  compute_graphcode()
  (gc_nodes, gc_edges)
      |
      v  PyG Data(x=..., edge_index=...)
"""

from __future__ import annotations

from typing import List, Tuple, Optional
import numpy as np
import torch

try:
    from torch_geometric.data import Data
except ImportError:
    # Minimal stub so curvature/bifiltration/graphcode still importable without PyG
    class Data:
        def __init__(self, x=None, edge_index=None, num_nodes=None, y=None):
            self.x = x
            self.edge_index = edge_index
            self.num_nodes = num_nodes
            self.y = y

from .curvature import compute_forman_ricci, auto_epsilon
from .bifiltration import build_bifiltration
from .graphcode import compute_graphcode, compute_crocker


def _global_features(
    temporal_edges: List[Tuple[int, int, float]],
    edge_curvatures: list,
) -> torch.Tensor:
    """
    Compute 13 graph-level summary features to augment the GIN embedding.

    Features (all normalised to reasonable scale):
      0: log(num_temporal_edges)
      1: log(num_unique_node_pairs)
      2: log(num_nodes)
      3: mean curvature
      4: std curvature (tanh-normalised)
      5: fraction of edges with kappa > 0  (community-interior edges)
      6: fraction of edges with kappa < 0  (bridge/bottleneck edges)
      7: temporal burstiness  (std_t / mean_t)
      8: temporal range normalised by std
      --- structural graph features ---
      9:  edge density  E_unique / (N*(N-1)/2)  -- how dense the contact net is
     10:  degree entropy  -sum(p_i log p_i) / log(N)  -- homogeneity of connectivity
     11:  repeat contact fraction  fraction of unique pairs seen >1 time -- regularity
     12:  mean curvature volatility (std of per-edge curvature change over time)
    """
    if not temporal_edges:
        return torch.zeros(13)

    n_edges = len(temporal_edges)
    pair_counts: dict = {}
    for u, v, t in temporal_edges:
        key = (min(u, v), max(u, v))
        pair_counts[key] = pair_counts.get(key, 0) + 1
    unique_pairs = len(pair_counts)
    nodes = set(u for u, v, t in temporal_edges) | set(v for u, v, t in temporal_edges)
    n_nodes = len(nodes)
    timestamps = [t for u, v, t in temporal_edges]

    kappas = np.array([k for u, v, t, k in edge_curvatures]) if edge_curvatures else np.zeros(1)
    mean_k = float(kappas.mean())
    std_k  = float(kappas.std()) + 1e-6
    frac_pos = float((kappas > 0).mean())
    frac_neg = float((kappas < 0).mean())

    ts = np.array(timestamps)
    mean_t = ts.mean() + 1e-6
    std_t  = ts.std()  + 1e-6
    burstiness = std_t / mean_t
    t_range_norm = (ts.max() - ts.min()) / (std_t + 1e-6)

    # --- structural features ---
    max_pairs = n_nodes * (n_nodes - 1) / 2 if n_nodes > 1 else 1
    edge_density = unique_pairs / max_pairs

    # Degree entropy of aggregated graph (O(E))
    deg: dict = {}
    for u, v, t in temporal_edges:
        deg[u] = deg.get(u, 0) + 1
        deg[v] = deg.get(v, 0) + 1
    deg_vals = np.array(list(deg.values()), dtype=float)
    deg_sum = deg_vals.sum() + 1e-8
    p = deg_vals / deg_sum
    deg_entropy = float(-np.sum(p * np.log(p + 1e-12)) / np.log(n_nodes + 1e-6))

    # Repeat contact fraction: pairs contacted more than once
    repeat_frac = float(sum(1 for c in pair_counts.values() if c > 1) / (unique_pairs + 1e-8))

    # Curvature volatility: std of curvature across edges, normalised
    curv_volatility = float(np.tanh(std_k / (abs(mean_k) + 1e-6)))

    # --- Spectral features of aggregated contact graph (O(N^3), fast for N≤200) ---
    # Build adjacency matrix of unique contacts
    node_list = sorted(nodes)
    node_idx  = {v: i for i, v in enumerate(node_list)}
    n = len(node_list)
    if n >= 2:
        A = np.zeros((n, n), dtype=np.float32)
        for key in pair_counts:
            u, v = key
            if u in node_idx and v in node_idx:
                i, j = node_idx[u], node_idx[v]
                A[i, j] = A[j, i] = 1.0
        # Normalised Laplacian L_norm = I - D^{-1/2} A D^{-1/2}
        deg_arr = A.sum(axis=1)
        d_inv_sqrt = np.where(deg_arr > 0, 1.0 / np.sqrt(deg_arr + 1e-8), 0.0)
        # eigenvalues via symmetric eigsh for speed (use only if n is manageable)
        if n <= 300:
            L = np.diag(d_inv_sqrt) @ A @ np.diag(d_inv_sqrt)
            L = np.eye(n, dtype=np.float32) - L
            eigvals = np.linalg.eigvalsh(L)
            eigvals_sorted = np.sort(eigvals)
            fiedler   = float(eigvals_sorted[1]) if n > 1 else 0.0   # algebraic connectivity
            spec_gap  = float(eigvals_sorted[-1] - eigvals_sorted[-2]) if n > 1 else 0.0
        else:
            # Too large: use degree-based proxy
            fiedler  = float(np.min(deg_arr) / (np.mean(deg_arr) + 1e-8))
            spec_gap = float(np.std(deg_arr) / (np.mean(deg_arr) + 1e-8))
    else:
        fiedler  = 0.0
        spec_gap = 0.0

    feats = [
        np.log1p(n_edges)      / 8.0,
        np.log1p(unique_pairs) / 7.0,
        np.log1p(n_nodes)      / 5.0,
        mean_k / 10.0,
        np.tanh(std_k  / 10.0),
        frac_pos,
        frac_neg,
        np.tanh(burstiness),
        np.tanh(t_range_norm / 10.0),
        float(np.tanh(edge_density * 5.0)),    # edge density
        float(np.clip(deg_entropy, 0.0, 1.0)), # degree entropy
        float(repeat_frac),                    # temporal regularity
        curv_volatility,                       # curvature volatility
        float(np.tanh(fiedler * 2.0)),         # Fiedler value (algebraic connectivity)
        float(np.tanh(spec_gap * 2.0)),        # spectral gap
    ]
    return torch.tensor(feats, dtype=torch.float)


def _temporal_spectral_features(
    temporal_edges: List[Tuple[int, int, float]],
) -> torch.Tensor:
    """
    Compute 6 spectral features from the per-timestep graph Laplacians.

    For each unique timestamp t, build the static graph G_t (all edges with
    that timestamp), compute its normalised Laplacian eigenvalues, and
    aggregate statistics ACROSS all timesteps.  This captures the temporal
    evolution of spectral structure — a complementary signal to curvature.

    General (not dataset-specific): works for any discrete-timestamp
    temporal graph.  Most useful for brain networks where connectivity
    topology changes meaningfully over time windows.

    Features:
      0: mean Fiedler value across timesteps  (avg algebraic connectivity)
      1: std  Fiedler value across timesteps  (temporal variability)
      2: mean spectral entropy across timesteps  (topological complexity)
      3: std  spectral entropy across timesteps
      4: mean max-eigenvalue across timesteps  (spectral radius)
      5: std  max-eigenvalue across timesteps
    """
    if not temporal_edges:
        return torch.zeros(6)

    # Group edges by timestamp
    from collections import defaultdict
    ts_edges: dict = defaultdict(list)
    for u, v, t in temporal_edges:
        ts_edges[t].append((u, v))

    timesteps = sorted(ts_edges.keys())
    if len(timesteps) == 0:
        return torch.zeros(6)

    # Collect all node IDs to build consistent matrices
    all_nodes = sorted(set(u for u, v, t in temporal_edges) |
                       set(v for u, v, t in temporal_edges))
    n = len(all_nodes)
    if n < 3 or n > 300:
        # Too small or too large for eigendecomposition — use zeros
        return torch.zeros(6)

    nidx = {v: i for i, v in enumerate(all_nodes)}

    fiedler_vals = []
    entropy_vals = []
    maxeig_vals  = []

    for t in timesteps:
        edges = ts_edges[t]
        if not edges:
            continue
        A = np.zeros((n, n), dtype=np.float32)
        for u, v in edges:
            i, j = nidx[u], nidx[v]
            A[i, j] = A[j, i] = 1.0
        deg = A.sum(axis=1)
        d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg + 1e-8), 0.0)
        L = np.eye(n, dtype=np.float32) - np.diag(d_inv_sqrt) @ A @ np.diag(d_inv_sqrt)
        eigvals = np.linalg.eigvalsh(L)
        eigvals = np.clip(eigvals, 0.0, 2.0)
        eigvals_sorted = np.sort(eigvals)

        fiedler = float(eigvals_sorted[1]) if n > 1 else 0.0
        maxeig  = float(eigvals_sorted[-1])

        # Spectral entropy: H = -sum(p * log p) over eigenvalue probabilities
        ev_sum = eigvals.sum() + 1e-8
        p = eigvals / ev_sum
        entropy = float(-np.sum(p * np.log(p + 1e-12)))
        # Normalise entropy by log(n) to [0,1]
        entropy /= float(np.log(n) + 1e-8)

        fiedler_vals.append(fiedler)
        entropy_vals.append(entropy)
        maxeig_vals.append(maxeig)

    if not fiedler_vals:
        return torch.zeros(6)

    def _safe_tanh(x):
        return float(np.tanh(x))

    feats = [
        _safe_tanh(float(np.mean(fiedler_vals)) * 2.0),
        _safe_tanh(float(np.std(fiedler_vals))  * 5.0),
        float(np.clip(np.mean(entropy_vals), 0.0, 1.0)),
        float(np.clip(np.std(entropy_vals),  0.0, 1.0)),
        _safe_tanh(float(np.mean(maxeig_vals)) / 2.0),
        _safe_tanh(float(np.std(maxeig_vals))  * 5.0),
    ]
    return torch.tensor(feats, dtype=torch.float)


def _betti_sequence_features(
    bifiltration: dict,
    T_grid: int,
    K_grid: int,
) -> torch.Tensor:
    """
    Compute 8 summary statistics from the Betti-0/1 number sequences.

    For each time step t (averaged over all curvature levels k), we compute
    H0 (connected components) and H1 (independent cycles) using Union-Find.
    This captures the RATE of topological change — e.g., how fast components
    merge (H0 decay) and when cycles form (H1 growth).  These are
    complementary to Graphcode node features and are general across all
    dataset types.

    Features:
      0: mean H0 (avg components across all t)
      1: max H0 (peak component count, normalised by n_nodes)
      2: normalised time-of-peak H0  (when are components maximal?)
      3: H0 decay rate  (max_H0 - final_H0) / (max_H0 + 1e-8)
      4: mean H1 (avg cycles across all t)
      5: max H1 (peak cycle count, normalised by n_nodes)
      6: normalised time-of-first-H1  (when does first cycle appear?)
      7: H1 at final step (remaining cycles at t_max, normalised)
    """
    birth_records = bifiltration.get('birth_records', [])
    if not birth_records:
        return torch.zeros(8)

    all_node_ids: set = set()
    for (i0, j0, u, v) in birth_records:
        all_node_ids.add(u)
        all_node_ids.add(v)
    n_nodes = max(all_node_ids) + 1 if all_node_ids else 1

    # For each curvature slice j, compute H0 and H1 sequence along time axis
    # using incremental Union-Find (O(alpha * E * T) total)
    h0_matrix = np.zeros((K_grid, T_grid), dtype=np.float32)
    h1_matrix = np.zeros((K_grid, T_grid), dtype=np.float32)

    for j in range(K_grid):
        parent = list(range(n_nodes))
        rank   = [0] * n_nodes
        seen_edges: set = set()
        seen_nodes: set = set()
        n_comp  = 0
        n_cycle = 0

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        # Build time-ordered edge sequence for this curvature level
        edges_at: dict = {}
        for (i0, j0, u, v) in birth_records:
            if j0 <= j:
                edges_at.setdefault(i0, []).append((u, v))

        for t in range(T_grid):
            for u, v in edges_at.get(t, []):
                if u not in seen_nodes:
                    seen_nodes.add(u)
                    n_comp += 1
                if v not in seen_nodes:
                    seen_nodes.add(v)
                    n_comp += 1
                key = (min(u, v), max(u, v))
                if key not in seen_edges:
                    seen_edges.add(key)
                    ru, rv = find(u), find(v)
                    if ru == rv:
                        n_cycle += 1
                    else:
                        n_comp -= 1
                        if rank[ru] < rank[rv]:
                            ru, rv = rv, ru
                        parent[rv] = ru
                        if rank[ru] == rank[rv]:
                            rank[ru] += 1
            h0_matrix[j, t] = n_comp
            h1_matrix[j, t] = n_cycle

    # Average over all curvature levels
    h0_seq = h0_matrix.mean(axis=0)  # (T_grid,)
    h1_seq = h1_matrix.mean(axis=0)  # (T_grid,)

    n_norm = float(n_nodes) + 1e-8
    mean_h0 = float(h0_seq.mean()) / n_norm
    max_h0  = float(h0_seq.max())  / n_norm
    t_peak_h0 = float(np.argmax(h0_seq)) / max(T_grid - 1, 1)
    h0_decay = float(h0_seq.max() - h0_seq[-1]) / (float(h0_seq.max()) + 1e-8)

    mean_h1 = float(h1_seq.mean()) / n_norm
    max_h1  = float(h1_seq.max())  / n_norm
    first_h1_step = float(np.argmax(h1_seq > 0)) / max(T_grid - 1, 1) if h1_seq.max() > 0 else 1.0
    h1_final = float(h1_seq[-1]) / n_norm

    feats = [
        float(np.tanh(mean_h0)),
        float(np.tanh(max_h0)),
        t_peak_h0,
        float(np.clip(h0_decay, 0.0, 1.0)),
        float(np.tanh(mean_h1)),
        float(np.tanh(max_h1)),
        first_h1_step,
        float(np.tanh(h1_final)),
    ]
    return torch.tensor(feats, dtype=torch.float)


class TCBGPipeline:
    """
    Convert one temporal graph into a Graphcode PyG Data object.

    Parameters
    ----------
    config : dict with keys:
        T_grid         : int   (default 30)
        K_grid         : int   (default 20)
        epsilon        : float | 'auto'
        hom_dim        : int or list of int  ([0,1])
        min_persistence: float (0.01)
        dataset_type   : str   ('social'|'brain'|'traffic')
        use_crocker    : bool  (False)
        num_timesteps  : int   (for auto epsilon, optional)
    """

    def __init__(self, config: dict):
        self.T_grid = int(config.get('T_grid', 30))
        self.K_grid = int(config.get('K_grid', 20))
        self.epsilon = config.get('epsilon', 'auto')
        self.hom_dim = config.get('hom_dim', [0, 1])
        self.min_persistence = float(config.get('min_persistence', 0.05))
        self.max_bars_per_level = int(config.get('max_bars_per_level', 30))
        self.dataset_type = config.get('dataset_type', 'social')
        self.use_crocker = bool(config.get('use_crocker', False))
        self.num_timesteps = config.get('num_timesteps', None)

    def process_graph(
        self,
        temporal_edges: List[Tuple[int, int, float]],
        nodes: Optional[List[int]] = None,
    ) -> Data:
        """
        Convert one temporal graph to a Graphcode PyG Data object.

        Parameters
        ----------
        temporal_edges : list of (u, v, t)
        nodes : optional list of node ids (inferred from edges if None)

        Returns
        -------
        torch_geometric.data.Data with:
            x          : (N_bars, 4) float32  node features
            edge_index : (2, E_gc) int64       Graphcode edges
            num_nodes  : int
        """
        if not temporal_edges:
            return Data(x=torch.zeros(1, 8), edge_index=torch.zeros(2, 0, dtype=torch.long))

        # --- 1. Compute Forman-Ricci curvature ---
        epsilon = self.epsilon
        if epsilon == 'auto':
            epsilon = auto_epsilon(
                temporal_edges,
                dataset_type=self.dataset_type,
                num_timesteps=self.num_timesteps,
            )

        edge_curvatures = compute_forman_ricci(temporal_edges, float(epsilon))

        if not edge_curvatures:
            return Data(x=torch.zeros(1, 8), edge_index=torch.zeros(2, 0, dtype=torch.long))

        # --- 2. Build bifiltration ---
        bifiltration = build_bifiltration(
            edge_curvatures,
            T_grid=self.T_grid,
            K_grid=self.K_grid,
        )

        if self.use_crocker:
            # CROCKER fallback: return flattened Betti matrix as "node features"
            crocker_h0 = compute_crocker(bifiltration, self.T_grid, self.K_grid, hom_dim=0)
            crocker_h1 = compute_crocker(bifiltration, self.T_grid, self.K_grid, hom_dim=1)
            flat = np.concatenate([crocker_h0.flatten(), crocker_h1.flatten()])
            x = torch.tensor(flat, dtype=torch.float).unsqueeze(0)
            return Data(x=x, edge_index=torch.zeros(2, 0, dtype=torch.long))

        # --- 3. Compute Graphcode ---
        gc_nodes, gc_edges = compute_graphcode(
            bifiltration,
            T_grid=self.T_grid,
            K_grid=self.K_grid,
            hom_dim=self.hom_dim,
            min_persistence=self.min_persistence,
            max_bars_per_level=self.max_bars_per_level,
        )

        # --- 4. Build PyG Data ---
        if not gc_nodes:
            x = torch.zeros(1, 8, dtype=torch.float)
            edge_index = torch.zeros(2, 0, dtype=torch.long)
        else:
            x = torch.tensor(gc_nodes, dtype=torch.float)
            if gc_edges:
                ei = torch.tensor(gc_edges, dtype=torch.long).t().contiguous()
                edge_index = torch.cat([ei, ei.flip(0)], dim=1)
            else:
                edge_index = torch.zeros(2, 0, dtype=torch.long)

        # --- 5. Global graph-level features ---
        # 15-dim structural/curvature + 6-dim temporal spectral = 21-dim total
        gf_struct  = _global_features(temporal_edges, edge_curvatures)
        gf_spectral = _temporal_spectral_features(temporal_edges)
        gf = torch.cat([gf_struct, gf_spectral], dim=0)  # (21,)

        return Data(x=x, edge_index=edge_index, num_nodes=x.shape[0],
                    gf=gf.unsqueeze(0))

    def process_dataset(
        self,
        graphs: List[Tuple[List[Tuple[int, int, float]], Optional[List[int]], int]],
        verbose: bool = True,
    ) -> List[Data]:
        """
        Process a full dataset.

        Parameters
        ----------
        graphs : list of (temporal_edges, nodes, label)
        verbose : print progress

        Returns
        -------
        list of PyG Data objects with data.y set to label
        """
        from tqdm import tqdm
        results = []
        iterator = tqdm(graphs, desc='Processing graphs') if verbose else graphs
        for temporal_edges, nodes, label in iterator:
            data = self.process_graph(temporal_edges, nodes)
            data.y = torch.tensor([label], dtype=torch.long)
            results.append(data)
        return results
