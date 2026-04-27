"""
src/gin_classifier.py
GIN classifier that operates on Graphcode graphs.

Architecture (from guide Section 7.4):
  - 3 GIN layers, hidden dim 64
  - Global mean pooling + global max pooling (concatenated)
  - 2-layer MLP classifier
  - Dropout 0.3 between layers
  - BatchNorm after each GIN layer

Node features: [birth_time, death_time, curvature_level, persistence]
               all normalised to [0,1] → node_feat_dim = 4.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch


class MLP(nn.Module):
    """Two-layer MLP used as GINConv internal network."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GINClassifier(nn.Module):
    """
    GIN-based classifier for Graphcode graphs, with Jumping Knowledge (JK-max).

    Architecture:
      - Input projection: node_feat_dim → hidden_dim
      - num_layers GIN convolutions, each followed by BatchNorm + ReLU + Dropout
      - JK-max: element-wise max across ALL layer outputs before pooling
        (prevents over-smoothing on path-like Graphcode graphs)
      - Global mean + max pooling → 2*hidden_dim
      - Optional concat of precomputed global graph features
      - 2-layer MLP classifier

    Parameters
    ----------
    node_feat_dim : int       – input feature dimension (8 by default)
    hidden_dim    : int       – GIN hidden / output dim
    num_layers    : int       – number of GIN conv layers
    num_classes   : int       – number of output classes
    dropout       : float     – dropout probability
    eps           : float     – GIN epsilon (0 = learnable)
    use_crocker   : bool      – if True, input is a flattened CROCKER matrix
    crocker_shape : tuple     – (K_grid, T_grid) only used when use_crocker=True
    global_feat_dim : int     – size of precomputed graph-level feature vector
    use_jk : bool             – if False, use only last GIN layer (disables JK-max)
    """

    def __init__(
        self,
        node_feat_dim: int = 8,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_classes: int = 2,
        dropout: float = 0.3,
        eps: float = 0.0,
        use_crocker: bool = False,
        crocker_shape: tuple = (20, 30),
        global_feat_dim: int = 0,
        use_jk: bool = True,
    ):
        super().__init__()
        self.use_crocker = use_crocker
        self.dropout = dropout
        self.global_feat_dim = global_feat_dim
        self.use_jk = use_jk

        if use_crocker:
            flat_dim = crocker_shape[0] * crocker_shape[1] * 2  # H0 + H1
            self.crocker_head = nn.Sequential(
                nn.Linear(flat_dim, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
            return

        # Input projection
        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)

        # GIN layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLP(hidden_dim, hidden_dim)
            self.convs.append(GINConv(mlp, eps=eps, train_eps=(eps == 0.0)))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # JK-max: element-wise max across layer outputs → still hidden_dim
        # mean+max pooling → 2*hidden_dim; optionally concat global features
        pool_dim = 2 * hidden_dim + global_feat_dim

        # Classification head (2-layer MLP)
        self.classifier = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        data: Data | Batch,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        data : PyG Data or Batch object with:
               data.x          – node features  (N, node_feat_dim)
               data.edge_index – (2, E)
               data.batch      – (N,) batch vector

        Returns
        -------
        logits : (batch_size, num_classes)
        """
        if self.use_crocker:
            # data.x is already the flattened CROCKER vector per graph
            return self.crocker_head(data.x)

        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Handle empty graphs (no nodes)
        if x is None or x.shape[0] == 0:
            # Return zero logits for the batch
            bs = int(batch.max().item()) + 1 if batch is not None and len(batch) > 0 else 1
            return torch.zeros(bs, self.classifier[-1].out_features,
                               device=edge_index.device if edge_index is not None else 'cpu')

        # Input projection
        x = F.relu(self.input_proj(x))

        # GIN layers with JK-max: collect all layer outputs
        x_layers = []
        for conv, bn in zip(self.convs, self.bns):
            if edge_index is not None and edge_index.shape[1] > 0:
                x = conv(x, edge_index)
            else:
                # No edges: just apply MLP (identity message passing)
                x = conv.nn(x)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x_layers.append(x)

        # JK-max: element-wise max across all layers (fixes over-smoothing on
        # path-like Graphcode graphs; pool_dim stays 2*hidden_dim)
        if self.use_jk:
            x_jk = torch.stack(x_layers, dim=0).max(dim=0).values  # (N, hidden_dim)
        else:
            x_jk = x_layers[-1]  # last layer only (no jumping knowledge)

        # Global pooling: mean + max concatenated
        mean_pool = global_mean_pool(x_jk, batch)
        max_pool  = global_max_pool(x_jk, batch)
        graph_repr = torch.cat([mean_pool, max_pool], dim=-1)

        # Optionally concatenate global graph-level features stored in data.gf
        if self.global_feat_dim > 0 and hasattr(data, 'gf') and data.gf is not None:
            graph_repr = torch.cat([graph_repr, data.gf], dim=-1)

        return self.classifier(graph_repr)

    def predict(self, data: Data | Batch) -> torch.Tensor:
        """Return class probabilities (softmax)."""
        return F.softmax(self.forward(data), dim=-1)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_gin(config: dict, num_classes: int) -> GINClassifier:
    """Construct GINClassifier from config dict."""
    return GINClassifier(
        node_feat_dim=config.get('node_feat_dim', 8),
        hidden_dim=config.get('gin_hidden', 64),
        num_layers=config.get('gin_layers', 3),
        num_classes=num_classes,
        dropout=config.get('gin_dropout', 0.3),
        eps=config.get('gin_eps', 0.0),
        use_crocker=config.get('use_crocker', False),
        crocker_shape=(
            config.get('K_grid', 20),
            config.get('T_grid', 30),
        ),
        global_feat_dim=config.get('global_feat_dim', 0),
        use_jk=config.get('use_jk', True),
    )
