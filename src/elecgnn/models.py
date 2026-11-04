"""Graph encoders and the concentration-weighted mixture model.

The architecture matches the original work: each component graph is encoded (four
GATv2Conv layers with four heads and an ELU, then a single-head output layer; or the
GCN equivalent without edge features), mean-pooled, scaled by its mole fraction,
and concatenated with the raw mole fractions (and optional graph-level scalars)
before a three-layer MLP head.

``share_encoder=False`` reproduces the original work (one encoder per slot);
``share_encoder=True`` applies one encoder to every slot. ``pooling="sum"`` keeps molecular
size information that mean pooling discards.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATv2Conv, GCNConv, global_add_pool, global_mean_pool

from .featurize import EDGE_DIM, NODE_DIM


class GATEncoder(nn.Module):
    def __init__(self, hidden: int, num_layers: int = 4, heads: int = 4, in_dim: int = NODE_DIM) -> None:
        super().__init__()
        self.input_norm = nn.BatchNorm1d(in_dim)
        self.layers = nn.ModuleList(
            [GATv2Conv(in_dim, hidden, heads=heads, edge_dim=EDGE_DIM, concat=True)]
            + [GATv2Conv(hidden * heads, hidden, heads=heads, edge_dim=EDGE_DIM, concat=True)
               for _ in range(num_layers - 1)]
        )  # fmt: skip
        self.out_layer = GATv2Conv(hidden * heads, hidden, heads=1, edge_dim=EDGE_DIM, concat=False)

    def forward(self, x, edge_index, edge_attr, return_attention: bool = False):
        x = self.input_norm(x)
        for layer in self.layers:
            x = F.elu(layer(x, edge_index, edge_attr))
        if return_attention:
            return self.out_layer(x, edge_index, edge_attr, return_attention_weights=True)
        return self.out_layer(x, edge_index, edge_attr)


class GCNEncoder(nn.Module):
    """GCN counterpart: same depth and width, no edge features, no attention.

    Graphs already contain explicit self-loops; ``GCNConv`` adds its own, as in the
    original scripts.
    """

    def __init__(self, hidden: int, num_layers: int = 4, in_dim: int = NODE_DIM) -> None:
        super().__init__()
        self.input_norm = nn.BatchNorm1d(in_dim)
        self.layers = nn.ModuleList(
            [GCNConv(in_dim, hidden)] + [GCNConv(hidden, hidden) for _ in range(num_layers - 1)]
        )
        self.out_layer = GCNConv(hidden, hidden)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.input_norm(x)
        for layer in self.layers:
            x = F.elu(layer(x, edge_index))
        return self.out_layer(x, edge_index)


def make_encoder(kind: str, hidden: int, in_dim: int = NODE_DIM) -> nn.Module:
    if kind == "gat":
        return GATEncoder(hidden, in_dim=in_dim)
    if kind == "gcn":
        return GCNEncoder(hidden, in_dim=in_dim)
    raise ValueError(f"unknown encoder: {kind}")


class MixtureModel(nn.Module):
    def __init__(self, kind: str, n_slots: int, hidden1: int = 64, hidden2: int = 128,
                 scalar_dim: int = 0, n_out: int = 1, share_encoder: bool = False,
                 n_extra: int = 0, pooling: str = "mean", node_dim: int = NODE_DIM) -> None:  # fmt: skip
        super().__init__()
        if pooling not in ("mean", "sum"):
            raise ValueError(f"unknown pooling: {pooling}")
        self.kind, self.n_slots, self.share_encoder, self.pooling = kind, n_slots, share_encoder, pooling
        n_enc = 1 if share_encoder else n_slots
        self.encoders = nn.ModuleList([make_encoder(kind, hidden1, node_dim) for _ in range(n_enc)])
        d_in = hidden1 * n_slots + n_slots + scalar_dim * n_slots + n_extra
        self.head = nn.Sequential(
            nn.Linear(d_in, hidden2), nn.LayerNorm(hidden2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden2, hidden2 // 2), nn.ReLU(), nn.Linear(hidden2 // 2, n_out),
        )  # fmt: skip
        self.head.apply(_xavier)

    def encoder(self, slot: int) -> nn.Module:
        return self.encoders[0 if self.share_encoder else slot]

    def descriptor(self, slots, conc: torch.Tensor, scalars: torch.Tensor,
                   extra: torch.Tensor | None = None) -> torch.Tensor:  # fmt: skip
        """Concentration-weighted pooled embeddings, mole fractions, scalars and row features."""
        if self.share_encoder:
            node_h = self._encode_shared(slots)
        else:
            node_h = [self.encoders[j](g.x, g.edge_index, g.edge_attr) for j, g in enumerate(slots)]
        pool = global_add_pool if self.pooling == "sum" else global_mean_pool
        pooled = [pool(h, g.batch, size=conc.shape[0]) * conc[:, j : j + 1]
                  for j, (h, g) in enumerate(zip(node_h, slots, strict=True))]  # fmt: skip
        parts = [*pooled, conc]
        if scalars.shape[-1] > 0:
            parts.append(scalars.flatten(1))
        if extra is not None and extra.shape[-1] > 0:
            parts.append(extra)
        return torch.cat(parts, dim=1)

    def _encode_shared(self, slots) -> list[torch.Tensor]:
        """Run the shared encoder once on all slots together.

        Encoding slots one call at a time would give BatchNorm per-slot statistics in
        training but blended running statistics at evaluation (e.g. solvent and anion
        graphs), which breaks evaluation. One joint call keeps both modes consistent.
        """
        sizes = [g.x.shape[0] for g in slots]
        offsets = torch.cumsum(torch.tensor([0, *sizes[:-1]], device=slots[0].x.device), 0)
        x = torch.cat([g.x for g in slots])
        edge_index = torch.cat([g.edge_index + off for g, off in zip(slots, offsets, strict=True)], dim=1)
        edge_attr = torch.cat([g.edge_attr for g in slots])
        return list(torch.split(self.encoders[0](x, edge_index, edge_attr), sizes))

    def forward(self, slots, conc: torch.Tensor, scalars: torch.Tensor,
                extra: torch.Tensor | None = None) -> torch.Tensor:  # fmt: skip
        return self.head(self.descriptor(slots, conc, scalars, extra))

    def load_encoder_weights(self, state_dict: dict) -> None:
        """Initialize every encoder from one pretrained encoder (``strict=False``, as originally)."""
        for enc in self.encoders:
            enc.load_state_dict(state_dict, strict=False)


def _xavier(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
