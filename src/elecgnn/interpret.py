"""Interpretation of trained mixture models.

* Integrated gradients (zero-feature baseline) give a per-atom attribution for every
  component graph; atom attributions of one graph sum approximately to the change in
  the (transformed) prediction when that graph's features are switched on.
* For GAT encoders, attention in the final layer is reported relative to uniform
  attention. Raw softmax weights scale as 1 / (number of incoming edges), so element- or
  group-averaged raw attention mostly reflects atom degree. ``attn_ratio`` is the
  attention an atom receives from its neighbours divided by what it would receive under
  uniform attention (1 = uniform); ``entropy_ratio`` is the entropy of an atom's incoming
  attention divided by the uniform entropy (1 = uniform, lower = focused).
* Embeddings: k-nearest-neighbour R^2 of the target from model descriptors, compared with
  a randomly initialized model of the same architecture (random GNNs already cluster
  molecules by chemistry, so structure in an embedding is meaningful only above this
  control), plus t-SNE coordinates for both.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset

from .data import collate
from .featurize import PAD_TOKEN
from .models import GATEncoder, MixtureModel

C_HYBRID = {Chem.rdchem.HybridizationType.SP: "C-sp", Chem.rdchem.HybridizationType.SP2: "C-sp2"}


def atom_groups(smiles: str) -> list[tuple[str, str]]:
    """(element, chemical-environment label) per atom, in the featurization atom order."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    out = []
    for a in mol.GetAtoms():
        el, nbrs = a.GetSymbol(), [n.GetSymbol() for n in a.GetNeighbors()]
        if el == "O":
            label = "O-carbonyl" if any(b.GetBondTypeAsDouble() == 2 for b in a.GetBonds()) else "O-ether"
            if a.GetFormalCharge() < 0:
                label = "O-anionic"
        elif el == "C":
            label = "C-F" if "F" in nbrs else C_HYBRID.get(a.GetHybridization(), "C-sp3")
        else:
            label = el
        out.append((el, label))
    return out


def _loader(ds, idx, batch_size: int = 64) -> DataLoader:
    return DataLoader(Subset(ds, list(idx)), batch_size=batch_size, shuffle=False, collate_fn=collate)


def _to(batch, device):
    slots, conc, scal, extra, y = batch
    return [g.to(device) for g in slots], conc.to(device), scal.to(device), extra.to(device), y.to(device)


def integrated_gradients(model: MixtureModel, batch, steps: int = 32) -> list[torch.Tensor]:
    """Per-node IG attribution for every slot of a collated batch (zero baseline)."""
    slots, conc, scal, extra, _ = batch
    out = []
    for j, g in enumerate(slots):
        x0, total = g.x.detach(), torch.zeros_like(g.x)
        for a in torch.linspace(1 / steps, 1, steps, device=x0.device):
            xs = (a * x0).clone().requires_grad_(True)
            gj = g.clone()
            gj.x = xs
            pred = model([gj if k == j else s for k, s in enumerate(slots)], conc, scal, extra).sum()
            total += torch.autograd.grad(pred, xs)[0]
        out.append((x0 * total / steps).sum(dim=1))
    return out


@torch.no_grad()
def attention_ratios(encoder: GATEncoder, g) -> tuple[np.ndarray, np.ndarray]:
    """Per-node (attention received / uniform) and (incoming entropy / uniform entropy)."""
    _, (ei, alpha) = encoder(g.x, g.edge_index, g.edge_attr, return_attention=True)
    src, dst, a = ei[0], ei[1], alpha.view(-1)
    n, dev = g.x.shape[0], g.x.device
    n_in = torch.bincount(dst, minlength=n).float()
    off = src != dst
    received = torch.zeros(n, device=dev).index_add_(0, src[off], a[off])
    expected = torch.zeros(n, device=dev).index_add_(0, src[off], 1.0 / n_in[dst[off]])
    ent = torch.zeros(n, device=dev).index_add_(0, dst, -a * torch.log(a.clamp(min=1e-12)))
    ratio = torch.where(expected > 0, received / expected, torch.full_like(received, float("nan")))
    ent_ratio = torch.where(n_in > 1, ent / torch.log(n_in), torch.full_like(ent, float("nan")))
    return ratio.cpu().numpy(), ent_ratio.cpu().numpy()


def atom_attributions(model: MixtureModel, ds, table, idx, slot_names: list[str], seed: int, split: str,
                      steps: int = 32, device: str | torch.device = "cpu") -> pd.DataFrame:  # fmt: skip
    """Long table: one row per atom of every real component of the rows ``idx``."""
    model.to(device).eval()
    rows, pos = [], 0
    idx = np.asarray(idx)
    for batch in _loader(ds, idx):
        batch = _to(batch, device)
        ig = integrated_gradients(model, batch, steps)
        slots = batch[0]
        n_graphs = slots[0].num_graphs
        for j, g in enumerate(slots):
            att = attention_ratios(model.encoder(j), g) if model.kind == "gat" else (None, None)
            ig_j = ig[j].detach().cpu().numpy()
            for b in range(n_graphs):
                r = idx[pos + b]
                smi = table.smiles[r, j]
                if table.conc[r, j] <= 0 or smi == PAD_TOKEN:
                    continue
                nodes = (g.batch == b).nonzero().view(-1).cpu().numpy()
                groups = atom_groups(smi)
                if len(groups) != len(nodes):  # featurization fell back; skip
                    continue
                for k, node in enumerate(nodes):
                    rows.append({
                        "row_id": table.row_id[r], "seed": seed, "split": split, "slot": slot_names[j],
                        "smiles": smi, "atom": k, "element": groups[k][0], "group": groups[k][1],
                        "ig": float(ig_j[node]),
                        "attn_ratio": float(att[0][node]) if att[0] is not None else np.nan,
                        "entropy_ratio": float(att[1][node]) if att[1] is not None else np.nan,
                    })  # fmt: skip
        pos += n_graphs
    return pd.DataFrame(rows)


@torch.no_grad()
def descriptors(model: MixtureModel, ds, idx, device: str | torch.device = "cpu") -> np.ndarray:
    model.to(device).eval()
    return np.concatenate([model.descriptor(*_to(b, device)[:4]).cpu().numpy() for b in _loader(ds, idx)])


def knn_r2(x: np.ndarray, y: np.ndarray, k: int = 5, seed: int = 0) -> float:
    pipe = make_pipeline(StandardScaler(), KNeighborsRegressor(k))
    cv = KFold(5, shuffle=True, random_state=seed)
    return float(cross_val_score(pipe, x, y, cv=cv, scoring="r2").mean())


def tsne(x: np.ndarray, seed: int = 0) -> np.ndarray:
    perplexity = min(30, max(5, (len(x) - 1) // 4))
    return TSNE(2, perplexity=perplexity, init="pca", random_state=seed).fit_transform(
        StandardScaler().fit_transform(x)
    )
