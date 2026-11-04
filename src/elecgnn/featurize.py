"""Molecular graph featurization.

Reproduces the original featurization exactly:

* 26-dim node features: element one-hot (18), Pauling electronegativity, Gasteiger
  charge, hybridization one-hot (SP, SP2, SP3, other), in-ring, aromatic.
* 2-dim edge features: bond order (1, 2, 3, 1.5) and bond length (Å) from an
  ETKDGv2 + MMFF geometry (fixed seed); self-loops carry (0, 0).

Padding slots and SMILES that cannot be embedded fall back to a single sp3 oxygen
node, as in the original scripts.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdPartialCharges
from torch_geometric.data import Data

RDLogger.DisableLog("rdApp.*")
log = logging.getLogger(__name__)

ELEMENTS = ["H", "Li", "B", "C", "N", "O", "F", "Al", "Si", "P", "S", "Cl",
            "As", "Br", "Rb", "Ag", "Cs", "Pt"]  # fmt: skip
ELEM_IDX = {e: i for i, e in enumerate(ELEMENTS)}
ELECTRONEGATIVITY = {
    "H": 2.20, "Li": 0.98, "B": 2.04, "C": 2.55, "N": 3.04, "O": 3.44, "F": 3.98,
    "Al": 1.61, "Si": 1.90, "P": 2.19, "S": 2.58, "Cl": 3.16, "As": 2.18, "Br": 2.96,
    "Rb": 0.82, "Ag": 3.16, "Cs": 0.79, "Pt": 2.28,
}  # fmt: skip
HYBRIDIZATION_IDX = {
    Chem.rdchem.HybridizationType.SP: 0,
    Chem.rdchem.HybridizationType.SP2: 1,
    Chem.rdchem.HybridizationType.SP3: 2,
}
BOND_ORDER = {
    Chem.rdchem.BondType.SINGLE: 1.0,
    Chem.rdchem.BondType.DOUBLE: 2.0,
    Chem.rdchem.BondType.TRIPLE: 3.0,
    Chem.rdchem.BondType.AROMATIC: 1.5,
}
NODE_DIM = len(ELEMENTS) + 1 + 1 + 4 + 1 + 1  # 26 (standard features)
EXTRA_ATOM_FEATURES = ["degree", "num_H", "formal_charge", "smallest_ring"]
NODE_DIM_EXTENDED = NODE_DIM + len(EXTRA_ATOM_FEATURES)  # 30


def node_dim(extended: bool = False) -> int:
    return NODE_DIM_EXTENDED if extended else NODE_DIM


EDGE_DIM = 2
PAD_TOKEN = "O"  # original LCE files pad empty slots with "O" at zero concentration


def _embed(smiles: str, seed: int) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv2()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise ValueError(f"3D embedding failed: {smiles}")
    if AllChem.MMFFOptimizeMolecule(mol) == -1:
        log.warning("MMFF parameters unavailable for %s; using the ETKDG geometry", smiles)
    rdPartialCharges.ComputeGasteigerCharges(mol)
    return mol


def _node_features(mol: Chem.Mol, extended: bool = False) -> torch.Tensor:
    x = np.zeros((mol.GetNumAtoms(), node_dim(extended)), dtype=np.float32)
    ring_info = mol.GetRingInfo()
    for i, atom in enumerate(mol.GetAtoms()):
        sym = atom.GetSymbol()
        if sym not in ELEM_IDX:
            raise ValueError(f"unsupported element: {sym}")
        x[i, ELEM_IDX[sym]] = 1.0
        x[i, 18] = ELECTRONEGATIVITY[sym]
        q = atom.GetDoubleProp("_GasteigerCharge")
        x[i, 19] = q if np.isfinite(q) else 0.0
        x[i, 20 + HYBRIDIZATION_IDX.get(atom.GetHybridization(), 3)] = 1.0
        x[i, 24] = float(atom.IsInRing())
        x[i, 25] = float(atom.GetIsAromatic())
        if extended:  # explicit hydrogens are graph nodes, so degree includes them
            rings = [len(r) for r in ring_info.AtomRings() if i in r]
            x[i, 26] = atom.GetDegree()
            x[i, 27] = atom.GetTotalNumHs(includeNeighbors=True)
            x[i, 28] = atom.GetFormalCharge()
            x[i, 29] = min(rings) if rings else 0.0
    return torch.from_numpy(x)


def _edges(mol: Chem.Mol) -> tuple[torch.Tensor, torch.Tensor]:
    pos = mol.GetConformer().GetPositions()
    src, dst, attr = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = [BOND_ORDER.get(bond.GetBondType(), 1.0), float(np.linalg.norm(pos[i] - pos[j]))]
        src += [i, j]
        dst += [j, i]
        attr += [feat, feat]
    for i in range(mol.GetNumAtoms()):
        src.append(i)
        dst.append(i)
        attr.append([0.0, 0.0])
    return (torch.tensor([src, dst], dtype=torch.long), torch.tensor(attr, dtype=torch.float))


def fallback_graph(extended: bool = False) -> Data:
    """Single sp3 oxygen node with a self-loop (original padding/fallback graph)."""
    x = torch.zeros((1, node_dim(extended)))
    x[0, ELEM_IDX["O"]] = 1.0
    x[0, 18] = ELECTRONEGATIVITY["O"]
    x[0, 22] = 1.0
    return Data(x=x, edge_index=torch.zeros((2, 1), dtype=torch.long),
                edge_attr=torch.zeros((1, EDGE_DIM)))  # fmt: skip


def smiles_to_graph(smiles: str | None, seed: int = 42, extended: bool = False) -> Data:
    """Featurize one molecule; padding or failures return :func:`fallback_graph`.

    ``extended`` adds degree, attached hydrogens, formal charge and smallest ring size.
    """
    if not isinstance(smiles, str) or smiles.strip() in ("", PAD_TOKEN):
        return fallback_graph(extended)
    try:
        mol = _embed(smiles, seed)
        edge_index, edge_attr = _edges(mol)
        return Data(x=_node_features(mol, extended), edge_index=edge_index, edge_attr=edge_attr)
    except ValueError as err:
        log.warning("fallback graph for %s: %s", smiles, err)
        return fallback_graph(extended)


class GraphCache:
    """Featurize each distinct SMILES once."""

    def __init__(self, seed: int = 42, extended: bool = False) -> None:
        self.seed, self.extended = seed, extended
        self._cache: dict[str, Data] = {}

    def __call__(self, smiles: str | None) -> Data:
        key = smiles if isinstance(smiles, str) else ""
        if key not in self._cache:
            self._cache[key] = smiles_to_graph(smiles, self.seed, self.extended)
        return self._cache[key]
