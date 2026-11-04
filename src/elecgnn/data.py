"""Load electrolyte tables into a common mixture representation.

Every task becomes a :class:`MixtureTable`: ``n_slots`` components per row, each with
a SMILES string, a mole fraction and (optionally) graph-level scalars.

* ``ioncon``: two slots, [solvent, anion], from ``unified_md.csv`` (in-house rows).
* ``lce``: six slots from ``Ref_LCEdata.xlsx`` (original LCE file layout).
* ``oedb``: two slots, [solvent, anion], OEDB rows, multi-task targets for pretraining.

Graph-level scalars are the solvent solvation free energy (``Solv_Ener``) and dielectric
constant (``Dielec``) from the cleaned in-house table. Components without an entry
(all salts, and solvents absent from that table) get 0, as in the original scripts;
``scalar_found`` records which entries were real so coverage can be reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from .featurize import PAD_TOKEN, GraphCache

SCALAR_COLUMNS = {"gsol": ["Solv_Ener"], "dielec": ["Dielec"], "both": ["Solv_Ener", "Dielec"]}
OEDB_TARGETS = {  # name -> (column in unified_md.csv, log10-transform?)
    "log_sigma": ("sigma_mS_cm", True),
    "log_eta": ("eta_mPa_s", True),
    "log_D_cation": ("D_cation_m2_s", True),
    "log_D_anion": ("D_anion_m2_s", True),
    "CN_solvent": ("CN_solvent", False),
    "CN_anion": ("CN_anion", False),
}


def canonical(smiles: object) -> str:
    """RDKit canonical SMILES; empty string for missing or invalid input."""
    if not isinstance(smiles, str) or not smiles.strip():
        return ""
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else ""


@dataclass
class MixtureTable:
    row_id: np.ndarray
    smiles: np.ndarray  # (n_rows, n_slots) object
    conc: np.ndarray  # (n_rows, n_slots) mole fractions
    y: np.ndarray  # (n_rows, n_targets)
    target_names: list[str]
    meta: pd.DataFrame
    scalars: np.ndarray = field(default=None)  # (n_rows, n_slots, scalar_dim)
    scalar_found: np.ndarray = field(default=None)  # (n_rows, n_slots) bool
    extra: np.ndarray = field(default=None)  # (n_rows, n_extra) row-level features
    extra_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.scalars is None:
            self.scalars = np.zeros((*self.conc.shape, 0), dtype=np.float32)
            self.scalar_found = np.zeros(self.conc.shape, dtype=bool)
        if self.extra is None:
            self.extra = np.zeros((len(self.row_id), 0), dtype=np.float32)

    @property
    def n_slots(self) -> int:
        return self.smiles.shape[1]

    @property
    def scalar_dim(self) -> int:
        return self.scalars.shape[2]

    @property
    def n_extra(self) -> int:
        return self.extra.shape[1]

    def __len__(self) -> int:
        return len(self.row_id)


def scalar_lookup(inhouse_csv: str | Path) -> pd.DataFrame:
    """Per-solvent ``Solv_Ener`` and ``Dielec`` indexed by canonical SMILES."""
    df = pd.read_csv(inhouse_csv)
    df["can"] = df["SMILES"].map(canonical)
    return df.groupby("can")[["Solv_Ener", "Dielec"]].first()


def attach_scalars(table: MixtureTable, lookup: pd.DataFrame | None, mode: str) -> MixtureTable:
    """Fill ``table.scalars`` for ``mode`` in {none, gsol, dielec, both}."""
    if mode == "none" or lookup is None:
        table.scalars = np.zeros((*table.conc.shape, 0), dtype=np.float32)
        table.scalar_found = np.zeros(table.conc.shape, dtype=bool)
        return table
    cols = SCALAR_COLUMNS[mode]
    n, s = table.conc.shape
    out = np.zeros((n, s, len(cols)), dtype=np.float32)
    found = np.zeros((n, s), dtype=bool)
    for i in range(n):
        for j in range(s):
            key = canonical(table.smiles[i, j])
            if key in lookup.index and table.conc[i, j] > 0:
                vals = lookup.loc[key, cols].to_numpy(dtype=float)
                if np.all(np.isfinite(vals)):
                    out[i, j] = vals
                    found[i, j] = True
    table.scalars, table.scalar_found = out, found
    return table


def load_ioncon(unified_csv: str | Path, source: str = "inhouse") -> MixtureTable:
    """Ionic conductivity (mS/cm) with slots [solvent, anion]."""
    u = pd.read_csv(unified_csv)
    u = u[u["source"] == source].reset_index(drop=True)
    return MixtureTable(
        row_id=u["row_id"].astype(str).to_numpy(),
        smiles=u[["solvent_smiles", "anion_smiles"]].to_numpy(dtype=object),
        conc=u[["x_solvent", "x_salt"]].to_numpy(dtype=np.float32),
        y=u[["sigma_mS_cm"]].to_numpy(dtype=np.float32),
        target_names=["sigma_mS_cm"],
        meta=u[["row_id", "salt", "solvent_smiles", "solvent_class", "molality", "T_K"]],
    )


def load_lce(xlsx: str | Path, n_slots: int = 6, prefix: str = "lce") -> MixtureTable:
    """Experimental log Coulombic efficiency with up to six components (.xlsx or .csv)."""
    df = pd.read_csv(xlsx) if str(xlsx).endswith(".csv") else pd.read_excel(xlsx)
    smiles = np.empty((len(df), n_slots), dtype=object)
    conc = np.zeros((len(df), n_slots), dtype=np.float32)
    for j in range(n_slots):
        conc[:, j] = df[f"conc{j + 1}"].fillna(0).to_numpy(dtype=np.float32)
        col = df[f"smiles{j + 1}"].fillna(PAD_TOKEN).astype(str).to_numpy()
        smiles[:, j] = np.where(conc[:, j] > 0, col, PAD_TOKEN)
    meta = pd.DataFrame({"row_id": [f"{prefix}_{i}" for i in range(len(df))],
                         "n_components": (conc > 0).sum(axis=1)})  # fmt: skip
    return MixtureTable(row_id=meta["row_id"].to_numpy(), smiles=smiles, conc=conc,
                        y=df[["LCE"]].to_numpy(dtype=np.float32), target_names=["LCE"],
                        meta=meta)  # fmt: skip


def load_oedb(unified_csv: str | Path, cations: tuple[str, ...] = ("Li",),
              drop_qc: bool = True) -> MixtureTable:  # fmt: skip
    """OEDB multi-task pretraining table (targets standardized later)."""
    u = pd.read_csv(unified_csv)
    u = u[(u["source"] == "oedb") & u["cation"].isin(cations)]
    if drop_qc:
        u = u[~u["qc_flag"].astype(bool)]
    u = u.reset_index(drop=True)
    y = np.column_stack([
        np.log10(u[col].where(u[col] > 0)) if is_log else u[col]
        for col, is_log in OEDB_TARGETS.values()
    ]).astype(np.float32)  # fmt: skip
    return MixtureTable(
        row_id=u["row_id"].astype(str).to_numpy(),
        smiles=u[["solvent_smiles", "anion_smiles"]].to_numpy(dtype=object),
        conc=u[["x_solvent", "x_salt"]].to_numpy(dtype=np.float32),
        y=y, target_names=list(OEDB_TARGETS),
        meta=u[["row_id", "salt", "solvent_smiles", "solvent_class", "molality"]],
    )  # fmt: skip


class MixtureDataset(Dataset):
    """Torch dataset over a :class:`MixtureTable` with cached graphs."""

    def __init__(self, table: MixtureTable, y: np.ndarray, cache: GraphCache | None = None):
        cache = cache or GraphCache()
        self.graphs = [[cache(s) for s in row] for row in table.smiles]
        self.conc = torch.tensor(np.asarray(table.conc, dtype=np.float32))
        self.scalars = torch.tensor(np.asarray(table.scalars, dtype=np.float32))
        self.extra = torch.tensor(np.asarray(table.extra, dtype=np.float32))
        self.y = torch.tensor(np.asarray(y, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.graphs[idx], self.conc[idx], self.scalars[idx], self.extra[idx], self.y[idx]


def collate(batch):
    """Batch to (slot graphs, conc, scalars, extra, y)."""
    graphs, conc, scalars, extra, y = zip(*batch, strict=True)
    slots = [Batch.from_data_list([g[j] for g in graphs]) for j in range(len(graphs[0]))]
    return slots, torch.stack(conc), torch.stack(scalars), torch.stack(extra), torch.stack(y)


def single_graphs(smiles: list[str], cache: GraphCache | None = None) -> list[Data]:
    cache = cache or GraphCache()
    return [cache(s) for s in smiles]


def concat_tables(a: MixtureTable, b: MixtureTable) -> MixtureTable:
    """Stack two tables with the same slots (e.g. LCE training rows + an external test set)."""
    if a.n_slots != b.n_slots or a.target_names != b.target_names:
        raise ValueError("tables differ in slots or targets")
    return MixtureTable(row_id=np.concatenate([a.row_id, b.row_id]),
                        smiles=np.concatenate([a.smiles, b.smiles]), conc=np.concatenate([a.conc, b.conc]),
                        y=np.concatenate([a.y, b.y]), target_names=a.target_names,
                        meta=pd.concat([a.meta, b.meta], ignore_index=True))  # fmt: skip


def load_oedb_task(unified_csv: str | Path, target: str, cations: tuple[str, ...] = ("Li",)) -> MixtureTable:
    """OEDB as a prediction task with one target (a key of ``OEDB_TARGETS``; log10 for transport)."""
    t = load_oedb(unified_csv, cations=cations)
    j = t.target_names.index(target)
    keep = np.isfinite(t.y[:, j])
    return MixtureTable(row_id=t.row_id[keep], smiles=t.smiles[keep], conc=t.conc[keep],
                        y=t.y[keep, j : j + 1], target_names=[target],
                        meta=t.meta[keep].reset_index(drop=True))  # fmt: skip
