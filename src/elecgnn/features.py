"""Two-step use of OEDB: predicted electrolyte-level properties as model inputs.

1. An ensemble of mixture models is trained on OEDB (Li salts, QC-filtered) to predict
   log10 sigma, log10 eta, log10 D+, log10 D-, CN_solvent and CN_anion from
   (solvent, anion, salt mole fraction).
2. The ensemble predicts these properties for any composition. Derived quantities are
   added: the apparent transference number t+ = D+ / (D+ + D-) and the Walden product
   log10(sigma * eta).
3. Selected columns are attached to a task table as row-level ``extra`` features.

For multi-component LCE formulations, OEDB covers only single-salt, single-solvent
systems, so properties are predicted for every (salt, solvent) pair at the
formulation's total salt mole fraction and averaged with weights
(salt fraction of all salt) x (solvent fraction of all solvent).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from rdkit import Chem

from .data import MixtureDataset, MixtureTable, canonical
from .featurize import PAD_TOKEN, GraphCache
from .metrics import regression_metrics
from .models import MixtureModel
from .train import TrainConfig, _inner_split, fit, predict

log = logging.getLogger(__name__)

BASE_COLUMNS = ["log_sigma", "log_eta", "log_D_cation", "log_D_anion", "CN_solvent", "CN_anion"]
FEATURE_SETS = {
    "transport": ["log_sigma", "log_eta", "log_D_cation", "log_D_anion"],
    "structure": ["CN_solvent", "CN_anion"],
    "physics": ["log_eta", "t_plus", "log_walden", "CN_anion"],
    "sigma": ["log_sigma"],
    "eta": ["log_eta"],
    "all": [*BASE_COLUMNS, "t_plus", "log_walden"],
}


class OEDBPredictor:
    """Ensemble of mixture models trained on OEDB targets (standardized, masked MSE)."""

    def __init__(self, kind: str = "gat", n_models: int = 5, hidden1: int = 64, hidden2: int = 128):
        self.kind, self.n_models, self.hidden1, self.hidden2 = kind, n_models, hidden1, hidden2
        self.models: list[MixtureModel] = []
        self.mu = self.sd = None
        self.oof_metrics: pd.DataFrame | None = None

    def fit(self, oedb: MixtureTable, cfg: TrainConfig, cache: GraphCache | None = None) -> OEDBPredictor:
        self.mu, self.sd = np.nanmean(oedb.y, axis=0), np.nanstd(oedb.y, axis=0)
        ds = MixtureDataset(oedb, (oedb.y - self.mu) / self.sd, cache)
        rows = []
        for seed in range(self.n_models):
            model = MixtureModel(self.kind, oedb.n_slots, self.hidden1, self.hidden2,
                                 n_out=oedb.y.shape[1], share_encoder=True)  # fmt: skip
            tr, stop = _inner_split(np.arange(len(oedb)), cfg.inner_frac, seed)
            model, _ = fit(model, ds, tr, stop, cfg)
            pred = predict(model, ds, stop, cfg) * self.sd + self.mu
            for j, name in enumerate(oedb.target_names):
                ok = np.isfinite(oedb.y[stop, j])
                m = regression_metrics(oedb.y[stop, j][ok], pred[ok, j])
                rows.append({"seed": seed, "target": name, "MAE": m["MAE"], "R2": m["R2"]})
            self.models.append(model.cpu())
            log.info("OEDB predictor %d/%d trained", seed + 1, self.n_models)
        self.oof_metrics = pd.DataFrame(rows)
        return self

    def predict(self, table: MixtureTable, cfg: TrainConfig, cache: GraphCache | None = None) -> pd.DataFrame:
        """Ensemble-mean predictions (original units: log10 for transport, raw CN)."""
        ds = MixtureDataset(table, np.zeros((len(table), len(BASE_COLUMNS)), dtype=np.float32), cache)
        idx = np.arange(len(table))
        preds = np.stack([predict(m.to(cfg.torch_device()), ds, idx, cfg) for m in self.models])
        out = pd.DataFrame(preds.mean(0) * self.sd + self.mu, columns=BASE_COLUMNS)
        out["ensemble_std_log_sigma"] = (preds[:, :, 0] * self.sd[0]).std(0)
        return add_derived(out)


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    d_cat, d_an = 10 ** df["log_D_cation"], 10 ** df["log_D_anion"]
    df["t_plus"] = d_cat / (d_cat + d_an)
    df["log_walden"] = df["log_sigma"] + df["log_eta"]
    return df


def strip_lithium(smiles: str) -> str:
    """Anion SMILES from a salt written as '[Li+].<anion>' (any fragment order)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    frags = [f for f in Chem.GetMolFrags(mol, asMols=True)
             if not (f.GetNumAtoms() == 1 and f.GetAtomWithIdx(0).GetSymbol() == "Li")]  # fmt: skip
    return Chem.MolToSmiles(max(frags, key=lambda f: f.GetNumAtoms())) if frags else ""


def is_salt(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    return mol is not None and any(a.GetSymbol() == "Li" for a in mol.GetAtoms())


def lce_pair_table(lce: MixtureTable) -> tuple[MixtureTable, pd.DataFrame]:
    """Expand each LCE formulation into (salt, solvent) pairs with mixing weights.

    Returns a two-slot pair table and a frame mapping pairs to formulations
    (``row_id``, ``weight``); weights sum to 1 per formulation.
    """
    pair_rows, smiles, conc = [], [], []
    for i in range(len(lce)):
        comps = [(s, c) for s, c in zip(lce.smiles[i], lce.conc[i], strict=True) if c > 0 and s != PAD_TOKEN]
        salts = [(strip_lithium(s), c) for s, c in comps if is_salt(s)]
        solvents = [(canonical(s), c) for s, c in comps if not is_salt(s)]
        x_salt = sum(c for _, c in salts)
        s_tot, v_tot = x_salt, sum(c for _, c in solvents)
        for a, ca in salts:
            for v, cv in solvents:
                smiles.append([v, a])
                conc.append([1 - x_salt, x_salt])
                pair_rows.append({"row_id": lce.row_id[i], "weight": (ca / s_tot) * (cv / v_tot)})
    pairs = pd.DataFrame(pair_rows)
    table = MixtureTable(row_id=np.array([f"pair{k}" for k in range(len(pairs))]),
                         smiles=np.array(smiles, dtype=object), conc=np.array(conc, dtype=np.float32),
                         y=np.zeros((len(pairs), 1), dtype=np.float32), target_names=["none"],
                         meta=pairs)  # fmt: skip
    return table, pairs


def lce_features(predictor: OEDBPredictor, lce: MixtureTable, cfg: TrainConfig,
                 cache: GraphCache | None = None) -> pd.DataFrame:  # fmt: skip
    """Weighted pair-average of predicted properties, one row per formulation."""
    table, pairs = lce_pair_table(lce)
    pred = predictor.predict(table, cfg, cache)
    cols = [c for c in pred.columns if c != "ensemble_std_log_sigma"]
    weighted = pred[cols].mul(pairs["weight"].to_numpy(), axis=0).assign(row_id=pairs["row_id"].to_numpy())
    out = weighted.groupby("row_id", sort=False)[cols].sum()
    return out.reindex(lce.row_id).reset_index()


def attach_features(table: MixtureTable, features: pd.DataFrame, feature_set: str) -> MixtureTable:
    """Attach a feature set (standardized over the table rows) as ``table.extra``.

    ``feature_set`` names one of ``FEATURE_SETS`` (OEDB-derived columns) or is ``"file"``:
    every column of the features file except ``row_id`` (e.g. a descriptor file).
    """
    cols = (
        [c for c in features.columns if c != "row_id"] if feature_set == "file" else FEATURE_SETS[feature_set]
    )
    f = features.set_index("row_id").reindex(table.row_id)[cols]
    if f.isna().any().any():
        raise ValueError(f"{int(f.isna().any(axis=1).sum())} rows have no OEDB features")
    x = f.to_numpy(dtype=np.float64)
    table.extra = ((x - x.mean(0)) / np.where(x.std(0) > 0, x.std(0), 1)).astype(np.float32)
    table.extra_names = cols
    return table
