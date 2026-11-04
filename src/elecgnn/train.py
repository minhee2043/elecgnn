"""Splits, training, cross-validation and pretraining.

Two early-stopping modes:

* ``inner`` (default): a fraction of each training fold is held out for early
  stopping and LR scheduling; the test fold is never seen during training.
* ``outer``: early stopping on the test fold itself, as in the original scripts. Use it
  only to reproduce original-protocol numbers; it is optimistically biased.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch import nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool

from .data import MixtureDataset, MixtureTable, collate, single_graphs
from .featurize import GraphCache
from .metrics import regression_metrics
from .models import MixtureModel, make_encoder

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    epochs: int = 5000
    patience: int = 500
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    warmup_epochs: int = 10
    sched_patience: int = 30
    sched_factor: float = 0.7
    clip: float = 1.0
    early_stop: str = "inner"  # "inner" or "outer"
    inner_frac: float = 0.1
    num_workers: int = 0
    device: str = "auto"

    def torch_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


# ── Splits ────────────────────────────────────────────────────────────────────
@dataclass
class Split:
    name: str
    train: np.ndarray
    test: np.ndarray


def original_kfold(n: int, n_splits: int = 5, seed: int = 42) -> list[Split]:
    """Original ionic-conductivity CV: one shuffled KFold reused for every seed."""
    kf = KFold(n_splits, shuffle=True, random_state=seed)
    return [Split(f"k{k}", tr, te) for k, (tr, te) in enumerate(kf.split(np.arange(n)))]


def original_lce_split(n: int, test_frac: float = 0.15, seed: int = 42, n_splits: int = 5):
    """Original LCE split: torch ``random_split`` holdout + KFold on the remainder.

    Returns (cv_splits, holdout_indices). Indices match the original split exactly because
    ``random_split`` draws ``torch.randperm(n, generator)``.
    """
    n_test = int(test_frac * n)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).numpy()
    trainval, holdout = perm[: n - n_test], perm[n - n_test :]
    kf = KFold(n_splits, shuffle=True, random_state=seed)
    splits = [Split(f"k{k}", trainval[tr], trainval[te])
              for k, (tr, te) in enumerate(kf.split(trainval))]  # fmt: skip
    return splits, holdout


def repeated_kfold(n: int, n_splits: int = 5, n_repeats: int = 5) -> list[Split]:
    """Repeated k-fold with a fresh shuffle per repeat (for more powerful paired tests)."""
    out = []
    for r in range(n_repeats):
        kf = KFold(n_splits, shuffle=True, random_state=r)
        out += [Split(f"r{r}k{k}", tr, te) for k, (tr, te) in enumerate(kf.split(np.arange(n)))]
    return out


def splits_from_column(folds: pd.Series, test_values: list | None = None) -> list[Split]:
    """One split per fold value: test = rows with that value, train = the rest."""
    values = test_values if test_values is not None else sorted(folds.unique())
    arr = folds.to_numpy()
    return [Split(f"{folds.name}={v}", np.where(arr != v)[0], np.where(arr == v)[0]) for v in values]


# ── Target transforms ─────────────────────────────────────────────────────────
TRANSFORMS: dict[str, tuple[Callable, Callable]] = {
    "identity": (lambda y: y, lambda y: y),
    "log1p": (np.log1p, np.expm1),
    "log10": (np.log10, lambda y: np.power(10.0, y)),
    # original GAT ionic-conductivity script: log1p of sigma in S/m (inputs are mS/cm)
    "log1p_S_per_m": (lambda y: np.log1p(y / 10), lambda y: 10 * np.expm1(y)),
}


# ── Training loop ─────────────────────────────────────────────────────────────
def _loader(ds, idx, cfg: TrainConfig, shuffle: bool) -> DataLoader:
    return DataLoader(Subset(ds, list(idx)), batch_size=cfg.batch_size, shuffle=shuffle,
                      collate_fn=collate, num_workers=cfg.num_workers)  # fmt: skip


def _to(batch, device):
    slots, conc, scal, extra, y = batch
    return [g.to(device) for g in slots], conc.to(device), scal.to(device), extra.to(device), y.to(device)


def masked_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = torch.isfinite(target)
    return ((pred - torch.nan_to_num(target)) ** 2 * mask).sum() / mask.sum().clamp(min=1)


@torch.no_grad()
def predict(model: nn.Module, ds, idx, cfg: TrainConfig) -> np.ndarray:
    device = cfg.torch_device()
    model.eval()
    out = [model(*_to(b, device)[:4]).cpu().numpy() for b in _loader(ds, idx, cfg, False)]
    return np.concatenate(out) if out else np.zeros((0, ds.y.shape[1]))


def fit(model: nn.Module, ds, train_idx, stop_idx, cfg: TrainConfig) -> tuple[nn.Module, list[dict]]:
    """Train with warmup → ReduceLROnPlateau, early stopping on ``stop_idx``."""
    device = cfg.torch_device()
    model.to(device)
    opt = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    warmup = LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=cfg.warmup_epochs)
    sched = ReduceLROnPlateau(opt, mode="min", factor=cfg.sched_factor, patience=cfg.sched_patience)
    train_dl = _loader(ds, train_idx, cfg, True)
    stop_dl = _loader(ds, stop_idx, cfg, False)
    best, best_state, wait, history = float("inf"), None, 0, []
    for epoch in range(cfg.epochs):
        model.train()
        tr = []
        for batch in train_dl:
            slots, conc, scal, extra, y = _to(batch, device)
            loss = masked_mse(model(slots, conc, scal, extra), y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            opt.step()
            tr.append(loss.item())
        model.eval()
        with torch.no_grad():
            vl = []
            for b in stop_dl:
                *inputs, yb = _to(b, device)
                vl.append(masked_mse(model(*inputs), yb).item())
        val = float(np.mean(vl))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(tr)), "stop_loss": val})
        warmup.step() if epoch < cfg.warmup_epochs else sched.step(val)
        if val < best:
            best, best_state, wait = val, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    model.load_state_dict(best_state)
    return model, history


def _inner_split(train_idx: np.ndarray, frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(train_idx)
    n_stop = max(1, int(round(frac * len(perm))))
    return perm[n_stop:], perm[:n_stop]


def _subsample(sp: Split, seed: int, k: int, train_frac: float, add_back: float) -> Split:
    if train_frac >= 1.0 and add_back <= 0.0:
        return sp
    rng = np.random.default_rng([seed, k])
    train, test = sp.train, sp.test
    if add_back > 0:
        moved = rng.choice(test, int(round(add_back * len(test))), replace=False)
        train, test = np.concatenate([train, moved]), np.setdiff1d(test, moved)
    if train_frac < 1.0:
        train = rng.choice(train, max(2, int(round(train_frac * len(train)))), replace=False)
    return Split(sp.name, np.sort(train), np.sort(test))


def run_cv(table: MixtureTable, splits: list[Split], make_model: Callable[[], MixtureModel],
           cfg: TrainConfig, seeds: list[int], transform: str = "identity",
           encoder_state: dict | None = None, holdout: np.ndarray | None = None,
           cache: GraphCache | None = None, train_frac: float = 1.0,
           add_back: float = 0.0, save_dir: Path | None = None) -> dict[str, pd.DataFrame]:  # fmt: skip
    """Cross-validate; returns predictions, per-fold metrics and training histories.

    Learning curves:
      ``add_back``   move this fraction of each test fold into training (e.g. some ether
                     rows become available); the rest stays the test set.
      ``train_frac`` keep only this fraction of the (resulting) training rows.
    Both draws depend only on (seed, split index), so runs that differ in anything else
    (e.g. pretraining) see identical rows and stay pairable.

    ``save_dir``: if given, each trained model is saved as ``seed{seed}__{split}.pt`` with
    its train/test row indices, for later interpretation.
    """
    fwd, inv = TRANSFORMS[transform]
    log_metrics = transform != "identity"
    ds = MixtureDataset(table, fwd(table.y.astype(np.float64)), cache)
    preds, metrics, hist, models = [], [], [], {}
    for seed in seeds:
        for k, sp in enumerate(splits):
            sp = _subsample(sp, seed, k, train_frac, add_back)
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = make_model()
            if encoder_state is not None:
                model.load_encoder_weights(encoder_state)
            if cfg.early_stop == "outer":
                tr, stop = sp.train, sp.test
            else:
                tr, stop = _inner_split(sp.train, cfg.inner_frac, seed)
            model, h = fit(model, ds, tr, stop, cfg)
            p = inv(predict(model, ds, sp.test, cfg)[:, 0].astype(np.float64))
            t = table.y[sp.test, 0]
            m = regression_metrics(t, p, log_metrics)
            metrics.append({"seed": seed, "split": sp.name, "n_train": len(tr), "n_test": len(sp.test), **m})
            preds.append(pd.DataFrame({"row_id": table.row_id[sp.test], "seed": seed,
                                       "split": sp.name, "y_true": t, "y_pred": p}))  # fmt: skip
            hist.append(pd.DataFrame(h).assign(seed=seed, split=sp.name))
            models[(seed, sp.name)] = (m["R2"], copy.deepcopy(model).cpu())
            if save_dir is not None:
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": model.state_dict(), "train": tr, "test": sp.test},
                           save_dir / f"seed{seed}__{sp.name}.pt")  # fmt: skip
            log.info("seed %d %s: MAE %.4f R2 %.4f", seed, sp.name, m["MAE"], m["R2"])
    out = {"predictions": pd.concat(preds, ignore_index=True), "fold_metrics": pd.DataFrame(metrics),
           "history": pd.concat(hist, ignore_index=True)}  # fmt: skip
    if holdout is not None:  # original protocol: evaluate the best fold model (by R2) once
        key = max(models, key=lambda k: models[k][0])
        best = models[key][1].to(cfg.torch_device())
        p = inv(predict(best, ds, holdout, cfg)[:, 0].astype(np.float64))
        t = table.y[holdout, 0]
        out["holdout"] = pd.DataFrame(
            [{"model": f"seed{key[0]}_{key[1]}", **regression_metrics(t, p, log_metrics)}]
        )
        out["predictions"] = pd.concat([out["predictions"], pd.DataFrame({
            "row_id": table.row_id[holdout], "seed": key[0], "split": "holdout", "y_true": t, "y_pred": p})],
            ignore_index=True)  # fmt: skip
    return out


# ── Pretraining ───────────────────────────────────────────────────────────────
class _SingleGraphModel(nn.Module):
    def __init__(self, kind: str, hidden: int, n_out: int = 1) -> None:
        super().__init__()
        self.encoder = make_encoder(kind, hidden)
        self.fc = nn.Linear(hidden, n_out)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, g):
        return self.fc(global_mean_pool(self.encoder(g.x, g.edge_index, g.edge_attr), g.batch))


def _collate_single(batch):
    return Batch.from_data_list([g for g, _ in batch]), torch.stack([t for _, t in batch])


def pretrain_dipole(smiles: list[str], dipoles: np.ndarray, kind: str, hidden: int = 64,
                    epochs: int = 1000, patience: int = 20, batch_size: int = 128,
                    device: str = "auto", seed: int = 42) -> dict:  # fmt: skip
    """Original dipole pretraining (AdamW 1e-3, wd 1e-5; early stopping on training loss)."""
    torch.manual_seed(seed)
    dev = TrainConfig(device=device).torch_device()
    graphs = single_graphs(smiles)
    y = torch.tensor(np.asarray(dipoles, dtype=np.float32))
    y = y.view(-1, 1) if y.ndim == 1 else y
    data = list(zip(graphs, y, strict=True))
    dl = DataLoader(data, batch_size=batch_size, shuffle=True, collate_fn=_collate_single)
    model = _SingleGraphModel(kind, hidden, n_out=y.shape[1]).to(dev)
    opt = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    warmup = LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=10)
    sched = ReduceLROnPlateau(opt, mode="min", factor=0.7, patience=30)
    best, best_state, wait = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        losses = []
        for g, t in dl:
            g, t = g.to(dev), t.to(dev)
            loss = nn.functional.mse_loss(model(g), t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        cur = float(np.mean(losses))
        warmup.step() if epoch < 10 else sched.step(cur)
        if cur < best:
            best, best_state, wait = cur, copy.deepcopy(model.encoder.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    log.info("dipole pretraining finished at epoch %d, loss %.4f", epoch + 1, best)
    return best_state


def pretrain_molecular(smiles: list[str], targets: np.ndarray, kind: str, **kw) -> dict:
    """Pretrain on one or more molecular properties (e.g. HOMO, LUMO, GAP, dipole), standardized.

    Same optimizer and stopping rule as the original dipole pretraining; with one column and
    no standardization this reduces to :func:`pretrain_dipole`.
    """
    t = np.asarray(targets, dtype=np.float64)
    t = t.reshape(-1, 1) if t.ndim == 1 else t
    z = (t - t.mean(0)) / t.std(0)
    return pretrain_dipole(smiles, z, kind, **kw)


def pretrain_oedb(table: MixtureTable, kind: str, cfg: TrainConfig, hidden1: int = 64,
                  hidden2: int = 128, seed: int = 42) -> tuple[dict, dict]:  # fmt: skip
    """Multi-task pretraining on OEDB with one shared encoder.

    Targets are standardized per column; missing values are masked. Returns the
    encoder state dict and the target scaling (for reporting).
    """
    torch.manual_seed(seed)
    mu, sd = np.nanmean(table.y, axis=0), np.nanstd(table.y, axis=0)
    ds = MixtureDataset(table, (table.y - mu) / sd)
    model = MixtureModel(kind, table.n_slots, hidden1, hidden2, scalar_dim=table.scalar_dim,
                         n_out=table.y.shape[1], share_encoder=True, n_extra=table.n_extra)  # fmt: skip
    tr, stop = _inner_split(np.arange(len(table)), cfg.inner_frac, seed)
    model, _ = fit(model, ds, tr, stop, cfg)
    return model.encoders[0].cpu().state_dict(), {"mean": mu.tolist(), "std": sd.tolist(),
                                                  "targets": table.target_names}  # fmt: skip


def config_dict(cfg: TrainConfig) -> dict:
    return asdict(cfg)
