"""Descriptor-based baselines for ionic conductivity, directly comparable with the GNN runs.

Same data (cleaned in-house table), same fold file, same seeds, same target transform
(log10 sigma) and the same metrics and output files as `elecgnn train`, so
`elecgnn compare results/descriptors/<split>/LGBM_none results/<gnn run>` pairs folds.

Changes from the original script:
* reads the cleaned table (solvent_db_1_clean.csv) instead of solvent_db.csv;
* uses the shared fold file instead of its own random KFold;
* the correlation filter (|r| > 0.9) is fitted inside each training fold (it was fitted on
  all rows, a mild leak) and refitted inside the hyperparameter search;
* trains on log10 sigma (the GNN default) instead of log1p.

Feature weighting (--weighting):
  none   standard features, including the salt/solvent ratio;
  molar  as in the GNNs: solvent-level descriptors x solvent mole fraction, anion-level
         descriptors x salt mole fraction, plus both mole fractions; composition fractions
         (O, F, C atom fractions) are already mixture-level and stay unweighted.

    python scripts/descriptor_baselines.py --fold-col fold_solvent --weighting none molar --shap
"""

import argparse
import itertools
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from elecgnn.descriptors import build_features
from elecgnn.metrics import regression_metrics
from elecgnn.train import splits_from_column

log = logging.getLogger("descriptors")

MODEL_NAMES = ("RFR", "ADR", "XGBR", "LGBM", "SVR", "RDG", "LSO", "GPR", "MLP")
METRICS = ["MAE", "RMSE", "R2", "Spearman", "MAE_log10", "median_fold_error", "bias_log10"]


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drop the later feature of every pair with |Pearson r| above ``threshold`` (fit on training rows)."""

    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        corr = X.corr().abs().fillna(0).to_numpy()
        upper = np.triu(corr, k=1)
        self.keep_ = [c for j, c in enumerate(X.columns) if not (upper[:, j] > self.threshold).any()]
        return self

    def transform(self, X):
        return pd.DataFrame(X)[self.keep_]


def _models(seed: int) -> dict:
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    return {
        "RFR": (RandomForestRegressor(random_state=seed), False,
                {"n_estimators": [100, 200], "max_depth": [None, 10, 20], "max_features": ["sqrt", 0.5]}),
        "ADR": (AdaBoostRegressor(random_state=seed), False,
                {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1, 0.5]}),
        "XGBR": (XGBRegressor(random_state=seed, verbosity=0), False,
                 {"n_estimators": [100, 200], "max_depth": [3, 5, 7], "learning_rate": [0.05, 0.1],
                  "subsample": [0.8, 1.0]}),
        "LGBM": (LGBMRegressor(random_state=seed, verbose=-1, n_jobs=1), False,
                 {"n_estimators": [100, 200, 300], "max_depth": [-1, 5, 10, 15],
                  "learning_rate": [0.01, 0.05, 0.1], "num_leaves": [15, 31, 63]}),
        "SVR": (SVR(kernel="rbf"), True,
                {"C": [1, 10, 100], "epsilon": [0.01, 0.1], "gamma": ["scale", "auto"]}),
        "RDG": (Ridge(), True, {"alpha": [0.1, 1.0, 10.0]}),
        "LSO": (Lasso(max_iter=10000), True, {"alpha": [0.01, 0.1, 1.0]}),
        "GPR": (GaussianProcessRegressor(kernel=ConstantKernel() * RBF(), random_state=seed), True,
                {"alpha": [1e-3, 1e-2, 1e-1], "n_restarts_optimizer": [1, 3]}),
        "MLP": (MLPRegressor(max_iter=500, random_state=seed), True,
                {"hidden_layer_sizes": [(128, 64), (128, 64, 32)], "learning_rate_init": [0.001, 0.005],
                 "alpha": [0.0001, 0.001]}),
    }  # fmt: skip


def search(estimator, scale: bool, grid: dict, seed: int, n_iter: int = 15):
    pipe = Pipeline([("filter", CorrelationFilter()), ("scale", StandardScaler() if scale else "passthrough"),
                     ("model", estimator)])  # fmt: skip
    grid = {f"model__{k}": v for k, v in grid.items()}
    cv = KFold(3, shuffle=True, random_state=seed)
    n_comb = len(list(itertools.product(*grid.values())))
    if n_comb <= n_iter:
        return GridSearchCV(pipe, grid, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1)
    return RandomizedSearchCV(pipe, grid, n_iter=n_iter, cv=cv, scoring="neg_mean_squared_error",
                              random_state=seed, n_jobs=-1)  # fmt: skip


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=Path("data/solvent_db_1_clean.csv"))
    ap.add_argument("--folds", type=Path, default=Path("data/folds_inhouse.csv"))
    ap.add_argument("--fold-col", required=True)
    ap.add_argument(
        "--test-folds", nargs="*", help="evaluate only these fold values (e.g. 0 for fold_ethers)"
    )
    ap.add_argument(
        "--models", nargs="+", default=["RFR", "ADR", "XGBR", "LGBM", "SVR", "RDG", "LSO", "GPR", "MLP"]
    )
    ap.add_argument("--weighting", nargs="+", default=["none", "molar"], choices=["none", "molar"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--shap", action="store_true", help="mean |SHAP| per feature for LGBM")
    ap.add_argument("--out", type=Path, help="default: results/descriptors/<fold-col>")
    a = ap.parse_args()
    out_root = a.out or Path("results/descriptors") / a.fold_col

    df = pd.read_csv(a.data)
    folds = pd.read_csv(a.folds).set_index("row_id")[a.fold_col].reindex(df["name"].astype(str))
    if folds.isna().any():
        raise SystemExit(f"{int(folds.isna().sum())} rows have no fold")
    test_vals = None if a.test_folds is None else [type(folds.iloc[0])(v) for v in a.test_folds]
    splits = splits_from_column(folds.rename(a.fold_col).reset_index(drop=True), test_vals)
    y = df["Ion_Con"].to_numpy(float) * 10  # S/m -> mS/cm
    y_log = np.log10(y)

    for weighting in a.weighting:
        X = build_features(df, weighting)
        for name in a.models:
            out = out_root / f"{name}_{weighting}"
            out.mkdir(parents=True, exist_ok=True)
            metrics, preds, shap_rows = [], [], []
            for seed in a.seeds:
                est, scale, grid = _models(seed)[name]
                for sp in splits:
                    s = search(est, scale, grid, seed).fit(X.iloc[sp.train], y_log[sp.train])
                    p = 10 ** s.best_estimator_.predict(X.iloc[sp.test])
                    m = regression_metrics(y[sp.test], p, log_scale=True)
                    metrics.append({"seed": seed, "split": sp.name, "n_train": len(sp.train),
                                    "n_test": len(sp.test), **m})  # fmt: skip
                    ids = df["name"].iloc[sp.test].to_numpy()
                    preds.append(pd.DataFrame({"row_id": ids, "seed": seed, "split": sp.name,
                                               "y_true": y[sp.test], "y_pred": p}))  # fmt: skip
                    if a.shap and name == "LGBM":
                        import shap

                        pre, model = s.best_estimator_[:-1], s.best_estimator_["model"]
                        Xt = pre.transform(X.iloc[sp.test])
                        sv = np.abs(shap.TreeExplainer(model).shap_values(Xt)).mean(0)
                        shap_rows += [{"seed": seed, "split": sp.name, "feature": f, "mean_abs_shap": v}
                                      for f, v in zip(Xt.columns, sv, strict=True)]  # fmt: skip
                    log.info("%s %s seed %d %s: MAE %.3f log10MAE %.3f", name, weighting, seed, sp.name,
                             m["MAE"], m["MAE_log10"])  # fmt: skip
            fm = pd.DataFrame(metrics)
            fm.to_csv(out / "fold_metrics.csv", index=False)
            pd.concat(preds, ignore_index=True).to_csv(out / "predictions.csv", index=False)
            summary = {k: {"mean": float(fm[k].mean()), "std": float(fm[k].std())} for k in METRICS}
            info = {"model": name, "weighting": weighting, "fold_col": a.fold_col, "transform": "log10",
                    "summary": summary}  # fmt: skip
            (out / "run.json").write_text(json.dumps(info, indent=2))
            if shap_rows:
                sh = pd.DataFrame(shap_rows)
                # features absent from a fold (removed by the filter) count as zero importance there
                n_runs = sh.groupby(["seed", "split"]).ngroups
                imp = (sh.groupby("feature")["mean_abs_shap"].sum() / n_runs).sort_values(ascending=False)
                imp.rename("mean_abs_shap").to_csv(out / "shap_importance.csv")
            log.info("== %s %s: MAE %.3f, log10 MAE %.3f", name, weighting, summary["MAE"]["mean"],
                     summary["MAE_log10"]["mean"])  # fmt: skip


if __name__ == "__main__":
    main()
