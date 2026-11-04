"""Regression metrics and paired model comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def regression_metrics(y_true, y_pred, log_scale: bool = False) -> dict[str, float]:
    """MAE, RMSE, R2, Spearman; with ``log_scale`` also log10-space errors.

    Log-scale metrics weight relative error evenly, so low-value systems (e.g. weakly
    conducting diluents) are not hidden by large absolute errors elsewhere. Predictions
    are floored at 1e-2 for the log metrics.
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    rho = stats.spearmanr(y_true, y_pred).statistic if len(y_true) > 2 else np.nan
    out = {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "R2": float(1 - np.sum(err**2) / ss_tot) if ss_tot > 0 else np.nan,
        "Spearman": float(rho),
    }
    if log_scale:
        lerr = np.log10(np.clip(y_pred, 1e-2, None)) - np.log10(y_true)
        out["MAE_log10"] = float(np.mean(np.abs(lerr)))
        out["median_fold_error"] = float(10 ** np.median(np.abs(lerr)))
        out["bias_log10"] = float(np.mean(lerr))
    return out


def corrected_resampled_ttest(diff: np.ndarray, n_train: float, n_test: float) -> tuple[float, float]:
    """Nadeau & Bengio (2003) corrected resampled t-test on paired fold differences.

    The variance is inflated by ``n_test / n_train`` to account for overlapping
    training sets across folds. Returns (t statistic, two-sided p-value).
    """
    diff = np.asarray(diff, float)
    k = len(diff)
    var = np.var(diff, ddof=1)
    if k < 2 or var == 0:
        return float("nan"), float("nan")
    t = diff.mean() / np.sqrt((1 / k + n_test / n_train) * var)
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    return float(t), float(p)


def compare_runs(a: pd.DataFrame, b: pd.DataFrame, metric: str = "MAE") -> dict[str, float]:
    """Paired comparison of two ``fold_metrics`` tables on matching (seed, split).

    With several splits, the Nadeau-Bengio corrected t-test accounts for overlapping
    training sets across folds. With a single split (only the seed varies, e.g. a
    leave-one-group-out test), training sets are identical across pairs, so a plain
    paired t-test over seeds is used; its conclusion then applies to that split only.
    """
    m = a.merge(b, on=["seed", "split"], suffixes=("_a", "_b"))
    if m.empty:
        raise ValueError("runs share no (seed, split) pairs; were they run on the same folds?")
    diff = (m[f"{metric}_a"] - m[f"{metric}_b"]).to_numpy()
    single_split = m["split"].nunique() == 1
    if single_split:
        res = stats.ttest_rel(m[f"{metric}_a"], m[f"{metric}_b"])
        t, p, test = float(res.statistic), float(res.pvalue), "paired t over seeds (single split)"
    else:
        t, p = corrected_resampled_ttest(diff, m["n_train_a"].mean(), m["n_test_a"].mean())
        test = "corrected resampled t (Nadeau-Bengio)"
    return {"metric": metric, "test": test, "n_pairs": len(m), "mean_a": float(m[f"{metric}_a"].mean()),
            "mean_b": float(m[f"{metric}_b"].mean()), "mean_diff": float(diff.mean()),
            "n_a_better": int((diff < 0).sum()), "t": t, "p": p}  # fmt: skip
