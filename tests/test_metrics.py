import numpy as np
import pandas as pd

from elecgnn.metrics import compare_runs, corrected_resampled_ttest, regression_metrics


def test_regression_metrics_perfect():
    m = regression_metrics([1, 2, 3, 4], [1, 2, 3, 4])
    assert m["MAE"] == 0 and m["R2"] == 1 and m["Spearman"] == 1


def test_corrected_ttest_inflates_variance():
    d = np.array([0.1, 0.2, 0.15, 0.05, 0.12])
    t, p = corrected_resampled_ttest(d, n_train=80, n_test=20)
    t_naive = d.mean() / np.sqrt(np.var(d, ddof=1) / len(d))
    assert 0 < t < t_naive and 0 < p < 1


def test_compare_runs_requires_matching_folds():
    a = pd.DataFrame({"seed": [0, 0], "split": ["k0", "k1"], "n_train": [8, 8], "n_test": [2, 2],
                      "MAE": [1.0, 1.2]})  # fmt: skip
    b = a.assign(MAE=[0.9, 1.0])
    out = compare_runs(a, b)
    assert out["n_pairs"] == 2 and out["mean_diff"] > 0


def test_log_scale_metrics():
    m = regression_metrics([1.0, 10.0], [2.0, 5.0], log_scale=True)
    assert abs(m["median_fold_error"] - 2.0) < 1e-9
    assert abs(m["bias_log10"]) < 1e-9  # one 2x over, one 2x under


def test_single_split_uses_paired_t():
    a = pd.DataFrame({"seed": [0, 1, 2], "split": ["e", "e", "e"], "n_train": 10, "n_test": 5,
                      "MAE": [1.0, 1.1, 0.9]})  # fmt: skip
    out = compare_runs(a, a.assign(MAE=[1.3, 1.5, 1.2]))
    assert out["test"].startswith("paired t") and out["n_a_better"] == 3
