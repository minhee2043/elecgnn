"""Uncertainty of the leave-class-out screening enrichment (Fig. 6b).

Enrichment EF10% is the fraction of the true top-10 % conductors, defined over all in-house
compositions, that appear among the top-10 % of predictions, divided by 0.10. Predictions are
pooled over the ten held-out classes, so each composition is predicted once, by a model that
never saw its class. Two uncertainties are reported:

* seeds: EF10% computed for each seed separately (mean and standard deviation);
* classes: 95 % bootstrap interval from resampling the held-out classes with replacement,
  using the seed-averaged predictions, which reflects which chemical families are in the set.

Predictions are averaged over seeds on the log10 scale, as in Fig. 6b; ranking is unchanged by
the logarithm, so the per-seed values are the same on either scale.

    python scripts/screening_uncertainty.py --results results --out results/screening_uncertainty.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = {"GAT": "class_seeds/gat_none", "GCN": "class_seeds/gcn_none",
        "LightGBM descriptors": "descriptors/fold_class/LGBM_molar",
        "GAT + descriptors": "hybrid/fold_class/gat_desc_molar"}  # fmt: skip


def enrichment(y_true: np.ndarray, y_pred: np.ndarray, top: float = 0.10) -> float:
    n_top = int(np.ceil(top * len(y_true)))
    truth = set(np.argsort(-y_true)[:n_top])
    picked = set(np.argsort(-y_pred)[:n_top])
    return len(truth & picked) / n_top / top


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--out", type=Path, default=Path("results/screening_uncertainty.csv"))
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    rows = []
    for name, run in RUNS.items():
        p = pd.read_csv(a.results / run / "predictions.csv")
        p = p[p["split"] != "holdout"]
        # average over seeds on the log10 scale, as the models are trained on log10(sigma) and as in Fig. 6b
        p["y_pred"] = np.log10(np.clip(p["y_pred"], 1e-2, None))
        per_seed = [enrichment(g.y_true.to_numpy(), g.y_pred.to_numpy())
                    for _, g in p.groupby("seed") if g.row_id.nunique() == len(g)]  # fmt: skip
        mean = p.groupby("row_id").agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean"),
                                       split=("split", "first")).reset_index()  # fmt: skip
        ef = enrichment(mean.y_true.to_numpy(), mean.y_pred.to_numpy())
        classes = mean["split"].unique()
        boot = []
        for _ in range(a.n_boot):
            pick = rng.choice(classes, len(classes), replace=True)
            s = pd.concat([mean[mean["split"] == c] for c in pick])
            boot.append(enrichment(s.y_true.to_numpy(), s.y_pred.to_numpy()))
        rows.append({"model": name, "EF10_seed_averaged": ef, "EF10_per_seed_mean": float(np.mean(per_seed)),
                     "EF10_per_seed_sd": float(np.std(per_seed, ddof=1)),
                     "n_seeds": len(per_seed), "EF10_class_boot_lo": float(np.percentile(boot, 2.5)),
                     "EF10_class_boot_hi": float(np.percentile(boot, 97.5))})  # fmt: skip
    out = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(out.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
