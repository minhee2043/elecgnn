"""Bootstrap confidence intervals for RMSE on the 13 external CE electrolytes.

Electrolytes are resampled with replacement (paired across models), so the interval for a
difference between two models accounts for the electrolytes they share. Graph models use
the seed-averaged prediction for each electrolyte.

    python scripts/external_ce_bootstrap.py --results results --out results/lce_ext13/bootstrap.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

GRAPH_RUNS = {"GCN": "lce_ext13/gcn_none", "GAT": "lce_ext13/gat_none",
              "GAT + OEDB physics": "lce_ext13/gat_physics",
              "GAT + all OEDB features": "lce_ext13/gat_all"}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--reference", default="linear_refit", help="model the differences are taken against")
    ap.add_argument("--out", type=Path, default=Path("results/lce_ext13/bootstrap.csv"))
    a = ap.parse_args()

    kim = pd.read_csv(a.results / "lce_ext13" / "kim_baseline_predictions.csv").sort_values("kim_no")
    y = kim["LCE"].to_numpy()
    preds = {c: kim[c].to_numpy() for c in ["linear_refit", "rf_refit", "kim_published"]}
    order = [f"lce_test_{i}" for i in range(len(kim))]
    test = pd.read_csv("data/LCE_test13.csv") if Path("data/LCE_test13.csv").exists() else None
    if test is not None:  # row ids follow the order of the test file; align them to kim_no
        test = test.assign(row_id=[f"lce_test_{i}" for i in range(len(test))])
        order = test.set_index("kim_no").loc[kim["kim_no"], "row_id"].tolist()
    for name, run in GRAPH_RUNS.items():
        f = a.results / run / "predictions.csv"
        if f.exists():
            p = pd.read_csv(f)
            p = p[p["split"] != "holdout"].groupby("row_id")["y_pred"].mean()
            preds[name] = p.loc[order].to_numpy()

    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(y), size=(a.n_boot, len(y)))

    def rmse(p, i=None):
        e = (p - y) if i is None else (p[i] - y[i])
        return np.sqrt(np.mean(e**2, axis=-1))

    ref = preds[a.reference]
    rows = []
    for name, p in preds.items():
        boot = rmse(p, idx)
        diff = boot - rmse(ref, idx)
        rows.append({"model": name, "RMSE": float(rmse(p)), "RMSE_lo": float(np.percentile(boot, 2.5)),
                     "RMSE_hi": float(np.percentile(boot, 97.5)),
                     f"diff_vs_{a.reference}": float(rmse(p) - rmse(ref)),
                     "diff_lo": float(np.percentile(diff, 2.5)), "diff_hi": float(np.percentile(diff, 97.5)),
                     "p_worse_than_reference": float(np.mean(diff <= 0))})  # fmt: skip
    out = pd.DataFrame(rows).sort_values("RMSE")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
