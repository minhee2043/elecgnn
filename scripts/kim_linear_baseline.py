"""Baseline: Kim et al.'s four composition features (FO, InOr, sO, aC) with linear regression,
trained on their dataset and scored on the 13 external electrolytes.

Feature values for the 13 test electrolytes are those listed in Kim et al., SI Table S2.

    python scripts/kim_linear_baseline.py --kim-xlsx data/pnas_2214357120_sd01.xlsx \\
        --test data/LCE_test13.csv --out results/lce_ext13/linear_kim_features.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

FEATURES = ["FO", "InOr", "sO", "aC"]
TABLE_S2 = pd.DataFrame({  # Kim No. -> FO, InOr, sO, aC
    "kim_no": [19, 1, 9, 4, 33, 34, 14, 18, 31, 10, 25, 15, 32],
    "FO": [0.149, 0.032, 0.222, 0.073, 0.127, 0.169, 2.336, 2.730, 0.045, 0.203, 0.135, 2.239, 0.185],
    "InOr": [1.174, 1.155, 2.001, 0.692, 0.884, 1.075, 1.888, 1.656, 1.225, 1.267, 1.260, 1.858, 1.360],
    "sO": [0.261, 0.255, 0.221, 0.124, 0.118, 0.112, 0.108, 0.079, 0.258, 0.107, 0.248, 0.112, 0.282],
    "aC": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.012, 0, 0],
})  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kim-xlsx", type=Path, required=True)
    ap.add_argument("--test", type=Path, default=Path("data/LCE_test13.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/lce_ext13/linear_kim_features.json"))
    a = ap.parse_args()
    train = pd.read_excel(a.kim_xlsx, "Dataset", header=1)
    train = train.loc[:, ~train.columns.astype(str).str.startswith("Unnamed")].dropna(
        subset=[*FEATURES, "LCE"]
    )
    test = pd.read_csv(a.test).merge(TABLE_S2, on="kim_no")
    a.out.parent.mkdir(parents=True, exist_ok=True)

    def rmse(p):
        return float(np.sqrt(np.mean((p - test["LCE"]) ** 2)))

    lin = LinearRegression().fit(train[FEATURES], train["LCE"])
    rf, rf_pred = [], []
    for seed in range(10):
        model = RandomForestRegressor(300, random_state=seed).fit(train[FEATURES], train["LCE"])
        rf_pred.append(model.predict(test[FEATURES]))
        rf.append(rmse(rf_pred[-1]))
    # per-electrolyte predictions, used for bootstrap confidence intervals
    preds = pd.DataFrame({"kim_no": test["kim_no"], "LCE": test["LCE"],
                          "linear_refit": lin.predict(test[FEATURES]), "rf_refit": np.mean(rf_pred, axis=0),
                          "kim_published": test["kim_pred_LCE"]})  # fmt: skip
    preds.to_csv(a.out.with_name("kim_baseline_predictions.csv"), index=False)
    out = {"RMSE": rmse(lin.predict(test[FEATURES])), "n_train": len(train),
           "rf_RMSE_mean": float(np.mean(rf)), "rf_RMSE_sd": float(np.std(rf, ddof=1)),
           "kim_published_RMSE": rmse(test["kim_pred_LCE"])}  # fmt: skip
    a.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
