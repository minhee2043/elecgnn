"""Applicability domain (RQ4): prediction error vs similarity to the nearest training solvent.

For every held-out solvent in each split, computes the mean |log10(pred / true)| and the
maximum Morgan-fingerprint Tanimoto similarity to the solvents in that split's training set.

Usage:
    python scripts/coverage_vs_error.py --runs fold_solvent=results/oedb_features/ioncon_fold_solvent/none \\
        fold_class=results/oedb_features/ioncon_fold_class/none --out results/coverage_vs_error.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import spearmanr

RDLogger.DisableLog("rdApp.*")
BINS = [0, 0.3, 0.45, 0.6, 1.0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--folds", type=Path, default=Path("data/folds_inhouse.csv"))
    ap.add_argument("--runs", nargs="+", required=True, help="fold_col=run_dir pairs")
    ap.add_argument("--out", type=Path, default=Path("results/coverage_vs_error.csv"))
    a = ap.parse_args()

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    folds = pd.read_csv(a.folds)
    fp = {s: gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in folds["solvent_smiles"].unique()}
    rows = []
    for spec in a.runs:
        col, run = spec.split("=", 1)
        p = pd.read_csv(Path(run) / "predictions.csv").merge(
            folds[["row_id", "solvent_smiles", "solvent_class", col]], on="row_id"
        )
        p = p[p["split"] != "holdout"]
        p["log_err"] = np.abs(np.log10(np.clip(p["y_pred"], 1e-2, None) / p["y_true"]))
        for (split, solv), g in p.groupby(["split", "solvent_smiles"]):
            k = int(split.split("=")[1])
            train = folds.loc[folds[col] != k, "solvent_smiles"].unique()
            sim = max(DataStructs.BulkTanimotoSimilarity(fp[solv], [fp[t] for t in train]))
            rows.append({"split": col, "solvent": solv, "solvent_class": g["solvent_class"].iloc[0],
                         "nn_similarity": sim, "log_err": g["log_err"].mean()})  # fmt: skip
    r = pd.DataFrame(rows)
    r.to_csv(a.out, index=False)
    for col, g in r.groupby("split"):
        rho = spearmanr(g.nn_similarity, g.log_err)[0]
        print(f"{col:13s} n={len(g):3d}  Spearman(similarity, log error) = {rho:+.2f}")
    r["sim_bin"] = pd.cut(r["nn_similarity"], BINS)
    print(r.groupby("sim_bin", observed=True)["log_err"].agg(["count", "median"]).round(2).to_string())


if __name__ == "__main__":
    main()
