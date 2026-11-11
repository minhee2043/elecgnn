"""Chemprop (D-MPNN) baseline on the shared folds, written in the elecgnn run format.

Each formulation is passed to Chemprop as two molecules (solvent and anion, one message-passing
network each) with the solvent and salt mole fractions as extra descriptors, and the target is
log10 of the conductivity. Splits, seeds and the held-out 10 % used for early stopping follow
the elecgnn runs, so ``elecgnn compare`` can pair the results with any GNN or descriptor run.

Chemprop runs in its own environment (it pins its own PyTorch):

    conda create -y -p /path/to/envs/chemprop python=3.11
    conda activate /path/to/envs/chemprop
    pip install "chemprop>=2.1,<2.3" pandas scipy scikit-learn
    pip install -e /path/to/electrolyte-gnn --no-deps     # for elecgnn.metrics

    python scripts/chemprop_baseline.py --check            # confirms the CLI options used here
    python scripts/chemprop_baseline.py --fold-col fold_ethers --test-folds 0 \\
        --seeds 0 1 2 3 4 5 6 7 8 9 --out results/chemprop/fold_ethers
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from elecgnn.metrics import regression_metrics  # noqa: E402

SMILES = ["solvent_smiles", "anion_smiles"]
DESCRIPTORS = ["x_solvent", "x_salt"]
REQUIRED_FLAGS = ["--smiles-columns", "--target-columns", "--descriptors-columns", "--splits-file"]


def check() -> None:
    out = subprocess.run(["chemprop", "train", "--help"], capture_output=True, text=True).stdout
    missing = [f for f in REQUIRED_FLAGS if f not in out]
    version = subprocess.run(["chemprop", "--version"], capture_output=True, text=True).stdout.strip()
    print(f"chemprop {version or '(version not reported)'}")
    print("all required options found" if not missing else f"missing options: {missing}")


def _run(cmd: list[str], verbose: bool) -> None:
    """Run a Chemprop command; on failure show the command and Chemprop's own error output."""
    res = subprocess.run(cmd, capture_output=not verbose, text=True)
    if res.returncode != 0:
        msg = (res.stderr or "")[-3000:] if not verbose else "(see output above)"
        raise SystemExit(f"Chemprop failed:\n  {' '.join(cmd)}\n--- Chemprop error output ---\n{msg}")


def run_one(df: pd.DataFrame, train: np.ndarray, test: np.ndarray, seed: int, a, work: Path) -> np.ndarray:
    rng = np.random.default_rng(seed)
    stop = rng.choice(
        train, max(1, int(round(0.1 * len(train)))), replace=False
    )  # as elecgnn --early-stop inner
    fit = np.setdiff1d(train, stop)
    data = df.iloc[np.concatenate([fit, stop, test])].reset_index(drop=True)
    n_fit, n_stop = len(fit), len(stop)
    split = [{"train": list(range(n_fit)), "val": list(range(n_fit, n_fit + n_stop)),
              "test": list(range(n_fit + n_stop, len(data)))}]  # fmt: skip
    data_csv, split_json, model_dir = work / "data.csv", work / "splits.json", work / "model"
    data[SMILES + DESCRIPTORS + ["y"]].to_csv(data_csv, index=False)
    split_json.write_text(json.dumps(split))
    train_cmd = ["chemprop", "train", "--data-path", str(data_csv), "--task-type", "regression",
                 "--smiles-columns", *SMILES, "--target-columns", "y", "--descriptors-columns", *DESCRIPTORS,
                 "--splits-file", str(split_json), "--epochs", str(a.epochs), "--patience", str(a.patience),
                 "--data-seed", str(seed), "--pytorch-seed", str(seed), "--num-workers", "0",
                 "--output-dir", str(model_dir)]  # fmt: skip
    _run(train_cmd, a.verbose)
    test_csv, preds_csv = work / "test.csv", work / "preds.csv"
    data.iloc[n_fit + n_stop :][SMILES + DESCRIPTORS].to_csv(test_csv, index=False)
    pred_cmd = ["chemprop", "predict", "--test-path", str(test_csv), "--model-paths", str(model_dir),
                "--smiles-columns", *SMILES, "--descriptors-columns", *DESCRIPTORS,
                "--preds-path", str(preds_csv), "--num-workers", "0"]  # fmt: skip
    _run(pred_cmd, a.verbose)
    pred = pd.read_csv(preds_csv)
    col = [c for c in pred.columns if c not in SMILES + DESCRIPTORS][-1]
    return pred[col].to_numpy(dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="only check the installed Chemprop CLI")
    ap.add_argument("--data", type=Path, default=Path("data/unified_md.csv"))
    ap.add_argument("--folds", type=Path, default=Path("data/folds_inhouse.csv"))
    ap.add_argument("--fold-col")
    ap.add_argument("--test-folds", nargs="*", type=int)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.check:
        check()
        return
    if not (a.fold_col and a.out):
        ap.error("--fold-col and --out are required")

    u = pd.read_csv(a.data)
    u = u[u["source"].isin(["inhouse", "thesis"])]
    folds = pd.read_csv(a.folds)[["row_id", a.fold_col]]
    df = folds.merge(u[["row_id", *SMILES, *DESCRIPTORS, "sigma_mS_cm"]], on="row_id", how="left")
    if df[SMILES + ["sigma_mS_cm"]].isna().any().any():
        raise SystemExit("rows in the fold file without matching in-house data")
    df["y"] = np.log10(df["sigma_mS_cm"])
    values = a.test_folds if a.test_folds else sorted(df[a.fold_col].unique())

    preds, metrics = [], []
    a.out.mkdir(parents=True, exist_ok=True)
    for seed in a.seeds:
        for v in values:
            test = np.where(df[a.fold_col] == v)[0]
            train = np.where(df[a.fold_col] != v)[0]
            work = Path(tempfile.mkdtemp(prefix="chemprop_"))
            try:
                y_hat = 10 ** run_one(df, train, test, seed, a, work)
            finally:
                shutil.rmtree(work, ignore_errors=True)
            split = f"{a.fold_col}={v}"
            y_true = df["sigma_mS_cm"].to_numpy()[test]
            preds.append(pd.DataFrame({"row_id": df["row_id"].to_numpy()[test], "seed": seed, "split": split,
                                       "y_true": y_true, "y_pred": y_hat}))  # fmt: skip
            metrics.append({"seed": seed, "split": split, "n_train": len(train), "n_test": len(test),
                            **regression_metrics(y_true, y_hat, log_scale=True)})  # fmt: skip
            print(f"seed {seed} {split}: log10 MAE {metrics[-1]['MAE_log10']:.3f}", flush=True)

    pd.concat(preds).to_csv(a.out / "predictions.csv", index=False)
    fm = pd.DataFrame(metrics)
    fm.to_csv(a.out / "fold_metrics.csv", index=False)
    cols = [c for c in fm.columns if c not in ("seed", "split", "n_train", "n_test")]
    summary = {c: {"mean": float(fm[c].mean()), "std": float(fm[c].std(ddof=1)) if len(fm) > 1 else 0.0}
               for c in cols}  # fmt: skip
    run = {"args": {k: str(v) for k, v in vars(a).items()}, "model": "chemprop", "summary": summary}
    (a.out / "run.json").write_text(json.dumps(run, indent=2))
    print(json.dumps({k: round(v["mean"], 3) for k, v in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
