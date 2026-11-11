"""Add a family hold-out column to an existing fold file without changing the other columns.

Every in-house composition whose solvent belongs to one of the given classes gets fold 0
(test); all others get fold -1 (always training). The resulting column is used like
``fold_ethers``::

    python scripts/add_family_fold.py --name fold_carbonates \\
        --classes "Linear carbonate" "Cyclic carbonate" "Fluorinated carbonate"
    elecgnn train ... --split file --fold-col fold_carbonates --test-folds 0
"""

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--folds", type=Path, default=Path("data/folds_inhouse.csv"))
    ap.add_argument("--unified", type=Path, default=Path("data/unified_md.csv"))
    ap.add_argument("--name", required=True, help="new column name, e.g. fold_carbonates")
    ap.add_argument("--classes", nargs="+", required=True, help="solvent classes to hold out together")
    ap.add_argument("--overwrite", action="store_true", help="replace the column if it already exists")
    a = ap.parse_args()

    folds = pd.read_csv(a.folds)
    if a.name in folds and not a.overwrite:
        raise SystemExit(f"{a.name} already exists in {a.folds} (use --overwrite)")
    u = pd.read_csv(a.unified)
    u = u[u["source"].isin(["inhouse", "thesis"])][["row_id", "solvent_class", "solvent_smiles"]]
    unknown = set(a.classes) - set(u.solvent_class)
    if unknown:
        raise SystemExit(f"unknown classes: {sorted(unknown)}; available: {sorted(u.solvent_class.unique())}")
    held = u[u.solvent_class.isin(a.classes)]
    col = folds["row_id"].map(dict.fromkeys(held.row_id, 0)).fillna(1).astype(int)
    if (col == 0).sum() != len(held):
        raise SystemExit("some held-out rows are missing from the fold file")
    folds[a.name] = col
    folds.to_csv(a.folds, index=False)
    print(f"{a.name}: {len(held)} test rows, {held.solvent_smiles.nunique()} solvents "
          f"({', '.join(a.classes)}) written to {a.folds}")  # fmt: skip


if __name__ == "__main__":
    main()
