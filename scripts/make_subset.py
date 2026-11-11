"""Write a copy of the unified data without the in-house rows of some solvent classes.

Used for nested model selection: hyperparameters are chosen by cross-validation on data that
contains no member of the held-out family, and only the chosen setting is then evaluated on
that family.

    python scripts/make_subset.py --exclude-classes "Ether" "Fluorinated ether" \\
        --out data/unified_md_noethers.csv
"""

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--unified", type=Path, default=Path("data/unified_md.csv"))
    ap.add_argument("--exclude-classes", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    u = pd.read_csv(a.unified)
    inhouse = u["source"].isin(["inhouse", "thesis"])
    unknown = set(a.exclude_classes) - set(u.loc[inhouse, "solvent_class"])
    if unknown:
        raise SystemExit(f"unknown classes: {sorted(unknown)}")
    drop = inhouse & u["solvent_class"].isin(a.exclude_classes)
    u[~drop].to_csv(a.out, index=False)
    kept = u[inhouse & ~drop]
    n_solv = kept.solvent_smiles.nunique()
    print(f"removed {int(drop.sum())} in-house rows; kept {len(kept)} rows of {n_solv} solvents")


if __name__ == "__main__":
    main()
