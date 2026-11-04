"""Write the molecular descriptors as a features file for hybrid models (graph + descriptors).

python scripts/build_descriptor_features.py --weighting molar --out data/descriptor_features_molar.csv
elecgnn train ... --extra-features data/descriptor_features_molar.csv --feature-set file
"""

import argparse
from pathlib import Path

import pandas as pd

from elecgnn.descriptors import build_features


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=Path("data/solvent_db_1_clean.csv"))
    ap.add_argument("--weighting", choices=["none", "molar"], default="molar")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    df = pd.read_csv(a.data)
    X = build_features(df, a.weighting)
    X.insert(0, "row_id", df["name"].astype(str))
    X.to_csv(a.out, index=False)
    print(f"wrote {X.shape[1] - 1} descriptor columns for {len(X)} compositions to {a.out}")


if __name__ == "__main__":
    main()
