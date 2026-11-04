"""Rename files and values produced by earlier versions of this code to the current names.

Earlier versions called the in-house MD data "thesis". This script updates an existing
project directory in place:

    data/folds_thesis.csv                  -> data/folds_inhouse.csv
    data/unified_md.csv   source "thesis"  -> "inhouse"
    data/overlap_comparison.csv  sigma_thesis_363K -> sigma_inhouse_363K
    results/thesis_reproduction/           -> results/original_protocol/

Saved run.json files that record ``--split thesis`` or ``--atom-features thesis`` keep
working: both values remain accepted as aliases.

    python scripts/migrate_legacy_names.py            # show what would change
    python scripts/migrate_legacy_names.py --apply    # make the changes
"""

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--root", type=Path, default=Path("."), help="project directory (contains data/ and results/)"
    )
    ap.add_argument("--apply", action="store_true", help="make the changes (default: dry run)")
    a = ap.parse_args()
    data, results = a.root / "data", a.root / "results"
    todo = []

    old, new = data / "folds_thesis.csv", data / "folds_inhouse.csv"
    if old.exists() and not new.exists():
        todo.append((f"rename {old} -> {new}", lambda: old.rename(new)))

    uni = data / "unified_md.csv"
    if uni.exists():
        u = pd.read_csv(uni)
        if (u["source"] == "thesis").any():
            n = int((u["source"] == "thesis").sum())
            todo.append(
                (
                    f"{uni}: source 'thesis' -> 'inhouse' ({n} rows)",
                    lambda: u.assign(source=u["source"].replace("thesis", "inhouse")).to_csv(
                        uni, index=False
                    ),
                )
            )

    ov = data / "overlap_comparison.csv"
    if ov.exists():
        c = pd.read_csv(ov)
        if "sigma_thesis_363K" in c:
            todo.append(
                (
                    f"{ov}: column sigma_thesis_363K -> sigma_inhouse_363K",
                    lambda: c.rename(columns={"sigma_thesis_363K": "sigma_inhouse_363K"}).to_csv(
                        ov, index=False
                    ),
                )
            )

    old_r, new_r = results / "thesis_reproduction", results / "original_protocol"
    if old_r.exists() and not new_r.exists():
        todo.append((f"rename {old_r} -> {new_r}", lambda: old_r.rename(new_r)))

    if not todo:
        print("nothing to migrate")
        return
    for msg, fn in todo:
        print(("applying: " if a.apply else "would apply: ") + msg)
        if a.apply:
            fn()


if __name__ == "__main__":
    main()
