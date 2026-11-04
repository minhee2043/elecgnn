"""Clean solvent_db_1.csv: fix wrong entries, fill resolvable NaNs, log every change.

Usage (defaults to ./data for input and output):
    python clean_solvent_db.py [--input solvent_db_1.csv] [--outdir DIR]

Fixes (each change is written to solvent_db_changelog.csv):
  1. Drop the 19 empty trailing rows.
  2. FEMC rows at ratios 2 and 5 carried the SMILES and DFT values of 2-fluoroethyl methyl
     carbonate (SI entry 177). Their atom fractions show the simulated molecule is FEMC
     (SI entry 179), so SMILES and all solvent descriptors are copied from the FEMC ratio-3 rows.
  3. Typos against the Wang et al. SI (Table S3): EDFEC FP "31,4" -> 31.4, FGBL FP 7.16 -> 71.6.
     DMSO and GBL Solv_Ion_Con were blank although the SI lists 13.90 and 14.30 (first value
     taken, as for the other "a/b" entries).
  4. Red/Ox labels were swapped: the CSV "Red" column holds the SI oxidation potential.
     Values are swapped so Ox = oxidation and Red = reduction potential (V vs Li+/Li).
  5. Refrac filled per solvent; anion/salt DFT values and binding energies filled per
     (SMILES, Ani_SMILES) pair.
  6. RDKit descriptors recomputed for every row with the original code's calls, which fixes
     1,2-diethoxyethane and FEMC (computed on the wrong molecules) and fills the 2 new rows.
     Electroneg uses corrected Pauling values (the original code swapped C and H; only C
     matters because the original averages over heavy atoms).
  7. Atom fractions recomputed with the original convention (Li and anion C not counted) but
     correct atom counts: the original used 5 F for PF6- (92 rows) and 11 atoms for DMSO (9 rows).
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

BASE = Path("data")

FEMC = "COC(=O)OCC(F)(F)F"
PAULING = {"H": 2.20, "C": 2.55, "N": 3.04, "O": 3.44, "F": 3.98, "P": 2.19, "S": 2.58}
# Anion atoms counted in the system total (original convention: no Li, no anion C).
ANION_ATOMS = {
    "[N-](S(=O)(=O)F)S(=O)(=O)F": {"O": 4, "F": 2, "N": 1, "S": 2, "P": 0},
    "F[P-](F)(F)(F)(F)F": {"O": 0, "F": 6, "N": 0, "S": 0, "P": 1},
    "C(F)(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F": {
        "O": 4,
        "F": 6,
        "N": 1,
        "S": 2,
        "P": 0,
    },
}
SOLVENT_COLS = [
    "SMILES", "HOMO", "LUMO", "VIP", "VEA", "AIP", "AEA", "Red", "Ox", "Solv_Ener", "Reorg_Ener",
    "Mol_Vol", "Dipole", "MP", "BP", "FP", "Solv_Ion_Con", "Viscosity", "Dielec", "Refrac",
]  # fmt: skip
PAIR_COLS = [
    "Ani_HOMO",
    "Ani_LUMO",
    "Salt_HOMO",
    "Salt_LUMO",
    "Ani_BE",
    "Sol_BE",
    "Diff_BE",
]
CLASS_FIX = {"C1COC(=O)O1": "Carbonate", "CC1COC(=O)O1": "Carbonate", "C1=COC(=O)O1": "Carbonate",
             "C1C(OC(=O)O1)F": "Carbonate", "C1CCOC1": "Ether",
             "C(C(F)(F)F)OCC(F)(F)F": "Fluorinated ether",
             "FC(F)C(F)(F)COC(F)(F)C(F)F": "Fluorinated ether"}  # fmt: skip

log: list[dict] = []


def setv(df: pd.DataFrame, idx, col: str, new, reason: str) -> None:
    old = df.at[idx, col]
    same = (pd.isna(old) and pd.isna(new)) or (not pd.isna(old) and not pd.isna(new) and str(old) == str(new))
    if same:
        return
    if isinstance(new, (int, float, np.floating)) and not pd.isna(old):
        try:
            if abs(float(old) - float(new)) < 1e-9:
                return
        except ValueError:
            pass
    log.append(
        {
            "name": df.at[idx, "name"],
            "column": col,
            "old": old,
            "new": new,
            "reason": reason,
        }
    )
    df.at[idx, col] = new


def atom_counts(smi: str, with_h: bool) -> Counter:
    mol = Chem.MolFromSmiles(smi)
    if with_h:
        mol = Chem.AddHs(mol)
    return Counter(a.GetSymbol() for a in mol.GetAtoms())


def rdkit_desc(smi: str) -> dict:
    mol = Chem.MolFromSmiles(smi)
    heavy = atom_counts(smi, with_h=False)
    en = sum(PAULING[e] * n for e, n in heavy.items()) / sum(heavy.values())
    return {
        "MolWt": Descriptors.ExactMolWt(mol),
        "ValElec": Descriptors.NumValenceElectrons(mol),
        "MinParCharge": Descriptors.MinPartialCharge(mol),
        "MaxParCharge": Descriptors.MaxPartialCharge(mol),
        "Electroneg": en,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", type=Path, default=BASE / "solvent_db_1.csv")
    ap.add_argument("--outdir", type=Path, default=BASE)
    args = ap.parse_args()
    OUT = args.outdir
    df = pd.read_csv(args.input)
    df = df.dropna(how="all").dropna(subset=["SMILES"]).reset_index(drop=True)
    for c in ["Solvent", "name"]:
        df[c] = df[c].str.strip()
    df["FP"] = df["FP"].astype(object)

    # 2. FEMC identity
    femc_ref = df[(df["SMILES"] == FEMC)].iloc[0]
    for i in df.index[df["Solvent"].str.contains("F-EMC") & (df["SMILES"] != FEMC)]:
        for c in SOLVENT_COLS:
            setv(
                df,
                i,
                c,
                femc_ref[c],
                "FEMC row had 2-fluoroethyl methyl carbonate data",
            )

    # 3. SI typos and blanks
    for i in df.index[df["SMILES"] == "CCOC(=O)OCC(F)F"]:
        setv(df, i, "FP", 31.4, "SI Table S3: 31.40")
    for i in df.index[df["SMILES"] == "O=C1OCCC1F"]:
        setv(df, i, "FP", 71.6, "SI Table S3: 71.60")
    for smi, val in {"CS(=O)C": 13.9, "O=C1OCCC1": 14.3}.items():
        for i in df.index[df["SMILES"] == smi]:
            setv(df, i, "Solv_Ion_Con", val, "SI Table S3 value missing in CSV")
    df["FP"] = pd.to_numeric(df["FP"])

    # 4. Red/Ox swap (values only; logged once per column, not per row)
    df[["Red", "Ox"]] = df[["Ox", "Red"]].to_numpy()
    log.append({"name": "ALL", "column": "Red/Ox", "old": "Red=oxidation", "new": "Ox=oxidation",
                "reason": "labels were swapped vs SI Table S1"})  # fmt: skip

    # 5. Fills
    for _, g in df.groupby("SMILES"):
        vals = g["Refrac"].dropna().unique()
        if len(vals) == 1:
            for i in g.index[g["Refrac"].isna()]:
                setv(df, i, "Refrac", vals[0], "same solvent")
    for _, g in df.groupby(["SMILES", "Ani_SMILES"]):
        for c in PAIR_COLS:
            vals = g[c].dropna().unique()
            if len(vals) == 1:
                for i in g.index[g[c].isna()]:
                    setv(df, i, c, vals[0], "same SMILES + Ani_SMILES")

    # 6. RDKit descriptors
    for i, r in df.iterrows():
        for c, v in rdkit_desc(r["SMILES"]).items():
            setv(df, i, c, v, "RDKit recompute")
        a = rdkit_desc(r["Ani_SMILES"])
        for c, v in {"MolWt_Ani": a["MolWt"], "ValElec_Ani": a["ValElec"],
                     "MaxParCharge_Ani": a["MaxParCharge"], "MinParCharge_Ani": a["MinParCharge"],
                     "Electroneg_Ani": a["Electroneg"]}.items():  # fmt: skip
            setv(df, i, c, v, "RDKit recompute")

    # 7. Atom fractions
    for i, r in df.iterrows():
        s = atom_counts(r["SMILES"], with_h=True)
        a = ANION_ATOMS[r["Ani_SMILES"]]
        nl, ns = r["Li/FSI #"], r["SOL #"]
        tot = nl * sum(a.values()) + ns * sum(s.get(k, 0) for k in "CHOFNSP")
        fr = {
            "O": (ns * s.get("O", 0) + nl * a["O"]) / tot,
            "sO": ns * s.get("O", 0) / tot,
            "aO": nl * a["O"] / tot,
            "sC": ns * s.get("C", 0) / tot,
            "F": (ns * s.get("F", 0) + nl * a["F"]) / tot,
            "sF": ns * s.get("F", 0) / tot,
            "aF": nl * a["F"] / tot,
        }
        for c, v in fr.items():
            setv(df, i, c, v, "atom fraction recompute")

    df["Group_corrected"] = df["SMILES"].map(CLASS_FIX).fillna(df["Group"])
    df.to_csv(OUT / "solvent_db_1_clean.csv", index=False)
    pd.DataFrame(log).to_csv(OUT / "solvent_db_changelog.csv", index=False)

    lg = pd.DataFrame(log)
    print(f"rows {len(df)}, changes {len(lg)}")
    print(lg.groupby("reason").size().to_string())
    na = df.isna().sum()
    print("remaining NaN:\n", na[na > 0].to_string())


if __name__ == "__main__":
    main()
