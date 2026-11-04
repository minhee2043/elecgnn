"""Step 1-2 of the Digital Discovery plan: unified MD dataset and shared fold files.

Inputs (default directory: ./data)
    solvent_db_1_clean.csv                 in-house MD (363 K)
    oedb-electrolytes-v2026-05-11_1.csv    OEDB (298 K)
    oedb_qc.csv                            QC flags from compare_oedb.py (optional)

Outputs
    unified_md.csv     one row per composition, both sources, common schema
    folds_inhouse.csv   fold assignments for the in-house rows, used by EVERY model:
                         fold_random_s0..s2   5-fold KFold, three different shuffles
                         fold_solvent         5-fold GroupKFold, groups = solvent
                         fold_class           one fold per solvent class (leave-class-out)
                         fold_ethers          0 = test (all ethers), 1 = train
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Descriptors import MolWt
from sklearn.model_selection import GroupKFold, KFold

RDLogger.DisableLog("rdApp.*")
BASE = Path("data")

# Explicit solvent classes by canonical SMILES. The original SMARTS classifier is NOT used:
# it labels EC, PC and GBL as "Cyclic ether", linear esters and BTFE as "Ether", VC as
# "Other", and misses EFEC/EDFEC/FEMC as fluorinated (its F pattern needs F on the carbon
# bonded to O), and one of its SMARTS does not parse.
CLASSES = {
    "Ether": ["COCCOC", "CCOCCOCC", "CCOCC", "C1COCO1", "C1CCOC1", "CC1CCCO1"],
    "Fluorinated ether": ["COCCOCCF", "COCCOCC(F)F", "FC(F)(F)COCC(F)(F)F",
                          "FC(F)C(F)(F)COC(F)(F)C(F)F"],
    "Linear carbonate": ["COC(=O)OC", "CCOC(=O)OC", "CCOC(=O)OCC"],
    "Cyclic carbonate": ["O=C1OCCO1", "CC1COC(=O)O1", "O=c1occo1", "C[C@@H]1OC(=O)O[C@H]1C"],
    "Fluorinated carbonate": ["O=C1OCC(F)O1", "COC(=O)OCC(F)(F)F", "CCOC(=O)OCCF",
                              "CCOC(=O)OCC(F)F"],
    "Ester": ["CCOC(C)=O", "COC(C)=O", "CCC(=O)OCC", "CCCC(=O)OCC", "CCCC(=O)OC",
              "COC(=O)CC(C)=O", "COC(=O)C(C)=O", "CC(=O)OC(C)=O"],
    "Fluorinated ester": ["CCOC(=O)CF", "CC(=O)OCCF", "CCC(=O)OCCF"],
    "Lactone": ["O=C1CCCO1", "O=C1OCCC1F"],
    "Sulfone/sulfoxide": ["O=S1(=O)CCCC1", "CS(C)=O"],
    "Amide/urea": ["CN(C)C=O", "CC(=O)N(C)C", "CN1CCN(C)C1=O"],
    "Nitrile": ["CC#N", "N#CCF", "N#CCCCC#N", "N#CCCC#N"],
    "Phosphate": ["CCOP(=O)(OCC)OCC", "COP(=O)(OC)OC"],
    "Ketone": ["CC(C)=O"],
}  # fmt: skip

# The 10 in-house ethers and fluorinated ethers (none of them is in OEDB).
ETHERS = CLASSES["Ether"] + CLASSES["Fluorinated ether"]

OEDB_COLS = {
    "Ionic Conductivity (mS/cm)": "sigma_mS_cm",
    "Viscosity (mPa·s)": "eta_mPa_s",
    "Cation's Diffusivity (m²/s)": "D_cation_m2_s",
    "Anion's Diffusivity (m²/s)": "D_anion_m2_s",
    "Solvent's Diffusivity (m²/s)": "D_solvent_m2_s",
    "Coordination (Cation←Anion)": "CN_anion",
    "Coordination (Cation ← Solvent)": "CN_solvent",
    "Density (g/cm³)": "density_g_cm3",
}


def can(smi: str) -> str:
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi))


_CLASS_OF = {Chem.MolToSmiles(Chem.MolFromSmiles(s)): c for c, ss in CLASSES.items() for s in ss}


def solvent_class(smi: str) -> str:
    return _CLASS_OF.get(smi, "Unassigned")


def add_composition(df: pd.DataFrame) -> pd.DataFrame:
    """Add molar mass, salt mole fraction and solvent mole fraction from molality."""
    mw = {s: MolWt(Chem.MolFromSmiles(s)) for s in df["solvent_smiles"].unique()}
    df["solvent_molwt"] = df["solvent_smiles"].map(mw)
    n_solv_per_kg = 1000 / df["solvent_molwt"]
    df["x_salt"] = df["molality"] / (df["molality"] + n_solv_per_kg)
    df["x_solvent"] = 1 - df["x_salt"]
    return df


def load_inhouse(path: Path) -> pd.DataFrame:
    t = pd.read_csv(path)
    out = pd.DataFrame({
        "source": "inhouse", "T_K": 363.0, "row_id": t["name"],
        "cation": "Li", "salt": t["Salt"],
        "solvent_smiles": t["SMILES"].map(can), "anion_smiles": t["Ani_SMILES"].map(can),
        "n_salt_box": t["Li/FSI #"], "n_solvent_box": t["SOL #"],
        "sigma_mS_cm": t["Ion_Con"] * 10,  # stored in S/m
        "qc_flag": False,
    })  # fmt: skip
    mw = out["solvent_smiles"].map(lambda s: MolWt(Chem.MolFromSmiles(s)))
    out["molality"] = 1000 * out["n_salt_box"] / (out["n_solvent_box"] * mw)
    return out


def load_oedb(path: Path, qc_path: Path) -> pd.DataFrame:
    o = pd.read_csv(path)
    out = pd.DataFrame({
        "source": "oedb", "T_K": 298.0, "row_id": o["ID"].astype(str),
        "cation": o["Cation"], "salt": o["Cation"] + o["Anion"],
        "solvent_smiles": o["Solvent SMILES"].map(can), "anion_smiles": o["Anion SMILES"].map(can),
        "molality": o["Concentration (mol/kg)"], "exp_id": o["exp-ID"],
    })  # fmt: skip
    for src, dst in OEDB_COLS.items():
        out[dst] = o[src]
    out["qc_flag"] = False
    if qc_path.exists():
        q = pd.read_csv(qc_path)
        out["qc_flag"] = out["row_id"].isin(q.loc[q["qc_flag"], "ID"].astype(str))
    return out


def make_folds(th: pd.DataFrame) -> pd.DataFrame:
    f = th[["row_id", "solvent_smiles", "salt", "molality"]].copy()
    f["solvent_class"] = f["solvent_smiles"].map(solvent_class)
    idx = np.arange(len(f))
    for s in range(3):
        col = np.empty(len(f), dtype=int)
        for k, (_, te) in enumerate(KFold(5, shuffle=True, random_state=s).split(idx)):
            col[te] = k
        f[f"fold_random_s{s}"] = col
    col = np.empty(len(f), dtype=int)
    for k, (_, te) in enumerate(GroupKFold(5).split(idx, groups=f["solvent_smiles"])):
        col[te] = k
    f["fold_solvent"] = col
    classes = sorted(f["solvent_class"].unique())
    f["fold_class"] = f["solvent_class"].map({c: i for i, c in enumerate(classes)})
    f["fold_ethers"] = np.where(f["solvent_smiles"].isin({can(e) for e in ETHERS}), 0, 1)
    return f


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", type=Path, default=BASE)
    a = ap.parse_args()

    th = load_inhouse(a.base / "solvent_db_1_clean.csv")
    oe = load_oedb(a.base / "oedb-electrolytes-v2026-05-11_1.csv", a.base / "oedb_qc.csv")
    uni = add_composition(pd.concat([th, oe], ignore_index=True))
    uni["solvent_class"] = uni["solvent_smiles"].map(solvent_class)
    uni["solvent_in_oedb"] = uni["solvent_smiles"].isin(set(oe["solvent_smiles"]))
    uni.to_csv(a.base / "unified_md.csv", index=False)

    folds = make_folds(uni[uni["source"] == "inhouse"].reset_index(drop=True))
    folds.to_csv(a.base / "folds_inhouse.csv", index=False)

    print(uni.groupby("source").agg(rows=("row_id", "size"), solvents=("solvent_smiles", "nunique"),
                                    qc_flagged=("qc_flag", "sum")).to_string())  # fmt: skip
    print("\nin-house solvent classes:\n", folds["solvent_class"].value_counts().to_string())
    print(
        "\nleave-solvent-out fold sizes:",
        folds["fold_solvent"].value_counts().sort_index().tolist(),
    )
    print("leave-ethers-out: test rows =", int((folds["fold_ethers"] == 0).sum()),
          "| ether solvents =", folds.loc[folds["fold_ethers"] == 0, "solvent_smiles"].nunique())  # fmt: skip


if __name__ == "__main__":
    main()
