"""External LCE test set: the 13 electrolytes measured by Kim et al. (PNAS 2023, 120,
e2214357120, SI Appendix Tables S2 and S3), in the layout of the 147-formulation training file.

Conversion follows Kim et al.'s own dataset convention (checked against the training file):
solvent molarity = volume fraction x pure-solvent molarity (density / MW from Kim's
"Molecular Database"); additives given in % are extra volume on top of the solvent blend
(e.g. "2% VC" -> 0.02); salt molarity as stated; mole fraction = molarity / total molarity.
The LCE target is the mean of the three replicate LCE values in Table S3.

Usage:
    python scripts/build_kim_test_set.py --kim-xlsx data/pnas_2214357120_sd01.xlsx --out data/LCE_test13.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Pure-solvent molarity (mol/L) from Kim et al.'s Molecular Database (overwritten from the file if given).
PURE_MOLARITY = {"EC": 14.989439, "DMC": 11.878594, "FEC": 13.709997, "DME": 9.631602,
                 "TEGDME": 4.539401, "DOL": 14.309242, "TTE": 6.602408, "VC": 15.689282}  # fmt: skip
DB_NAMES = {"EC": "EC (C3H4O3)", "DMC": "DMC (C3H6O3)", "FEC": "FEC (C3H3FO3)", "DME": "DME (C4H10O2)",
            "TEGDME": "Tetraglyme", "DOL": "DOL (C3H6O2)", "TTE": "TTE (C5H4F8O)",
            "VC": "VC (C3H2O3)"}  # fmt: skip
# SMILES exactly as written in the training file, so test and training graphs are identical.
SMILES = {
    "EC": "C1COC(=O)O1", "DMC": "COC(=O)OC", "FEC": "C1C(OC(=O)O1)F", "DME": "COCCOC",
    "TEGDME": "COCCOCCOCCOCCOC", "DOL": "O1CCOC1", "TTE": "FC(F)C(F)(F)COC(F)(F)C(F)F", "VC": "C1=COC(=O)O1",
    "LiFSI": "FS([N-]S(F)(=O)=O)(=O)=O.[Li+]", "LiPF6": "[Li+].F[P-](F)(F)(F)(F)F",
    "LiTFSI": "[Li+].C(F)(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F", "LiNO3": "[Li+].[N+](=O)([O-])[O-]",
}  # fmt: skip

# Kim No. (Table S2): solvents {name: volume ratio}, additive volumes {name: extra volume fraction},
# salts {name: mol/L}, replicate LCE (Table S3), Kim et al.'s predicted LCE (Table S3).
TEST_SET = [
    (19, {"EC": 1, "DMC": 1}, {}, {"LiPF6": 1.0}, [1.081, 1.122, 1.009], 1.170),
    (1, {"EC": 1, "DMC": 1}, {}, {"LiFSI": 1.0}, [1.062, 1.097, 1.123], 1.187),
    (9, {"EC": 1, "FEC": 1, "DMC": 1}, {}, {"LiFSI": 4.0}, [1.467, 1.464, 1.474], 1.508),
    (4, {"DME": 1, "TEGDME": 1}, {}, {"LiFSI": 1.0}, [1.331, 1.377, 1.443], 1.691),
    (33, {"DME": 1, "TEGDME": 1}, {}, {"LiFSI": 2.0}, [1.443, 1.458, 1.602], 1.758),
    (34, {"DME": 1, "TEGDME": 1}, {}, {"LiFSI": 3.0}, [1.634, 1.646, 1.708], 1.821),
    (14, {"FEC": 1.25, "DMC": 1.25, "TTE": 7.5}, {}, {"LiFSI": 1.0}, [1.865, 1.804, 1.828], 2.103),
    (18, {"DME": 0.83, "TEGDME": 0.83, "DOL": 0.83, "TTE": 7.5}, {}, {"LiFSI": 1.0},
     [2.198, 2.294, 2.330], 2.211),
    (31, {"EC": 1, "DMC": 1}, {}, {"LiFSI": 1.0, "LiNO3": 0.02}, [1.354, 1.359, 1.292], 1.190),
    (10, {"DME": 1, "TEGDME": 1}, {}, {"LiFSI": 4.0}, [1.793, 1.602, 1.736], 1.882),
    (25, {"EC": 1, "DMC": 1}, {}, {"LiTFSI": 1.0}, [1.221, 1.124, 1.153], 1.077),
    (15, {"EC": 0.83, "FEC": 0.83, "DMC": 0.83, "TTE": 7.5}, {}, {"LiFSI": 1.0},
     [2.099, 2.081, 2.133], 2.077),
    (32, {"EC": 1, "DMC": 1}, {"VC": 0.02}, {"LiFSI": 1.0}, [1.125, 1.174, 1.089], 1.114),
]  # fmt: skip


def recipe(solvents: dict, additives: dict, salts: dict) -> str:
    total = sum(solvents.values())
    s = "/".join(solvents) + " (" + ":".join(f"{v:g}" for v in solvents.values()) + " v/v)"
    extra = "".join(f" + {100 * v:g}% {k}" for k, v in additives.items())
    salt = " + ".join(f"{m:g} M {k}" for k, m in salts.items())
    return f"{salt} in {s}{extra}" if total else salt


def build(pure: dict) -> pd.DataFrame:
    rows = []
    for no, solv, add, salts, lce_reps, kim_pred in TEST_SET:
        vtot = sum(solv.values())
        mol = {k: v / vtot * pure[k] for k, v in solv.items()}
        mol.update({k: v * pure[k] for k, v in add.items()})
        mol.update(salts)
        total = sum(mol.values())
        row = {}
        for j, (name, m) in enumerate(mol.items(), start=1):
            row[f"smiles{j}"], row[f"comp{j}"], row[f"conc{j}"] = (
                SMILES[name],
                100 * m / total,
                round(m / total, 3),
            )
        for j in range(len(mol) + 1, 7):
            row[f"smiles{j}"], row[f"comp{j}"], row[f"conc{j}"] = "O", 0, 0
        row.update({"LCE": float(np.mean(lce_reps)), "compounds": len(mol),
                    "LCE_sd": float(np.std(lce_reps, ddof=1)), "kim_no": no, "kim_pred_LCE": kim_pred,
                    "recipe": recipe(solv, add, salts)})  # fmt: skip
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kim-xlsx", type=Path, help="Kim et al. Dataset S1 (to read pure-solvent molarities)")
    ap.add_argument("--out", type=Path, default=Path("data/LCE_test13.csv"))
    a = ap.parse_args()
    pure = dict(PURE_MOLARITY)
    if a.kim_xlsx is not None:
        db = pd.read_excel(a.kim_xlsx, "Molecular Database", header=2).set_index("Solvent")["mol/L"]
        pure.update({k: float(db[v]) for k, v in DB_NAMES.items() if v in db.index})
    df = build(pure)
    df.to_csv(a.out, index=False)
    rmse = np.sqrt(np.mean((df.kim_pred_LCE - df.LCE) ** 2))
    print(f"wrote {len(df)} formulations to {a.out}")
    print(f"Kim et al. linear model on these 13: RMSE {rmse:.3f}, MSE {rmse**2:.3f} (LCE units)")
    print(f"replicate noise (mean SD of LCE): {df.LCE_sd.mean():.3f}")


if __name__ == "__main__":
    main()
