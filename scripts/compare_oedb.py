"""Cross-dataset consistency (RQ1): in-house MD (363 K) vs OEDB (298 K), and OEDB QC.

Run once with ``--qc-only`` before ``build_unified_and_folds.py``, then again after it.

Outputs (in --base)
    overlap_comparison.csv  in-house vs OEDB sigma on overlapping Li systems
                            (OEDB interpolated in log sigma at the in-house molality)
    oedb_qc.csv             every OEDB row with sigma / sigma_Nernst-Einstein and a QC flag
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt
from scipy.stats import spearmanr

NA, E, KB, R = 6.022e23, 1.602e-19, 1.381e-23, 8.314
CONC, SIG = "Concentration (mol/kg)", "Ionic Conductivity (mS/cm)"
D_CAT, D_AN = "Cation's Diffusivity (m²/s)", "Anion's Diffusivity (m²/s)"


def interp_log(g: pd.DataFrame, col: str, m: float) -> float:
    g = g.dropna(subset=[col]).sort_values("molality")
    return float(np.exp(np.interp(m, g["molality"], np.log(g[col]))))


def compare(uni: pd.DataFrame) -> pd.DataFrame:
    th = uni[uni["source"] == "inhouse"]
    oe = uni[(uni["source"] == "oedb") & (uni["cation"] == "Li")]
    rows = []
    for _, r in th[th["solvent_in_oedb"]].iterrows():
        if not 0.5 <= r["molality"] <= 4.0:
            continue
        g = oe[(oe["solvent_smiles"] == r["solvent_smiles"]) & (oe["anion_smiles"] == r["anion_smiles"])]
        m = r["molality"]
        rows.append({
            "solvent_smiles": r["solvent_smiles"], "salt": r["salt"], "molality": m,
            "sigma_inhouse_363K": r["sigma_mS_cm"], "sigma_oedb_298K": interp_log(g, "sigma_mS_cm", m),
            "eta_oedb_298K": interp_log(g, "eta_mPa_s", m),
        })  # fmt: skip
    c = pd.DataFrame(rows)
    c["ratio_363_298"] = c["sigma_inhouse_363K"] / c["sigma_oedb_298K"]
    c["apparent_Ea_kJ_mol"] = R * np.log(c["ratio_363_298"]) / (1 / 298 - 1 / 363) / 1000
    return c


def nernst_einstein_qc(o: pd.DataFrame, temp: float = 298.0) -> pd.DataFrame:
    m_cat = {"Li": 6.94, "Na": 22.99, "K": 39.10}
    m_salt = o["Cation"].map(m_cat) + o["Anion SMILES"].map(lambda s: MolWt(Chem.MolFromSmiles(s)))
    m = o[CONC]
    n_pairs = o["Density (g/cm³)"] * 1e6 * m / (1000 + m * m_salt) * NA  # per m^3
    sigma_ne = E**2 / (KB * temp) * n_pairs * (o[D_CAT] + o[D_AN]) * 10  # S/m -> mS/cm
    q = o[["ID", "Cation", "Anion", "Solvent", CONC, SIG]].copy()
    q["sigma_over_sigma_NE"] = o[SIG] / sigma_ne
    q["qc_flag"] = q[SIG].isna() | q["sigma_over_sigma_NE"].isna() | (q["sigma_over_sigma_NE"] > 1.5)
    return q


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", type=Path, default=Path("data"))
    ap.add_argument("--oedb", default="oedb-electrolytes-v2026-05-11_1.csv")
    ap.add_argument("--qc-only", action="store_true", help="write oedb_qc.csv and stop")
    a = ap.parse_args()

    q = nernst_einstein_qc(pd.read_csv(a.base / a.oedb))
    q.to_csv(a.base / "oedb_qc.csv", index=False)
    print(f"OEDB QC flags: {int(q['qc_flag'].sum())} of {len(q)}")
    if a.qc_only:
        return

    c = compare(pd.read_csv(a.base / "unified_md.csv"))
    c.to_csv(a.base / "overlap_comparison.csv", index=False)
    rho = spearmanr(c["sigma_inhouse_363K"], c["sigma_oedb_298K"])[0]
    print(f"{len(c)} comparable rows; Spearman = {rho:.2f}")
    for salt, g in c.groupby("salt"):
        rho = spearmanr(g["sigma_inhouse_363K"], g["sigma_oedb_298K"])[0]
        print(f"  {salt}: n={len(g)}, Spearman = {rho:.2f}")
    print("apparent Ea (kJ/mol):", c["apparent_Ea_kJ_mol"].describe().round(1).to_dict())


if __name__ == "__main__":
    main()
