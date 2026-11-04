"""Experimental Raman spectra (OEDB experimental set) as a check on MD solvation structure.

For every salt spectrum the pure-solvent spectrum of the same solvent is scaled and
subtracted (with a linear baseline) using points just outside the anion window; the
remaining anion band is fitted with fixed-position Gaussian components (free anion and
Li+-coordinated anion: contact ion pairs and aggregates). The coordinated-anion fraction

    f_bound = area(coordinated components) / area(all components)

is the experimental counterpart of MD's Li <- anion coordination.

Component positions are literature values for each anion and may move by +-``SHIFT_TOL``
cm-1; widths are fitted within ``FWHM_RANGE``. Spectra whose solvent has its own band in the
anion window (after scaling) are flagged ``solvent_overlap`` and reported separately,
because a Li+-shifted solvent band there can mimic a coordinated-anion component.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

# Anion: (window lo, window hi, [(component, centre cm-1, coordinated?)])
COMPONENTS = {
    "FSI": (700, 780, [("free", 718.0, False), ("CIP", 731.0, True), ("AGG", 746.0, True)]),
    "TFSI": (728, 762, [("free", 740.5, False), ("bound", 746.0, True)]),
    "PF6": (728, 760, [("free", 741.0, False), ("bound", 746.0, True)]),
    "BF4": (752, 790, [("free", 765.0, False), ("bound", 773.0, True)]),
    "ClO4": (918, 955, [("free", 932.0, False), ("CIP", 939.0, True), ("AGG", 946.0, True)]),
    "TfO": (1020, 1062, [("free", 1032.0, False), ("CIP", 1041.0, True), ("AGG", 1052.0, True)]),
}
SHIFT_TOL = 4.0
FWHM_RANGE = (3.0, 22.0)
PAD = 80  # cm-1 either side of the window used to scale the solvent spectrum and baseline

# Raman solvent codes -> SMILES. Codes present in the OEDB MD table are read from it; the
# rest (ethers and sulfones absent from OEDB MD) are listed here.
EXTRA_SOLVENT_SMILES = {
    "DME": "COCCOC", "G2": "COCCOCCOC", "G3": "COCCOCCOCCOC", "DOL": "C1COCO1", "THF": "C1CCOC1",
    "DEE": "CCOCCOCC",  # 1,2-diethoxyethane
    "DIOX": "C1COCCO1", "THP": "C1CCOCC1", "DMM": "COCOC",
    "SL": "O=S1(=O)CCCC1", "MSL": "CC1CCS(=O)(=O)C1", "EMS": "CCS(C)(=O)=O", "EiPS": "CCS(=O)(=O)C(C)C",
}  # fmt: skip
ANION_ALIASES = {"TfO": "OTf"}  # Raman code -> OEDB MD code


def load_spectra(path: str | Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """(metadata, wavenumbers, spectra) from the OEDB experimental CSV."""
    raw = pd.read_csv(path, low_memory=False)
    cols = [c for c in raw.columns if c.startswith("intensity")]
    nu = np.array([float(c.split("=")[1].split()[0]) for c in cols])
    meta = raw[["id", "cmp_id", "cation", "anion", "solvent", "Concentration (mol/kg)"]].rename(
        columns={"Concentration (mol/kg)": "molality"}
    )
    return meta, nu, raw[cols].to_numpy(float)


def subtract_solvent(nu, spec, solvent_spec, lo, hi, pad: int = PAD):
    """Spectrum minus (a * solvent + b + c * nu), fitted outside [lo, hi]; also returns a."""
    out = ((nu >= lo - pad) & (nu < lo)) | ((nu > hi) & (nu <= hi + pad))
    X = np.column_stack([solvent_spec, np.ones_like(nu), nu])
    coef, *_ = np.linalg.lstsq(X[out], spec[out], rcond=None)
    return spec - X @ coef, coef[0]


def _gauss(x, amp, cen, fwhm):
    return amp * np.exp(-4 * np.log(2) * (x - cen) ** 2 / fwhm**2)


def fit_components(x: np.ndarray, y: np.ndarray, comps: list[tuple[str, float, bool]]) -> dict:
    """Least-squares Gaussian fit with bounded centres and widths; areas per component."""
    k = len(comps)
    ymax = max(float(y.max()), 1e-9)
    p0, lo, hi = [], [], []
    for _, c, _ in comps:
        p0 += [ymax / k, c, 9.0]
        lo += [0.0, c - SHIFT_TOL, FWHM_RANGE[0]]
        hi += [np.inf, c + SHIFT_TOL, FWHM_RANGE[1]]

    def resid(p):
        return sum(_gauss(x, *p[3 * i : 3 * i + 3]) for i in range(k)) - y

    sol = least_squares(resid, p0, bounds=(lo, hi))
    areas = {name: float(sol.x[3 * i] * sol.x[3 * i + 2]) for i, (name, _, _) in enumerate(comps)}
    total = sum(areas.values())
    bound = sum(a for (name, _, b), a in zip(comps, areas.values(), strict=True) if b)
    ss_res = float(np.sum(resid(sol.x) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1e-12
    return {"f_bound": bound / total if total > 0 else np.nan, "fit_R2": 1 - ss_res / ss_tot,
            **{f"area_{n}": a for n, a in areas.items()},
            **{f"centre_{n}": float(sol.x[3 * i + 1]) for i, (n, _, _) in enumerate(comps)},
            **{f"fwhm_{n}": float(sol.x[3 * i + 2]) for i, (n, _, _) in enumerate(comps)}}  # fmt: skip


def analyse(path: str | Path, overlap_threshold: float = 0.1) -> pd.DataFrame:
    """Coordinated-anion fraction for every salt spectrum whose anion has defined components."""
    meta, nu, S = load_spectra(path)
    pure = {r.solvent: S[i] for i, r in meta.iterrows() if r.molality == 0}
    rows = []
    for i, r in meta.iterrows():
        if r.anion not in COMPONENTS or r.molality <= 0 or r.solvent not in pure:
            continue
        lo, hi, comps = COMPONENTS[r.anion]
        resid, scale = subtract_solvent(nu, S[i], pure[r.solvent], lo, hi)
        win = (nu >= lo) & (nu <= hi)
        y = np.clip(resid[win], 0, None)
        # does the (scaled) solvent itself have a band in the window?
        solv = scale * pure[r.solvent]
        solv_band = solv[win] - np.interp(nu[win], [lo, hi], [solv[win][0], solv[win][-1]])
        overlap = float(np.clip(solv_band, 0, None).max() / max(y.max(), 1e-9))
        fit = fit_components(nu[win], y, comps) if y.max() > 0 else {"f_bound": np.nan, "fit_R2": np.nan}
        rows.append({**r.to_dict(), **fit, "solvent_overlap": overlap > overlap_threshold,
                     "overlap_ratio": overlap})  # fmt: skip
    return pd.DataFrame(rows)


def solvent_smiles_map(md_csv: str | Path) -> dict[str, str]:
    md = pd.read_csv(md_csv)
    m = md.groupby("Solvent")["Solvent SMILES"].first().to_dict()
    return {**m, **EXTRA_SOLVENT_SMILES}


def anion_smiles_map(md_csv: str | Path) -> dict[str, str]:
    md = pd.read_csv(md_csv)
    m = md.groupby("Anion")["Anion SMILES"].first().to_dict()
    return {**m, **{k: m[v] for k, v in ANION_ALIASES.items() if v in m}}


def composition_frame(raman: pd.DataFrame, md_csv: str | Path) -> pd.DataFrame:
    """One row per Raman composition: canonical SMILES and mole fractions as in the unified dataset."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors

    RDLogger.DisableLog("rdApp.*")

    def can(smi):
        return Chem.MolToSmiles(Chem.MolFromSmiles(smi))

    smap, amap = solvent_smiles_map(md_csv), anion_smiles_map(md_csv)
    comp = raman[["anion", "solvent", "molality"]].drop_duplicates().reset_index(drop=True)
    comp["solvent_smiles"] = comp.solvent.map(smap)
    comp["anion_smiles"] = comp.anion.map(amap)
    comp = comp.dropna(subset=["solvent_smiles", "anion_smiles"]).reset_index(drop=True)
    comp["solvent_smiles"] = comp.solvent_smiles.map(can)
    comp["anion_smiles"] = comp.anion_smiles.map(can)
    mw = comp.solvent_smiles.map(lambda smi: Descriptors.MolWt(Chem.MolFromSmiles(smi)))
    comp["x_salt"] = comp.molality / (comp.molality + 1000.0 / mw)
    comp["x_solvent"] = 1 - comp.x_salt
    comp["row_id"] = comp.anion + "|" + comp.solvent + "|" + comp.molality.round(4).astype(str)
    return comp


def composition_table(raman: pd.DataFrame, md_csv: str | Path):
    """MixtureTable (solvent, anion slots) for the OEDB predictor, plus the composition frame."""
    from .data import MixtureTable

    comp = composition_frame(raman, md_csv)
    table = MixtureTable(row_id=comp.row_id.to_numpy(),
                         smiles=comp[["solvent_smiles", "anion_smiles"]].to_numpy(object),
                         conc=comp[["x_solvent", "x_salt"]].to_numpy(np.float32),
                         y=np.zeros((len(comp), 1), np.float32), target_names=["none"],
                         meta=comp[["row_id", "anion", "solvent", "molality"]])  # fmt: skip
    return table, comp


def summarise(df: pd.DataFrame, pred_col: str | None = None) -> pd.DataFrame:
    """Spearman agreement of the Raman coordinated fraction with MD (and predicted) coordination.

    Rows: anion x comparison x group. ``pooled`` uses every concentration; ``m=1`` / ``m=2``
    compare solvents at matched molality (the ranking of solvents by ion pairing). "Molality
    only" ranks by salt concentration alone: a predictor must beat it to show that it knows
    anything about the solvent.
    """
    from scipy.stats import spearmanr

    ok = df[~df.solvent_overlap & df.f_bound.notna()]
    out = []

    def add(anion, comparison, group, x, y):
        sel = x.notna() & y.notna()
        if sel.sum() >= 5:
            out.append({"anion": anion, "comparison": comparison, "group": group, "n": int(sel.sum()),
                        "spearman": float(spearmanr(x[sel], y[sel])[0])})  # fmt: skip

    for anion, g in ok.groupby("anion"):
        add(anion, "MD (linked OEDB compositions)", "pooled", g.f_bound, g.CN_anion_MD)
        for m in (1.0, 2.0):
            h = g[np.isclose(g.molality, m)]
            add(anion, "MD (linked OEDB compositions)", f"m={m:g}", h.f_bound, h.CN_anion_MD)
        if pred_col is not None and pred_col in g:
            for covered, label in [(True, "solvents in OEDB MD"), (False, "solvents absent from OEDB MD")]:
                h = g[g.solvent_in_oedb == covered]
                add(anion, f"OEDB predictor, {label}", "pooled", h.f_bound, h[pred_col])
                # baseline: ranking by salt concentration alone (more salt -> more ion pairing)
                add(anion, f"Molality only, {label}", "pooled", h.f_bound, h.molality)
    return pd.DataFrame(out)
