"""Molecular descriptors for the ionic-conductivity task.

Shared by the descriptor baselines and the hybrid (graph + descriptor) models, so both use
exactly the same features.

Weighting:
  none   standard features, including the salt/solvent ratio;
  molar  as in the GNNs: solvent-level descriptors x solvent mole fraction, anion-level
         descriptors x salt mole fraction, plus both mole fractions; composition fractions
         (O, F, C atom fractions) are already mixture-level and stay unweighted.
"""

from __future__ import annotations

import pandas as pd

SOLVENT_LEVEL = ["HOMO", "LUMO", "VIP", "VEA", "AIP", "AEA", "Red", "Ox", "Solv_Ener", "Reorg_Ener",
                 "Mol_Vol", "Dipole", "MP", "BP", "Dielec", "MolWt", "ValElec", "MinParCharge",
                 "MaxParCharge", "Electroneg"]  # fmt: skip
ANION_LEVEL = ["MolWt_Ani", "ValElec_Ani", "Electroneg_Ani"]
COMPOSITION = ["O", "sO", "aO", "sC", "F", "sF", "aF"]


def build_features(df: pd.DataFrame, weighting: str) -> pd.DataFrame:
    """Descriptor matrix (one row per composition) from the cleaned in-house table."""
    x_salt = df["Li/FSI #"] / (df["Li/FSI #"] + df["SOL #"])
    X = df[SOLVENT_LEVEL + ANION_LEVEL + COMPOSITION].astype(float).copy()
    if weighting == "none":
        X["Salt_Solv_Ratio"] = df["Li/FSI #"] / df["SOL #"]
    elif weighting == "molar":
        X[SOLVENT_LEVEL] = X[SOLVENT_LEVEL].mul(1 - x_salt, axis=0)
        X[ANION_LEVEL] = X[ANION_LEVEL].mul(x_salt, axis=0)
        X["x_solvent"], X["x_salt"] = 1 - x_salt, x_salt
    else:
        raise ValueError(f"unknown weighting: {weighting}")
    return X
