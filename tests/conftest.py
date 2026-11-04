import numpy as np
import pandas as pd
import pytest

from elecgnn.data import MixtureTable


@pytest.fixture
def tiny_table() -> MixtureTable:
    """Twelve two-component mixtures with a smooth synthetic target."""
    solvents = ["COC(=O)OC", "COCCOC", "C1CCOC1", "O=C1OCCO1"] * 3
    anions = ["[N-](S(=O)(=O)F)S(=O)(=O)F", "F[P-](F)(F)(F)(F)F", "[O-]Cl(=O)(=O)=O"] * 4
    x_salt = np.linspace(0.1, 0.4, 12, dtype=np.float32)
    conc = np.column_stack([1 - x_salt, x_salt]).astype(np.float32)
    y = (5 + 10 * x_salt).reshape(-1, 1).astype(np.float32)
    return MixtureTable(row_id=np.array([f"r{i}" for i in range(12)]),
                        smiles=np.column_stack([solvents, anions]).astype(object), conc=conc,
                        y=y, target_names=["y"], meta=pd.DataFrame({"i": range(12)}))  # fmt: skip
