import numpy as np
import pandas as pd

from elecgnn.data import MixtureTable
from elecgnn.features import add_derived, attach_features, lce_pair_table, strip_lithium


def test_strip_lithium_any_order():
    assert strip_lithium("[Li+].[O-]Cl(=O)(=O)=O") == strip_lithium("[O-]Cl(=O)(=O)=O.[Li+]")
    assert "Li" not in strip_lithium("FS([N-]S(F)(=O)=O)(=O)=O.[Li+]")


def test_lce_pair_weights_sum_to_one():
    smiles = np.array([["CC1COC(=O)O1", "O=C1OCC(F)O1", "[Li+].[O-]Cl(=O)(=O)=O", "O"]], dtype=object)
    conc = np.array([[0.875, 0.051, 0.074, 0.0]], dtype=np.float32)
    lce = MixtureTable(row_id=np.array(["f0"]), smiles=smiles, conc=conc,
                       y=np.zeros((1, 1)), target_names=["LCE"], meta=pd.DataFrame())  # fmt: skip
    table, pairs = lce_pair_table(lce)
    assert len(pairs) == 2  # one salt x two solvents
    assert np.isclose(pairs["weight"].sum(), 1.0)
    assert np.allclose(table.conc[:, 1], 0.074)


def test_derived_and_attach(tiny_table):
    f = pd.DataFrame({"log_sigma": np.linspace(0, 1, 12), "log_eta": np.zeros(12),
                      "log_D_cation": -10.0, "log_D_anion": -10.0,
                      "CN_solvent": 3.0, "CN_anion": np.linspace(0, 2, 12)})  # fmt: skip
    f = add_derived(f).assign(row_id=tiny_table.row_id)
    assert np.allclose(f["t_plus"], 0.5)
    t = attach_features(tiny_table, f, "physics")
    assert t.n_extra == 4 and np.allclose(t.extra.mean(0), 0, atol=1e-6)


def test_attach_features_file_uses_all_columns(tiny_table):
    f = pd.DataFrame({"row_id": tiny_table.row_id, "Dielec": np.arange(12.0), "Mol_Vol": np.ones(12) * 2})
    t = attach_features(tiny_table, f, "file")
    assert t.extra_names == ["Dielec", "Mol_Vol"] and t.n_extra == 2
