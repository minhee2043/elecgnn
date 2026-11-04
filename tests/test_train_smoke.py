from functools import partial

import pandas as pd

from elecgnn.data import attach_scalars
from elecgnn.models import MixtureModel
from elecgnn.train import TrainConfig, original_kfold, run_cv


def test_run_cv_smoke(tiny_table):
    cfg = TrainConfig(epochs=2, patience=2, batch_size=4, device="cpu")
    make = partial(MixtureModel, "gcn", 2)
    res = run_cv(tiny_table, original_kfold(len(tiny_table), n_splits=3), make, cfg, seeds=[0],
                 transform="log1p")  # fmt: skip
    assert len(res["fold_metrics"]) == 3
    assert set(res["predictions"]["row_id"]) == set(tiny_table.row_id)


def test_attach_scalars_marks_found_entries(tiny_table):
    lookup = pd.DataFrame({"Solv_Ener": [-0.2], "Dielec": [3.1]}, index=["COC(=O)OC"])
    t = attach_scalars(tiny_table, lookup, "both")
    assert t.scalar_dim == 2
    assert t.scalar_found[:, 0].sum() == 3 and not t.scalar_found[:, 1].any()
