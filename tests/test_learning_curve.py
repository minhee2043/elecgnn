import numpy as np

from elecgnn.train import Split, _subsample


def test_add_back_moves_test_rows_and_is_reproducible():
    sp = Split("s", np.arange(10), np.arange(10, 20))
    a = _subsample(sp, seed=0, k=0, train_frac=1.0, add_back=0.5)
    b = _subsample(sp, seed=0, k=0, train_frac=1.0, add_back=0.5)
    assert len(a.test) == 5 and len(a.train) == 15
    assert not set(a.train) & set(a.test)
    assert np.array_equal(a.train, b.train)


def test_train_frac_keeps_fraction():
    sp = Split("s", np.arange(100), np.arange(100, 120))
    out = _subsample(sp, seed=1, k=0, train_frac=0.25, add_back=0.0)
    assert len(out.train) == 25 and np.array_equal(out.test, sp.test)


def test_molecular_pretraining_multi_target():
    from elecgnn.train import pretrain_molecular

    smiles = ["CCO", "COC(=O)OC", "C1COC(=O)O1", "COCCOC"]
    y = np.array([[-8.0, 0.5], [-9.0, 0.2], [-10.0, 0.1], [-7.5, 0.9]])
    state = pretrain_molecular(smiles, y, "gcn", epochs=2, device="cpu")
    assert any(k.startswith("layers") for k in state)
