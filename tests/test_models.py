import pytest
import torch

from elecgnn.data import MixtureDataset, collate
from elecgnn.models import MixtureModel


@pytest.mark.parametrize("kind", ["gat", "gcn"])
@pytest.mark.parametrize("share", [False, True])
@pytest.mark.parametrize("scalar_dim", [0, 2])
def test_forward_shapes(tiny_table, kind, share, scalar_dim):
    tiny_table.scalars = torch.zeros((len(tiny_table), 2, scalar_dim)).numpy()
    ds = MixtureDataset(tiny_table, tiny_table.y)
    slots, conc, scal, extra, _ = collate([ds[i] for i in range(4)])
    model = MixtureModel(kind, n_slots=2, scalar_dim=scalar_dim, share_encoder=share, n_out=3)
    assert model(slots, conc, scal, extra).shape == (4, 3)
    assert len(model.encoders) == (1 if share else 2)


def test_zero_concentration_slot_contributes_only_through_conc(tiny_table):
    ds = MixtureDataset(tiny_table, tiny_table.y)
    slots, conc, scal, extra, _ = collate([ds[0]])
    conc[:, 1] = 0.0
    model = MixtureModel("gat", n_slots=2).eval()
    desc = model.descriptor(slots, conc, scal, extra)
    assert torch.all(desc[:, 64:128] == 0)


def test_load_encoder_weights_copies_into_every_slot():
    src = MixtureModel("gat", n_slots=1).encoders[0].state_dict()
    model = MixtureModel("gat", n_slots=3)
    model.load_encoder_weights(src)
    for enc in model.encoders:
        for k, v in enc.state_dict().items():
            assert torch.equal(v, src[k])


def test_shared_encoder_batchnorm_sees_all_slots_at_once(tiny_table):
    ds = MixtureDataset(tiny_table, tiny_table.y)
    slots, conc, scal, extra, _ = collate([ds[i] for i in range(4)])
    model = MixtureModel("gat", n_slots=2, share_encoder=True).train()
    bn = model.encoders[0].input_norm
    model(slots, conc, scal, extra)
    all_x = torch.cat([g.x for g in slots])
    assert bn.num_batches_tracked.item() == 1  # one BatchNorm update, not one per slot
    assert torch.allclose(bn.running_mean, 0.1 * all_x.mean(0), atol=1e-5)


def test_extended_atom_features_and_sum_pooling(tiny_table):
    from elecgnn.featurize import NODE_DIM_EXTENDED, GraphCache, smiles_to_graph

    g = smiles_to_graph("O=C1OCCO1", extended=True)  # ethylene carbonate
    assert g.x.shape[1] == NODE_DIM_EXTENDED == 30
    ring_atoms = g.x[g.x[:, 24] == 1]
    assert torch.all(ring_atoms[:, 29] == 5)  # smallest ring size of the 5-membered ring
    ds = MixtureDataset(tiny_table, tiny_table.y, GraphCache(extended=True))
    slots, conc, scal, extra, _ = collate([ds[i] for i in range(3)])
    for pooling in ("mean", "sum"):
        model = MixtureModel("gcn", 2, pooling=pooling, node_dim=NODE_DIM_EXTENDED).eval()
        assert model(slots, conc, scal, extra).shape == (3, 1)


def test_sum_pooling_scales_with_size():
    import numpy as np
    import pandas as pd

    from elecgnn.data import MixtureTable

    smiles = np.array([["COCCOC", "O"], ["COCCOCCOCCOCCOC", "O"]], dtype=object)
    conc = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    t = MixtureTable(row_id=np.array(["a", "b"]), smiles=smiles, conc=conc,
                     y=np.zeros((2, 1)), target_names=["y"], meta=pd.DataFrame())  # fmt: skip
    ds = MixtureDataset(t, t.y)
    slots, conc, scal, extra, _ = collate([ds[0], ds[1]])
    torch.manual_seed(0)
    m = MixtureModel("gcn", 2, pooling="sum").eval()
    d = m.descriptor(slots, conc, scal, extra)[:, :64]
    assert d[1].abs().sum() > 2 * d[0].abs().sum()  # tetraglyme (larger) gives a larger summed embedding
