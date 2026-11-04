import numpy as np
import torch

from elecgnn.data import MixtureDataset, collate
from elecgnn.interpret import atom_attributions, atom_groups, attention_ratios, integrated_gradients
from elecgnn.models import MixtureModel


def test_atom_groups_dmc():
    groups = [g for _, g in atom_groups("COC(=O)OC")]
    assert groups.count("O-carbonyl") == 1 and groups.count("O-ether") == 2
    assert groups.count("H") == 6


def test_ig_completeness(tiny_table):
    torch.manual_seed(0)
    ds = MixtureDataset(tiny_table, tiny_table.y)
    batch = collate([ds[0]])
    model = MixtureModel("gat", 2).eval()
    ig = integrated_gradients(model, batch, steps=64)
    slots, conc, scal, extra, _ = batch
    full = model(slots, conc, scal, extra).item()
    z = slots[0].clone()
    z.x = torch.zeros_like(z.x)
    base = model([z, slots[1]], conc, scal, extra).item()
    assert abs(ig[0].sum().item() - (full - base)) < 0.05 * max(1.0, abs(full - base))


def test_uniform_attention_gives_ratio_one(tiny_table):
    ds = MixtureDataset(tiny_table, tiny_table.y)
    g = collate([ds[0]])[0][0]
    enc = MixtureModel("gat", 2).eval().encoder(0)
    for p in enc.out_layer.parameters():  # zero attention logits -> uniform softmax
        torch.nn.init.zeros_(p)
    ratio, ent = attention_ratios(enc, g)
    ok = ~np.isnan(ratio)
    assert np.allclose(ratio[ok], 1.0, atol=1e-5) and np.allclose(ent[~np.isnan(ent)], 1.0, atol=1e-5)


def test_atom_attributions_table(tiny_table):
    ds = MixtureDataset(tiny_table, tiny_table.y)
    df = atom_attributions(MixtureModel("gat", 2).eval(), ds, tiny_table, [0, 1], ["solvent", "anion"],
                           seed=0, split="s", steps=4)  # fmt: skip
    assert set(df.slot) == {"solvent", "anion"} and df.attn_ratio.notna().any()
