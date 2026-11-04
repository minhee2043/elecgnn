from elecgnn.featurize import EDGE_DIM, NODE_DIM, GraphCache, smiles_to_graph


def test_dmc_graph_matches_original_figure():
    g = smiles_to_graph("COC(=O)OC")  # dimethyl carbonate: 12 atoms, 11 bonds
    assert g.x.shape == (12, NODE_DIM) == (12, 26)
    assert g.edge_attr.shape == (2 * 11 + 12, EDGE_DIM)  # both directions + self-loops


def test_padding_and_invalid_fall_back_to_oxygen():
    for s in ["O", "", None, "not_a_smiles"]:
        g = smiles_to_graph(s)
        assert g.x.shape == (1, NODE_DIM)
        assert g.x[0, 5] == 1.0  # oxygen one-hot


def test_cache_reuses_graphs():
    cache = GraphCache()
    assert cache("CCO") is cache("CCO")
