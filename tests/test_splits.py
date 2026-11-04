import numpy as np
import pandas as pd
import torch
from torch.utils.data import random_split

from elecgnn.train import original_kfold, original_lce_split, splits_from_column


def test_lce_split_matches_torch_random_split():
    n = 147
    trainval, test = random_split(range(n), [n - int(0.15 * n), int(0.15 * n)],
                                  generator=torch.Generator().manual_seed(42))  # fmt: skip
    splits, holdout = original_lce_split(n)
    assert sorted(holdout) == sorted(test.indices)
    cv_rows = np.concatenate([s.test for s in splits])
    assert sorted(cv_rows) == sorted(trainval.indices)


def test_kfold_partitions_rows():
    splits = original_kfold(20)
    assert sorted(np.concatenate([s.test for s in splits])) == list(range(20))


def test_splits_from_column_with_selected_test_fold():
    col = pd.Series([0, 1, 1, 0, 1], name="fold_ethers")
    (sp,) = splits_from_column(col, [0])
    assert list(sp.test) == [0, 3] and list(sp.train) == [1, 2, 4]
