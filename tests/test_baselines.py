import numpy as np
import torch

from neurobpf.gnn.baselines import mlp_ae_scores, ocsvm_scores


def test_ocsvm_scores_shape_and_anomalous_scores_higher():
    rng = np.random.default_rng(0)
    feats = np.concatenate(
        [
            rng.normal(0, 0.1, size=(50, 5)),
            rng.normal(10, 0.1, size=(3, 5)),
        ],
        axis=0,
    ).astype(float)
    s = ocsvm_scores(feats)
    assert s.shape == (53,)
    assert s[-3:].mean() > s[:-3].mean()


def test_mlp_ae_scores_outliers_above_normal():
    rng = np.random.default_rng(1)
    feats = np.concatenate(
        [
            rng.normal(0, 0.1, size=(50, 5)),
            rng.normal(10, 0.1, size=(3, 5)),
        ],
        axis=0,
    ).astype(float)
    s = mlp_ae_scores(torch.tensor(feats, dtype=torch.float32), hidden=2, epochs=150, seed=0)
    assert s.shape == (53,)
    assert s[-3:].mean() > s[:-3].mean() * 1.2