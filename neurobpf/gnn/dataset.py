import pickle
from pathlib import Path

import numpy as np
import torch

from neurobpf.gnn.features import GraphTensor, snapshot_to_graph


def make_adj(g: GraphTensor) -> torch.Tensor:
    n = g.x.shape[0]
    adj = torch.zeros((n, n), dtype=torch.float32)
    src, dst = g.edge_index
    adj[src, dst] = 1.0
    adj[dst, src] = 1.0
    return adj


def make_sparse_adj(g: GraphTensor, self_loops: bool = True) -> torch.Tensor:
    n = g.x.shape[0]
    src = g.edge_index[0]
    dst = g.edge_index[1]
    both = [torch.stack([src, dst]), torch.stack([dst, src])]
    if self_loops:
        both.append(torch.stack([torch.arange(n), torch.arange(n)]))
    idx = torch.unique(torch.cat(both, dim=1), dim=1)
    vals = torch.ones(idx.shape[1], dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, vals, (n, n)).coalesce()


def load_pickled_graphs(path: Path):
    with Path(path).open("rb") as fh:
        data = pickle.load(fh)
    graphs = [
        snapshot_to_graph(snap, i, run_id)
        for i, (snap, run_id) in enumerate(zip(data["snapshots"], data["run_ids"]))
    ]
    return graphs, data["run_types"], data["run_ids"]


def normalize_x(x):
    return x


def split_by_run(graphs, train_frac=0.6, val_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    runs = sorted({g.run_id for g in graphs})
    rng.shuffle(runs)
    if not runs:
        return [], [], []
    n_train = max(1, int(round(len(runs) * train_frac)))
    n_val = max(0, int(round(len(runs) * val_frac)))
    train_runs = set(runs[:n_train])
    val_runs = set(runs[n_train : n_train + n_val])
    test_runs = set(runs[n_train + n_val :])
    train = [g for g in graphs if g.run_id in train_runs]
    val = [g for g in graphs if g.run_id in val_runs]
    test = [g for g in graphs if g.run_id in test_runs]
    return train, val, test