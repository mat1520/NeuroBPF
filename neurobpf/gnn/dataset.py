import pickle
from pathlib import Path

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
        snapshot_to_graph(snap, i) for i, snap in enumerate(data["snapshots"])
    ]
    return graphs, data["run_types"], data["run_ids"]


def normalize_x(x):
    return x