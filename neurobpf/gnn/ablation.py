import copy

import torch

from neurobpf.gnn.features import GraphTensor


def shuffled_graphs(graphs, seed=0):
    out = []
    for gi, g in enumerate(graphs):
        n = g.x.shape[0]
        gen = torch.Generator().manual_seed(seed + gi)
        perm = torch.randperm(n, generator=gen)
        if torch.equal(perm, torch.arange(n)):
            perm = torch.roll(perm, 1)
        s = copy.copy(g)
        s.edge_index = perm[g.edge_index]
        out.append(s)
    return out