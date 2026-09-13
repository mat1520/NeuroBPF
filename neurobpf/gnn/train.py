from pathlib import Path

import numpy as np
import torch

from neurobpf.gnn.dataset import make_adj
from neurobpf.gnn.features import GraphTensor
from neurobpf.gnn.model import GAE


def train_gae(
    train_graphs,
    hidden_dim=64,
    z_dim=64,
    epochs=150,
    lr=1e-3,
    patience=20,
    seed=0,
):
    torch.manual_seed(seed)
    in_dim = train_graphs[0].x.shape[1]
    model = GAE(in_dim=in_dim, hidden_dim=hidden_dim, z_dim=z_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = None
    best_state = None
    stalled = 0
    for _ in range(epochs):
        model.train()
        total = 0.0
        for g in train_graphs:
            adj = make_adj(g)
            optimizer.zero_grad()
            loss = model.train_loss(g.x, adj)
            loss.backward()
            optimizer.step()
            total += loss.item()
        model.eval()
        with torch.no_grad():
            val_adj = make_adj(train_graphs[-1])
            val_loss = model.train_loss(train_graphs[-1].x, val_adj).item()
        if best_loss is None or val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stalled = 0
        else:
            stalled += 1
            if stalled >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def compute_threshold(model, normal_graphs, percentile=99.0):
    scores = []
    model.eval()
    for g in normal_graphs:
        s = node_scores_of(model, g).flatten().tolist()
        scores.extend(s)
    if not scores:
        return 0.0
    return float(np.percentile(scores, percentile))


def node_scores_of(model, g: GraphTensor):
    with torch.no_grad():
        adj = make_adj(g)
        return model.node_scores(g.x, adj)


def save_model(path, model, meta: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(meta)
    meta["hidden_dim"] = model.conv1.lin.out_features
    meta["z_dim"] = model.conv2.lin.out_features
    torch.save(
        {"state_dict": model.state_dict(), "meta": meta}, path
    )


def load_model(path, in_dim):
    saved = torch.load(Path(path), weights_only=False)
    meta = saved["meta"]
    model = GAE(in_dim=in_dim, hidden_dim=meta["hidden_dim"], z_dim=meta["z_dim"])
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model