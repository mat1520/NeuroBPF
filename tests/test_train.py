import torch

from neurobpf.gnn.dataset import make_adj
from neurobpf.gnn.features import FEAT_DIM, snapshot_to_graph
from neurobpf.gnn.model import GAE
from neurobpf.gnn.train import compute_threshold, load_model, save_model, train_gae
from neurobpf.graphbuilder import EdgeData, NodeData, Snapshot


def _mini_graph(n_extra=0):
    nodes = {"hub": NodeData(node_id="hub", kind="process", label="hub", uid=1000, pid=1)}
    edges = []
    for k in range(3 + n_extra):
        nid = f"leaf{k}"
        nodes[nid] = NodeData(node_id=nid, kind="process", label=nid, uid=1000, pid=k + 2)
        edges.append(EdgeData(src="hub", dst=nid, label="EXEC", ts=1.0))
    return Snapshot(window_start=0.0, nodes=nodes, edges=edges, malicious=set())


def _models():
    graphs = [snapshot_to_graph(_mini_graph(), i) for i in range(2)]
    trained = train_gae(graphs, hidden_dim=8, z_dim=8, epochs=40, seed=0)
    torch.manual_seed(0)
    untrained = GAE(in_dim=FEAT_DIM, hidden_dim=8, z_dim=8)
    untrained.eval()
    return trained, untrained, graphs


def test_train_gae_returns_eval_model():
    model, _, _ = _models()
    assert not model.training


def test_train_gae_reduces_loss():
    model, untrained, graphs = _models()
    g = graphs[0]
    with torch.no_grad():
        before = untrained.train_loss(g.x, make_adj(g)).item()
        after = model.train_loss(g.x, make_adj(g)).item()
    assert after < before


def test_train_seed_coverage_or_single_graph():
    graph = snapshot_to_graph(_mini_graph(), 0)
    model = train_gae([graph], hidden_dim=8, z_dim=8, epochs=20, seed=0)
    assert model is not None


def test_compute_threshold_percentile():
    _, _, graphs = _models()
    model = train_gae(graphs, hidden_dim=8, z_dim=8, epochs=30, seed=0)
    import numpy as np

    zscores = []
    for g in graphs:
        with torch.no_grad():
            s = model.node_scores(g.x, make_adj(g)).flatten()
            zscores.append(((s - s.mean()) / (s.std() + 1e-12)).numpy())
    pooled = np.concatenate(zscores)
    thr = compute_threshold(model, graphs, percentile=50.0)
    assert isinstance(thr, float)
    assert abs(thr - float(np.percentile(pooled, 50.0))) < 1e-4


def test_save_load_model_roundtrip(tmp_path):
    torch.manual_seed(0)
    graph = snapshot_to_graph(_mini_graph(), 0)
    model = train_gae([graph], hidden_dim=11, z_dim=9, epochs=5, seed=0)
    path = tmp_path / "model.pt"
    save_model(path, model, {"epochs": 5})
    restored = load_model(path, in_dim=FEAT_DIM)
    assert restored.conv1.lin.weight.shape == (11, FEAT_DIM)
    assert restored.conv2.lin.weight.shape == (9, 11)
    with torch.no_grad():
        z1 = model.encode(graph.x, make_adj(graph))
        z2 = restored.encode(graph.x, make_adj(graph))
    assert torch.allclose(z1, z2)


def test_save_model_meta_has_dims(tmp_path):
    graph = snapshot_to_graph(_mini_graph(), 0)
    model = train_gae([graph], hidden_dim=10, z_dim=6, epochs=3, seed=0)
    path = tmp_path / "m.pt"
    save_model(path, model, {"source": "test"})
    saved = torch.load(path, weights_only=False)
    assert saved["meta"]["hidden_dim"] == 10
    assert saved["meta"]["z_dim"] == 6