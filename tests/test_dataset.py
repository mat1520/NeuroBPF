import pickle

import torch

from neurobpf.gnn.dataset import load_pickled_graphs, make_adj
from neurobpf.gnn.features import snapshot_to_graph
from neurobpf.graphbuilder import EdgeData, NodeData, Snapshot


def _snapshot():
    nodes = {
        "p:1": NodeData(node_id="p:1", kind="process", label="bash", uid=1000, pid=1),
        "p:2": NodeData(node_id="p:2", kind="process", label="cat", uid=1000, pid=2),
    }
    edges = [
        EdgeData(src="p:1", dst="p:2", label="EXEC", ts=1.0),
        EdgeData(src="p:2", dst="p:1", label="READ", ts=1.0),
    ]
    return Snapshot(window_start=0.0, nodes=nodes, edges=edges, malicious={"p:2"})


def test_make_adj_both_directions():
    g = snapshot_to_graph(_snapshot())
    adj = make_adj(g)
    assert adj.shape == (2, 2)
    assert adj.dtype == torch.float32
    assert adj[0, 0].item() == 0.0
    assert adj[0, 1].item() == 1.0
    assert adj[1, 0].item() == 1.0
    assert adj[1, 1].item() == 0.0
    assert torch.allclose(adj, adj.T)


def test_make_adj_symmetrizes_duplicates():
    nodes = {
        "a": NodeData(node_id="a", kind="file", label="a", uid=0),
        "b": NodeData(node_id="b", kind="file", label="b", uid=0),
    }
    edges = [EdgeData(src="a", dst="b", label="READ", ts=1.0)]
    g = snapshot_to_graph(Snapshot(0.0, nodes, edges, set()))
    adj = make_adj(g)
    assert adj[0, 1].item() == 1.0
    assert adj[1, 0].item() == 1.0


def test_load_pickled_graphs_roundtrip(tmp_path):
    snap = _snapshot()
    data = {
        "snapshots": [snap],
        "run_types": ["normal"],
        "run_ids": ["normal_0"],
    }
    pkl = tmp_path / "graph.pkl"
    with pkl.open("wb") as fh:
        pickle.dump(data, fh)
    graphs, run_types, run_ids = load_pickled_graphs(pkl)
    assert run_types == ["normal"]
    assert run_ids == ["normal_0"]
    assert graphs[0].snapshot_id == 0
    assert graphs[0].node_ids == ["p:1", "p:2"]
    assert graphs[0].malicious == [False, True]


def test_load_pickled_graphs_sequential_ids(tmp_path):
    snaps = [_snapshot(), _snapshot()]
    data = {"snapshots": snaps, "run_types": ["a", "b"], "run_ids": ["r0", "r1"]}
    pkl = tmp_path / "g.pkl"
    with pkl.open("wb") as fh:
        pickle.dump(data, fh)
    graphs, _, _ = load_pickled_graphs(pkl)
    assert [g.snapshot_id for g in graphs] == [0, 1]