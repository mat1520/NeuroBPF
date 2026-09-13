import torch

from neurobpf.gnn.features import EDGE_TYPES, FEAT_DIM, GraphTensor, snapshot_to_graph
from neurobpf.graphbuilder import EdgeData, NodeData, Snapshot


def _snapshot():
    nodes = {
        "p:1": NodeData(node_id="p:1", kind="process", label="bash", uid=1000, pid=1),
        "p:2": NodeData(node_id="p:2", kind="process", label="daemon", uid=0, pid=2),
        "f:/etc/passwd": NodeData(node_id="f:/etc/passwd", kind="file", label="/etc/passwd", uid=0),
        "f:/tmp/x": NodeData(node_id="f:/tmp/x", kind="file", label="/tmp/x", uid=0),
        "f:/old": NodeData(node_id="f:/old", kind="file", label="/old", uid=0),
        "n:8.8.8.8:53": NodeData(node_id="n:8.8.8.8:53", kind="net", label="8.8.8.8:53", uid=0),
    }
    edges = [
        EdgeData(src="p:1", dst="p:2", label="EXEC", ts=1.0),
        EdgeData(src="p:1", dst="f:/etc/passwd", label="READ", ts=1.0),
        EdgeData(src="p:1", dst="f:/tmp/x", label="WRITE", ts=1.0),
        EdgeData(src="p:2", dst="f:/tmp/x", label="WRITE", ts=1.1),
        EdgeData(src="p:2", dst="f:/old", label="UNLINK", ts=1.2),
        EdgeData(src="p:1", dst="n:8.8.8.8:53", label="CONNECT", ts=1.3),
        EdgeData(src="p:2", dst="n:8.8.8.8:53", label="ACCEPT", ts=1.4),
    ]
    snap = Snapshot(window_start=5.0, nodes=nodes, edges=edges, malicious={"p:2", "f:/old"})
    return snap


def _ids(snap):
    return list(snap.nodes)


def test_constants():
    assert EDGE_TYPES == ("EXEC", "READ", "WRITE", "UNLINK", "CONNECT", "ACCEPT")
    assert FEAT_DIM == 12


def test_graph_tensor_structure():
    snap = _snapshot()
    g = snapshot_to_graph(snap, snapshot_id=7)
    assert isinstance(g, GraphTensor)
    n = len(snap.nodes)
    m = len(snap.edges)
    assert g.x.shape == (n, FEAT_DIM)
    assert g.x.dtype == torch.float32
    assert g.edge_index.shape == (2, m)
    assert g.edge_index.dtype == torch.long
    assert g.edge_type.shape == (m,)
    assert g.node_ids == _ids(snap)
    assert g.snapshot_id == 7
    assert g.malicious == [False, True, False, False, True, False]


def test_kind_one_hot():
    g = snapshot_to_graph(_snapshot())
    kinds = g.x[:, :3]
    assert torch.allclose(kinds.sum(dim=1), torch.ones(kinds.shape[0]))
    assert kinds[0].tolist() == [1.0, 0.0, 0.0]
    assert kinds[3].tolist() == [0.0, 1.0, 0.0]
    assert kinds[5].tolist() == [0.0, 0.0, 1.0]


def test_is_root():
    g = snapshot_to_graph(_snapshot())
    assert g.x[0, 3].item() == 0.0
    assert g.x[1, 3].item() == 1.0
    assert g.x[2, 3].item() == 1.0


def test_degree_norms():
    g = snapshot_to_graph(_snapshot())
    assert g.x[1, 4].item() == 0.5
    assert g.x[1, 5].item() == 0.75
    assert g.x[2, 4].item() == 0.5
    assert g.x[3, 4].item() == 1.0
    assert g.x[3, 5].item() == 0.0
    assert g.x[0, 4].item() == 0.0


def test_edge_type_counts():
    g = snapshot_to_graph(_snapshot())
    read_idx = 6 + EDGE_TYPES.index("READ")
    write_idx = 6 + EDGE_TYPES.index("WRITE")
    accept_idx = 6 + EDGE_TYPES.index("ACCEPT")
    assert g.x[2, read_idx].item() == 0.5
    assert g.x[3, write_idx].item() == 1.0
    assert g.x[5, accept_idx].item() == 0.5


def test_edge_index_maps_src_dst():
    g = snapshot_to_graph(_snapshot())
    ids = {nid: i for i, nid in enumerate(g.node_ids)}
    assert g.edge_index[0, 0].item() == ids["p:1"]
    assert g.edge_index[1, 0].item() == ids["p:2"]
    assert g.edge_type[0].item() == EDGE_TYPES.index("EXEC")
    m = g.edge_index.shape[1]
    for k in range(m):
        assert 0 <= g.edge_index[0, k].item() < len(g.node_ids)
        assert 0 <= g.edge_index[1, k].item() < len(g.node_ids)


def test_single_node_snapshot_no_crash():
    snap = Snapshot(
        window_start=0.0,
        nodes={"p:1": NodeData(node_id="p:1", kind="process", label="bash", uid=1000, pid=1)},
        edges=[],
        malicious=set(),
    )
    g = snapshot_to_graph(snap)
    assert g.x.shape == (1, FEAT_DIM)
    assert torch.all(torch.isfinite(g.x))


def test_node_feature_dims_are_bounded():
    g = snapshot_to_graph(_snapshot())
    assert (g.x >= 0.0).all()
    assert (g.x <= 1.0).all()