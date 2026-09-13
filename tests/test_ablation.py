import torch

from neurobpf.gnn import ablation
from neurobpf.gnn.features import snapshot_to_graph
from neurobpf.graphbuilder import EdgeData, NodeData, Snapshot


def _graph():
    nodes = {"hub": NodeData("hub", "process", "hub", 1000, 1)}
    edges = []
    for k in range(4):
        nid = f"leaf{k}"
        nodes[nid] = NodeData(nid, "process", nid, 1000, k + 2)
        edges.append(EdgeData("hub", nid, "EXEC", 1.0))
    return Snapshot(0.0, nodes, edges, {"leaf1", "leaf3"})


def test_shuffled_graphs_preserve_nodes_but_permute_adjacency():
    g = snapshot_to_graph(_graph(), 0, run_id="r0")
    out = ablation.shuffled_graphs([g], seed=0)
    assert len(out) == 1
    s = out[0]
    assert s.node_ids == g.node_ids
    assert s.malicious == g.malicious
    assert not torch.equal(s.edge_index, g.edge_index)


def test_shuffled_deterministic_per_seed():
    g = snapshot_to_graph(_graph(), 0, run_id="r0")
    a = ablation.shuffled_graphs([g], seed=7)[0]
    b = ablation.shuffled_graphs([g], seed=7)[0]
    assert torch.equal(a.edge_index, b.edge_index)