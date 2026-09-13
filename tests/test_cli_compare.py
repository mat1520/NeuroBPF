import json
import pickle

import neurobpf.cli as cli
from neurobpf.graphbuilder import EdgeData, NodeData, Snapshot


def _snap(malicious):
    nodes = {"hub": NodeData("hub", "process", "hub", 1000, 1)}
    edges = []
    for k in range(4):
        nid = f"leaf{k}"
        nodes[nid] = NodeData(nid, "process", nid, 1000, k + 2)
        edges.append(EdgeData("hub", nid, "EXEC", 1.0))
    return Snapshot(0.0, nodes, edges, malicious)


def _pkl(tmp_path):
    runs = []
    for i in range(6):
        runs.append((f"n{i}", "normal", _snap(set())))
    for i in range(3):
        runs.append((f"a{i}", "attack", _snap({"leaf1"})))
    data = {
        "snapshots": [s for _, _, s in runs],
        "run_types": [t for _, t, _ in runs],
        "run_ids": [r for r, _, _ in runs],
    }
    p = tmp_path / "g.pkl"
    with p.open("wb") as fh:
        pickle.dump(data, fh)
    return p


def test_compare_writes_report(tmp_path):
    p = _pkl(tmp_path)
    out = tmp_path / "rep.json"
    cli.compare(p, out, seeds=1)
    rep = json.loads(out.read_text())
    assert set(rep) == {"per_seed", "summary"}
    assert rep["per_seed"][0]["method"] in {"gnn", "gnn_shuffled", "ocsvm", "mlp_ae"}
    methods = {row["method"] for row in rep["per_seed"]}
    assert methods == {"gnn", "gnn_shuffled", "ocsvm", "mlp_ae"}