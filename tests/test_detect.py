import numpy as np
import torch

from neurobpf.gnn.detect import (
    _roc_auc,
    annotate,
    evaluate,
    node_anomaly_scores,
)
from neurobpf.gnn.features import FEAT_DIM, GraphTensor


def _graph(x, adj, node_ids, malicious, snapshot_id):
    pairs = torch.tensor(adj, dtype=torch.long)
    return GraphTensor(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=pairs.T,
        edge_type=torch.zeros(pairs.shape[0], dtype=torch.long),
        node_ids=node_ids,
        malicious=malicious,
        snapshot_id=snapshot_id,
    )


class FakeModel:
    def __init__(self, score_rows):
        self._rows = [torch.tensor(r, dtype=torch.float32) for r in score_rows]
        self.evals = 0

    def eval(self):
        self.evals += 1
        return self

    def node_scores(self, x, adj):
        return self._rows.pop(0).reshape(x.shape[0])


def _fake_model_for(graphs, per_graph_score_lists):
    return FakeModel(per_graph_score_lists)


def test_node_anomaly_scores_numpy():
    g = _graph(
        [[1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * 3,
        [[0, 1], [0, 2]],
        ["a", "b", "c"],
        [False, False, False],
        0,
    )
    model = FakeModel([[0.1, 0.5, 0.9]])
    scores = node_anomaly_scores(model, g)
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (3,)
    assert scores[0] == 0.1


def test_roc_auc_known_order():
    scores = [0.1, 0.2, 0.3, 0.4, 0.9]
    labels = [0, 1, 0, 1, 1]
    auc = _roc_auc(scores, labels)
    assert abs(auc - 5.0 / 6.0) < 1e-9


def test_roc_auc_ties():
    scores = [0.5, 0.5, 0.2, 0.7]
    labels = [1, 0, 0, 1]
    auc = _roc_auc(scores, labels)
    assert abs(auc - 0.875) < 1e-9


def test_roc_auc_single_class():
    assert _roc_auc([0.1, 0.2], [1, 1]) == 1.0
    assert _roc_auc([0.1, 0.2], [0, 0]) == 1.0


def _eval_inputs():
    g0 = _graph(
        [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * 2,
        [[0, 1]],
        ["a", "b"],
        [False, False],
        0,
    )
    g1 = _graph(
        [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * 3,
        [[0, 1], [1, 2]],
        ["c", "d", "e"],
        [True, False, True],
        1,
    )
    g2 = _graph(
        [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * 3,
        [[0, 1], [1, 2]],
        ["f", "g", "h"],
        [True, False, False],
        2,
    )
    graphs = [g0, g1, g2]
    scores = [[0.1, 0.2], [0.9, 0.1, 0.95], [0.1, 0.9, 0.8]]
    return graphs, scores


def test_evaluate_shapes_and_values():
    graphs, scores = _eval_inputs()
    model = _fake_model_for(graphs, scores)
    res = evaluate(model, graphs, ["normal", "attack", "attack"], threshold=0.5)
    assert set(res) >= {"auc_roc", "recall_at_topk", "per_run_recall"}
    assert abs(res["auc_roc"] - 7.0 / 15.0) < 1e-9
    assert abs(res["recall_at_topk"] - 0.5) < 1e-9
    assert res["per_run_recall"][1] == 1.0
    assert res["per_run_recall"][2] == 0.0


def test_evaluate_single_class_auc_safe():
    g = _graph(
        [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * 2,
        [[0, 1]],
        ["a", "b"],
        [False, False],
        0,
    )
    model = FakeModel([[0.1, 0.2]])
    res = evaluate(model, [g], ["normal"], 0.5)
    assert res["auc_roc"] == 1.0


def test_annotate_global_shift_not_flagged_as_anomalous():
    g = _graph(
        [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * 4,
        [[0, 1], [1, 2]],
        ["a", "b", "c", "d"],
        [False, False, True, False],
        0,
    )
    model = FakeModel([[100.0, 100.0, 300.0, 100.0]])
    out = annotate(model, [g], ["r0"], ["attack"], 1.0)
    flag = [n["anomalous"] for n in out[0]["snapshots"][0]["nodes"]]
    assert flag == [False, False, True, False]


def test_annotate_shape_and_normalization():
    graphs, scores = _eval_inputs()
    model = _fake_model_for(graphs, scores)
    out = annotate(model, graphs, ["r0", "r1", "r2"], ["normal", "attack", "attack"], 0.5)
    assert len(out) == 3
    e0 = out[0]
    assert set(e0) == {"run_id", "run_type", "snapshots"}
    assert e0["run_id"] == "r0"
    assert e0["run_type"] == "normal"
    assert len(e0["snapshots"]) == 1
    snap = e0["snapshots"][0]
    assert "ts" in snap
    node = snap["nodes"][0]
    assert set(node) == {"id", "label", "kind", "score", "anomalous"}
    edge = snap["edges"][0]
    assert set(edge) == {"src", "dst", "label"}
    norm = [n["score"] for n in snap["nodes"]]
    assert min(norm) >= 0.0
    assert max(norm) <= 1.0
    assert norm[0] == 0.0
    assert norm[1] == 1.0
    e2 = out[2]
    snap2 = e2["snapshots"][0]
    nodes2 = snap2["nodes"]
    assert nodes2[0]["anomalous"] is False
    assert nodes2[1]["anomalous"] is True
    assert nodes2[2]["anomalous"] is True