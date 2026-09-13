import numpy as np

from neurobpf.gnn.dataset import make_adj
from neurobpf.gnn.features import EDGE_TYPES


def node_anomaly_scores(model, g):
    model.eval()
    scores = model.node_scores(g.x, make_adj(g))
    return scores.detach().cpu().numpy()


def _roc_auc(scores, labels):
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return 1.0
    order = np.argsort(s, kind="mergesort")
    sorted_s = s[order]
    ranks = np.empty_like(sorted_s, dtype=float)
    i = 0
    n = len(s)
    while i < n:
        j = i
        while j + 1 < n and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[i : j + 1] = (i + 1 + j + 1) / 2.0
        i = j + 1
    rank = np.empty(n, dtype=float)
    rank[order] = ranks
    r_pos = float(rank[y].sum())
    return (r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _recall_at_k_for_run(scores, malicious, k):
    if k == 0:
        return 1.0
    order = np.argsort(scores)[::-1][:k]
    hit = sum(1 for i in order if malicious[i])
    return hit / k


def _standardize(scores):
    s = np.asarray(scores, dtype=float)
    std = float(s.std())
    if std < 1e-12:
        return np.zeros_like(s)
    return (s - float(s.mean())) / std


def evaluate(model, graphs, run_types, threshold):
    scores_by_run = [node_anomaly_scores(model, g) for g in graphs]
    z_scores = [_standardize(s) for s in scores_by_run]
    all_scores = []
    all_labels = []
    for g, s in zip(graphs, z_scores):
        all_scores.extend(s.tolist())
        all_labels.extend(g.malicious)
    auc = _roc_auc(all_scores, all_labels)
    run_recalls = {}
    topk_list = []
    for i, (g, s) in enumerate(zip(graphs, scores_by_run)):
        k = sum(1 for m in g.malicious if m)
        recall = _recall_at_k_for_run(s, np.asarray(g.malicious, dtype=bool), k)
        run_recalls[i] = recall
        if k > 0:
            topk_list.append(recall)
    recall_at_topk = float(np.mean(topk_list)) if topk_list else 1.0
    return {
        "auc_roc": float(auc),
        "recall_at_topk": recall_at_topk,
        "per_run_recall": run_recalls,
    }


def _kind_from_id(nid):
    if nid.startswith("p:"):
        return "process"
    if nid.startswith("f:"):
        return "file"
    if nid.startswith("n:"):
        return "net"
    return "unknown"


def _label_from_id(nid):
    if ":" in nid:
        return nid.split(":", 1)[1]
    return nid


def _annotated_node(g, i, score, zscore, threshold):
    nid = g.node_ids[i]
    if g.node_labels:
        label = g.node_labels[i]
    else:
        label = _label_from_id(nid)
    if g.node_kinds:
        kind = g.node_kinds[i]
    else:
        kind = _kind_from_id(nid)
    return {
        "id": nid,
        "label": label,
        "kind": kind,
        "score": float(score),
        "anomalous": bool(float(zscore) > threshold),
    }


def annotate(model, graphs, run_ids, run_types, threshold):
    out = []
    for gi, g in enumerate(graphs):
        raw = node_anomaly_scores(model, g)
        zscores = _standardize(raw)
        lo = float(raw.min())
        hi = float(raw.max())
        span = hi - lo if hi > lo else 1.0
        scores = (raw - lo) / span
        nodes = []
        for i in range(len(g.node_ids)):
            nodes.append(_annotated_node(g, i, scores[i], zscores[i], threshold))
        edges = []
        src, dst = g.edge_index
        for k in range(src.shape[0]):
            edges.append(
                {
                    "src": g.node_ids[src[k].item()],
                    "dst": g.node_ids[dst[k].item()],
                    "label": EDGE_TYPES[g.edge_type[k].item()],
                }
            )
        ts = getattr(g, "ts", 0.0)
        entry = {
            "run_id": run_ids[gi],
            "run_type": run_types[gi],
            "snapshots": [{"ts": ts, "nodes": nodes, "edges": edges}],
        }
        out.append(entry)
    return out