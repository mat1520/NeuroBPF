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


def precision_at_k(scores, labels, k):
    order = np.argsort(scores)[::-1][:k]
    return float(sum(1 for i in order if labels[i]) / k)


def recall_at_k(scores, labels, k):
    if k == 0:
        return 1.0
    order = np.argsort(scores)[::-1][:k]
    return float(sum(1 for i in order if labels[i]) / k)


def ndcg_at_k(scores, labels, k):
    order = np.argsort(scores)[::-1]
    gains = np.asarray([1.0 if labels[i] else 0.0 for i in order], dtype=float)[:k]
    ideal = np.sort(np.asarray(labels, dtype=float))[::-1][:k]
    dcg = float(np.sum(gains / np.log2(np.arange(2, len(gains) + 2))))
    idcg = float(np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2))))
    return dcg / idcg if idcg > 0 else 0.0


def fpr_at_budget(scores, labels, budget):
    order = np.argsort(scores)[::-1][:budget]
    benign = sum(1 for i in order if not labels[i])
    total_benign = sum(1 for l in labels if not l)
    return float(benign / total_benign) if total_benign else 0.0


def threshold_report(scores, labels, threshold):
    flags = np.asarray(scores) >= threshold
    y = np.asarray(labels, dtype=bool)
    tp = int((flags & y).sum())
    fp = int((flags & ~y).sum())
    fn = int((~flags & y).sum())
    tn = int((~flags & ~y).sum())
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "tpr": float(tp / (tp + fn)) if tp + fn else 0.0,
        "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
    }


def _group_run_metrics(scores_by_graph, graphs):
    runs = {}
    for i, g in enumerate(graphs):
        runs.setdefault(g.run_id, []).append(i)
    run_scores = []
    run_labels = []
    per_run = {}
    for rid, idxs in runs.items():
        rows = np.stack([np.asarray(scores_by_graph[i]) for i in idxs])
        score = np.max(rows, axis=0)
        labels = np.any(
            np.stack([np.asarray(graphs[i].malicious, dtype=bool) for i in idxs]),
            axis=0,
        )
        run_scores.append(score)
        run_labels.append(labels)
        if labels.sum() > 0:
            per_run[rid] = recall_at_k(score, labels, int(labels.sum()))
    scores = np.concatenate(run_scores)
    labels = np.concatenate(run_labels)
    return scores, labels, per_run


def _paper_metrics(scores, labels):
    n_total = len(scores)
    k1 = max(int(round(n_total * 0.01)), 1)
    k5 = max(int(round(n_total * 0.05)), 1)
    return {
        "auc_roc": float(_roc_auc(scores, labels)),
        "recall_at_1pct": recall_at_k(scores, labels, k1),
        "recall_at_5pct": recall_at_k(scores, labels, k5),
        "precision_at_1pct": precision_at_k(scores, labels, k1),
        "ndcg_at_5pct": ndcg_at_k(scores, labels, k5),
        "fpr_budget_1": fpr_at_budget(scores, labels, 1),
        "fpr_budget_5": fpr_at_budget(scores, labels, 5),
        "fpr_budget_10": fpr_at_budget(scores, labels, 10),
    }


def run_level_evaluate(model, graphs, threshold):
    scores_by_graph = [node_anomaly_scores(model, g) for g in graphs]
    scores, labels, per_run = _group_run_metrics(scores_by_graph, graphs)
    res = _paper_metrics(scores, labels)
    res["threshold_report"] = threshold_report(scores, labels, threshold)
    res["per_run_recall"] = per_run
    res["n_runs"] = len({g.run_id for g in graphs})
    res["n_pos"] = int(labels.sum())
    return res


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
        "threshold_report": threshold_report(all_scores, all_labels, threshold),
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
    runs = {}
    for gi, g in enumerate(graphs):
        rid = run_ids[gi]
        entry = runs.get(rid)
        if entry is None:
            entry = {"run_id": rid, "run_type": run_types[gi], "snapshots": []}
            runs[rid] = entry
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
        entry["snapshots"].append({"ts": ts, "nodes": nodes, "edges": edges})
    return list(runs.values())