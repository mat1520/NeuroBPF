from dataclasses import dataclass, field

import torch

from neurobpf.graphbuilder import Snapshot

EDGE_TYPES = ("EXEC", "READ", "WRITE", "UNLINK", "CONNECT", "ACCEPT")
FEAT_DIM = 12

KIND_ONE_HOT = {
    "process": (1.0, 0.0, 0.0),
    "file": (0.0, 1.0, 0.0),
    "net": (0.0, 0.0, 1.0),
}


@dataclass
class GraphTensor:
    x: torch.Tensor
    edge_index: torch.LongTensor
    edge_type: torch.LongTensor
    node_ids: list[str]
    malicious: list[bool]
    snapshot_id: int
    node_labels: list[str] = field(default_factory=list)
    node_kinds: list[str] = field(default_factory=list)
    ts: float = 0.0


def snapshot_to_graph(snap: Snapshot, snapshot_id: int = 0) -> GraphTensor:
    node_ids = list(snap.nodes)
    index = {nid: i for i, nid in enumerate(node_ids)}
    n = len(node_ids)
    m = len(snap.edges)
    in_deg = [0] * n
    out_deg = [0] * n
    counts = [[0] * len(EDGE_TYPES) for _ in range(n)]
    edge_index = torch.zeros((2, m), dtype=torch.long)
    edge_type = torch.zeros(m, dtype=torch.long)
    for k, e in enumerate(snap.edges):
        src = index[e.src]
        dst = index[e.dst]
        edge_index[0, k] = src
        edge_index[1, k] = dst
        edge_type[k] = EDGE_TYPES.index(e.label)
        out_deg[src] += 1
        in_deg[dst] += 1
        counts[src][edge_type[k]] += 1
        counts[dst][edge_type[k]] += 1
    max_in = max(max(in_deg), 1)
    max_out = max(max(out_deg), 1)
    all_counts = [c for row in counts for c in row]
    max_count = max(max(all_counts), 1) if all_counts else 1
    x = torch.zeros((n, FEAT_DIM))
    for i, nid in enumerate(node_ids):
        node = snap.nodes[nid]
        x[i, :3] = torch.tensor(KIND_ONE_HOT.get(node.kind, (0.0, 0.0, 0.0)))
        x[i, 3] = 1.0 if node.kind == "process" and node.uid == 0 else 0.0
        x[i, 4] = in_deg[i] / max_in
        x[i, 5] = out_deg[i] / max_out
        for j in range(len(EDGE_TYPES)):
            x[i, 6 + j] = counts[i][j] / max_count
    malicious = [nid in snap.malicious for nid in node_ids]
    return GraphTensor(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        node_ids=node_ids,
        malicious=malicious,
        snapshot_id=snapshot_id,
        node_labels=[snap.nodes[nid].label for nid in node_ids],
        node_kinds=[snap.nodes[nid].kind for nid in node_ids],
        ts=snap.window_start,
    )