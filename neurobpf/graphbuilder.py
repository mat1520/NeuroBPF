from dataclasses import dataclass, field


@dataclass(slots=True)
class NodeData:
    node_id: str
    kind: str
    label: str
    uid: int = 0
    pid: int = -1


@dataclass(slots=True)
class EdgeData:
    src: str
    dst: str
    label: str
    ts: float


@dataclass
class Snapshot:
    window_start: float
    nodes: dict[str, NodeData]
    edges: list[EdgeData]
    malicious: set[str]


class _Builder:
    def __init__(self, malicious_pids):
        self.malicious_pids = set(malicious_pids)
        self.registry: dict[str, NodeData] = {}
        self.edges: list[EdgeData] = []
        self.touches: dict[str, set[int]] = {}

    def add_node(self, node_id, kind, label, event):
        existing = self.registry.get(node_id)
        if existing is None:
            self.registry[node_id] = NodeData(
                node_id=node_id,
                kind=kind,
                label=label,
                uid=event.uid if kind == "process" else 0,
                pid=event.pid if kind == "process" else -1,
            )
        else:
            if kind == "process":
                existing.label = label
                existing.uid = event.uid
                existing.pid = event.pid

    def touch(self, node_id, pid):
        self.touches.setdefault(node_id, set()).add(pid)

    def add_edge(self, src, dst, label, ts):
        self.edges.append(EdgeData(src=src, dst=dst, label=label, ts=ts))

    def is_malicious(self, node_id):
        if node_id.startswith("p:"):
            pid = int(node_id.split(":", 1)[1])
            return pid in self.malicious_pids
        return any(pid in self.malicious_pids for pid in self.touches.get(node_id, ()))

    def apply(self, e: Event):
        pid = e.pid
        if e.event == "process_start":
            self.add_node(f"p:{pid}", "process", e.comm, e)
            if e.ppid > 0:
                parent = f"p:{e.ppid}"
                self.add_node(parent, "process", f"pid {e.ppid}", e)
                self.add_edge(parent, f"p:{pid}", "EXEC", e.ts)
            self.touch(f"p:{pid}", pid)
        elif e.event in ("file_read", "file_write", "file_unlink"):
            node = f"f:{e.target}"
            self.add_node(node, "file", str(e.target), e)
            self.ensure_process(e)
            self.touch(node, pid)
            label = {"file_read": "READ", "file_write": "WRITE", "file_unlink": "UNLINK"}[e.event]
            self.add_edge(f"p:{pid}", node, label, e.ts)
        elif e.event == "file_rename":
            new_node = f"f:{e.target}"
            self.add_node(new_node, "file", str(e.target), e)
            self.ensure_process(e)
            self.touch(new_node, pid)
            self.add_edge(f"p:{pid}", new_node, "WRITE", e.ts)
            if e.exe_path:
                old_node = f"f:{e.exe_path}"
                self.add_node(old_node, "file", e.exe_path, e)
                self.touch(old_node, pid)
                self.add_edge(f"p:{pid}", old_node, "UNLINK", e.ts)
        elif e.event in ("net_connect", "net_accept"):
            node = f"n:{e.target}"
            self.add_node(node, "net", str(e.target), e)
            self.ensure_process(e)
            self.touch(node, pid)
            label = "CONNECT" if e.event == "net_connect" else "ACCEPT"
            self.add_edge(f"p:{pid}", node, label, e.ts)

    def ensure_process(self, e: Event):
        nid = f"p:{e.pid}"
        if nid not in self.registry:
            self.add_node(nid, "process", e.comm or f"pid {e.pid}", e)

    def snapshot(self, window_start):
        return Snapshot(
            window_start=window_start,
            nodes=dict(self.registry),
            edges=list(self.edges),
            malicious={n for n in self.registry if self.is_malicious(n)},
        )


def build_snapshots(events, window: float = 5.0, malicious_pids=()):
    events = list(events)
    builder = _Builder(malicious_pids)
    start_ts = events[0].ts if events else 0.0
    cur_start = start_ts
    boundary = cur_start + window
    snapshots = []
    for e in events:
        while e.ts >= boundary:
            snapshots.append(builder.snapshot(cur_start))
            cur_start = boundary
            boundary += window
        builder.apply(e)
    snapshots.append(builder.snapshot(cur_start))
    return snapshots