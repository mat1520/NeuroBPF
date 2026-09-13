import pytest

from neurobpf.events import Event
from neurobpf.graphbuilder import EdgeData, NodeData, Snapshot, build_snapshots


def test_process_chain_exec_edges():
    events = [
        Event(ts=1.0, event="process_start", pid=100, ppid=1, comm="bash"),
        Event(ts=1.1, event="process_start", pid=101, ppid=100, comm="ls"),
        Event(ts=1.2, event="process_start", pid=102, ppid=101, comm="cat"),
    ]
    snap = build_snapshots(events)[-1]
    exec_edges = {(e.src, e.dst) for e in snap.edges if e.label == "EXEC"}
    assert ("p:1", "p:100") in exec_edges
    assert ("p:100", "p:101") in exec_edges
    assert ("p:101", "p:102") in exec_edges
    assert all(n.kind == "process" for n in snap.nodes.values())
    assert set(snap.nodes) == {"p:1", "p:100", "p:101", "p:102"}


def test_no_exec_edge_when_parent_absent():
    events = [Event(ts=1.0, event="process_start", pid=100, ppid=-1, comm="bash")]
    snap = build_snapshots(events)[-1]
    assert all(e.label != "EXEC" for e in snap.edges)


def test_file_and_net_nodes_with_correct_edges():
    events = [
        Event(ts=1.0, event="file_write", pid=100, target_type="file", target="/tmp/x"),
        Event(ts=1.1, event="file_read", pid=100, target_type="file", target="/etc/passwd"),
        Event(ts=1.2, event="net_connect", pid=100, target_type="net", target="1.2.3.4:53"),
        Event(ts=1.3, event="net_accept", pid=101, target_type="net", target="9.8.7.6:22"),
    ]
    snap = build_snapshots(events)[-1]
    assert snap.nodes["f:/tmp/x"].kind == "file"
    assert snap.nodes["n:1.2.3.4:53"].kind == "net"
    labels = {(e.src, e.dst, e.label) for e in snap.edges}
    assert ("p:100", "f:/tmp/x", "WRITE") in labels
    assert ("p:100", "f:/etc/passwd", "READ") in labels
    assert ("p:100", "n:1.2.3.4:53", "CONNECT") in labels
    assert ("p:101", "n:9.8.7.6:22", "ACCEPT") in labels


def test_file_rename_edges():
    events = [
        Event(
            ts=1.0,
            event="file_rename",
            pid=100,
            target_type="file",
            target="/tmp/new",
            exe_path="/tmp/old",
        )
    ]
    snap = build_snapshots(events)[-1]
    assert "f:/tmp/new" in snap.nodes
    assert "f:/tmp/old" in snap.nodes
    labels = {(e.src, e.dst, e.label) for e in snap.edges}
    assert ("p:100", "f:/tmp/new", "WRITE") in labels
    assert ("p:100", "f:/tmp/old", "UNLINK") in labels


def test_net_node_name_uses_ip_and_port():
    events = [Event(ts=1.0, event="net_connect", pid=100, target_type="net", target="10.9.9.5:40001")]
    snap = build_snapshots(events)[-1]
    assert "n:10.9.9.5:40001" in snap.nodes


def test_empty_events_still_emit_snapshot():
    snaps = build_snapshots([])
    assert len(snaps) == 1
    assert snaps[0].nodes == {}
    assert snaps[0].edges == []
    assert snaps[0].malicious == set()


def test_malicious_marking_propagates_from_pid():
    events = [
        Event(ts=1.0, event="process_start", pid=100, ppid=1, comm="evil"),
        Event(ts=1.1, event="file_write", pid=100, target_type="file", target="/etc/passwd"),
        Event(ts=1.2, event="file_read", pid=100, target_type="file", target="/etc/shadow"),
        Event(ts=1.3, event="net_connect", pid=100, target_type="net", target="5.6.7.8:443"),
    ]
    snap = build_snapshots(events, malicious_pids=[100])[-1]
    assert "p:100" in snap.malicious
    assert "f:/etc/passwd" in snap.malicious
    assert "f:/etc/shadow" in snap.malicious
    assert "n:5.6.7.8:443" in snap.malicious


def test_benign_process_not_marked():
    events = [
        Event(ts=1.0, event="process_start", pid=100, ppid=1, comm="good"),
        Event(ts=1.1, event="file_read", pid=100, target_type="file", target="/etc/passwd"),
    ]
    snap = build_snapshots(events, malicious_pids=[999])[-1]
    assert "p:100" not in snap.malicious
    assert "f:/etc/passwd" not in snap.malicious


def test_file_touched_by_malicious_and_benign_pid_is_malicious():
    events = [
        Event(ts=1.0, event="file_read", pid=200, target_type="file", target="/etc/passwd"),
        Event(ts=1.1, event="file_read", pid=300, target_type="file", target="/etc/passwd"),
    ]
    snap = build_snapshots(events, malicious_pids=[300])[-1]
    assert "f:/etc/passwd" in snap.malicious


def test_window_splitting_keeps_process_node_cumulative():
    events = [
        Event(ts=1.0, event="process_start", pid=100, ppid=1, comm="bash"),
        Event(ts=6.0, event="file_write", pid=100, target_type="file", target="/tmp/x"),
        Event(ts=6.1, event="net_connect", pid=100, target_type="net", target="1.2.3.4:443"),
    ]
    snaps = build_snapshots(events, window=5.0)
    assert len(snaps) == 2
    first, second = snaps
    assert "p:100" in first.nodes
    assert "f:/tmp/x" not in first.nodes
    assert "f:/tmp/x" in second.nodes
    assert "n:1.2.3.4:443" in second.nodes
    assert set(first.nodes).issubset(set(second.nodes))


def test_snapshot_records_window_start():
    events = [
        Event(ts=10.0, event="process_start", pid=100, ppid=1, comm="bash"),
        Event(ts=16.0, event="process_start", pid=101, ppid=100, comm="ls"),
    ]
    snaps = build_snapshots(events, window=5.0)
    assert snaps[0].window_start == 10.0
    assert snaps[1].window_start == 15.0
    assert "p:100" in snaps[1].nodes
    assert "p:101" in snaps[1].nodes