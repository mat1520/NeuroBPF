from neurobpf.events import Event
from neurobpf.graphbuilder import build_snapshots


def test_action_before_process_start_creates_process_node():
    evs = [
        Event(ts=0.0, event="file_read", pid=5555, target_type="file", target="/etc/passwd"),
        Event(ts=0.5, event="net_connect", pid=5555, target_type="net", target="1.1.1.1:443"),
    ]
    snaps = build_snapshots(evs, window=5.0)
    assert "p:5555" in snaps[0].nodes
    assert snaps[0].nodes["p:5555"].kind == "process"
    labels = {(e.src, e.dst): e.label for e in snaps[0].edges}
    assert labels == {
        ("p:5555", "f:/etc/passwd"): "READ",
        ("p:5555", "n:1.1.1.1:443"): "CONNECT",
    }


def test_action_before_parent_start_creates_parent_process_node():
    evs = [
        Event(ts=0.0, event="process_start", pid=7777, ppid=7776, comm="sh", exe_path="/bin/sh"),
        Event(ts=0.5, event="file_write", pid=7777, target_type="file", target="/tmp/x"),
    ]
    snaps = build_snapshots(evs, window=5.0)
    assert "p:7776" in snaps[0].nodes
    assert ("p:7776", "p:7777") in {(e.src, e.dst) for e in snaps[0].edges}