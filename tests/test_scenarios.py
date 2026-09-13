import json

from neurobpf.events import Event, read_events
from neurobpf.scenarios import NormalSimulator, PidSource, ScenarioResult, write_run


def test_pid_source_next_and_allocate():
    s = PidSource(base_pid=100)
    assert s.next() == 100
    assert s.allocate() == 101
    assert s.next() == 102


def test_pid_source_default_base():
    s = PidSource()
    assert s.next() == 100


def test_normal_generate_realistic_stream():
    sim = NormalSimulator(seed=1)
    events = sim.generate(duration=30.0)
    assert events
    event_types = {e.event for e in events}
    assert {"process_start", "process_exit", "file_read"} <= event_types
    pids = {e.pid for e in events if e.event == "process_start"}
    assert 100 in pids
    assert 101 in pids
    comms = {e.comm for e in events if e.event == "process_start"}
    assert {"systemd", "sshd", "cron"} <= comms
    assert comms & {"ls", "cat", "grep", "ps", "sort", "sleep"}
    targets = {e.target for e in events if e.event == "process_start"}


def test_normal_timestamps_non_decreasing():
    sim = NormalSimulator(seed=2)
    events = sim.generate(duration=20.0)
    ts = [e.ts for e in events]
    assert ts == sorted(ts)


def test_normal_deterministic_per_seed():
    assert NormalSimulator(seed=7).generate(30.0) == NormalSimulator(seed=7).generate(30.0)
    assert NormalSimulator(seed=7).generate(30.0) != NormalSimulator(seed=8).generate(30.0)


def test_normal_never_connects_to_high_ports():
    sim = NormalSimulator(seed=3)
    events = sim.generate(duration=60.0)
    for e in events:
        if e.event == "net_connect":
            port = int(str(e.target).rsplit(":", 1)[1])
            assert port < 1024


def test_normal_run_result():
    res = NormalSimulator(seed=4).run(10.0)
    assert isinstance(res, ScenarioResult)
    assert res.run_type == "normal"
    assert res.name == "normal"
    assert res.malicious_pids == set()
    assert res.events


def test_write_run_persists_files(tmp_path):
    res = ScenarioResult(
        events=[Event(ts=1.0, event="process_start", pid=1, ppid=0, comm="x")],
        malicious_pids={5, 3},
        name="normal",
        run_type="normal",
    )
    write_run(tmp_path, "run_0", res)
    evf = tmp_path / "run_0.events.ndjson"
    lbf = tmp_path / "run_0.labels.json"
    assert evf.exists()
    assert lbf.exists()
    events = list(read_events(evf))
    assert len(events) == 1
    labels = json.loads(lbf.read_text())
    assert labels["run_type"] == "normal"
    assert labels["name"] == "normal"
    assert sorted(labels["malicious_pids"]) == [3, 5]