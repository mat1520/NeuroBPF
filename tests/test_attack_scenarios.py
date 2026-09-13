import pytest

from neurobpf.events import Event
from neurobpf.graphbuilder import build_snapshots
from neurobpf.scenarios import NormalSimulator, ScenarioResult, build_attack, generate_dataset

ATTACK_NAMES = ("code_injection", "persistence", "exfiltration", "lateral_movement", "supply_chain")
BASELINE_PIDS = {100, 101, 102, 103, 104}
ENTRY_COMMS = {
    "code_injection": "appserver",
    "persistence": "sh",
    "exfiltration": "svc",
    "lateral_movement": "/usr/bin/scp",
    "supply_chain": "pkgmgr",
}


def test_build_attack_invalid_name_raises():
    with pytest.raises(ValueError):
        build_attack("nope", seed=1)


@pytest.mark.parametrize("name", ATTACK_NAMES)
def test_attack_result_shape_and_determinism(name):
    res = build_attack(name, seed=1)
    assert isinstance(res, ScenarioResult)
    assert res.run_type == "attack"
    assert res.name == name
    assert res.malicious_pids
    assert res.events
    assert build_attack(name, seed=1) == res


@pytest.mark.parametrize("name", ATTACK_NAMES)
def test_attack_spawns_from_background_child(name):
    res = build_attack(name, seed=1)
    entry = ENTRY_COMMS[name]
    spawned = [
        e
        for e in res.events
        if e.event == "process_start" and e.comm == entry and e.ppid in BASELINE_PIDS
    ]
    assert spawned


def test_attack_pids_are_marked_malicious():
    res = build_attack("exfiltration", seed=1)
    svc_pids = {e.pid for e in res.events if e.event == "process_start" and e.comm == "svc"}
    assert svc_pids
    assert svc_pids <= res.malicious_pids


def test_normal_background_still_present_in_attack():
    res = build_attack("supply_chain", seed=1)
    comms = {e.comm for e in res.events if e.event == "process_start"}
    assert {"systemd", "sshd", "cron"} <= comms


def test_code_injection_payload_written_and_executed():
    res = build_attack("code_injection", seed=1)
    writes = [e for e in res.events if e.event == "file_write" and e.target == "/tmp/.payload"]
    execs = [e for e in res.events if e.event == "process_start" and e.comm == "/tmp/.payload"]
    connects = [e for e in res.events if e.event == "net_connect" and e.target == "203.0.113.7:1337"]
    assert writes
    assert execs
    assert connects
    assert execs[0].ppid == writes[0].pid
    payload_pids = {e.pid for e in execs}
    assert payload_pids <= res.malicious_pids
    assert writes[0].pid in res.malicious_pids


def test_persistence_autostart_paths_written():
    res = build_attack("persistence", seed=1)
    writes = {e.target for e in res.events if e.event == "file_write"}
    assert "/home/user/.bashrc" in writes
    assert "/etc/crontab" in writes
    assert "/home/user/.config/systemd/user/p.service" in writes
    assert "/tmp/.bin" in writes
    execs = [e for e in res.events if e.event == "process_start" and e.comm == "/tmp/.bin"]
    assert execs
    assert execs[0].pid in res.malicious_pids


def test_exfiltration_beacon_burst_of_high_port_connects():
    res = build_attack("exfiltration", seed=1)
    burst = [
        e
        for e in res.events
        if e.event == "net_connect"
        and str(e.target).startswith("10.9.9.")
        and int(str(e.target).rsplit(":", 1)[1]) >= 1024
    ]
    assert len(burst) >= 3
    ts = [e.ts for e in burst]
    assert max(ts) - min(ts) <= 2.0
    dns = [e for e in res.events if e.event == "net_connect" and e.target == "198.51.100.66:53"]
    assert dns
    assert len(burst) >= 5


def test_lateral_movement_accept_and_ssh_key_write():
    res = build_attack("lateral_movement", seed=1)
    accepts = [e for e in res.events if e.event == "net_accept" and e.target == "10.1.1.7:22"]
    keys = [
        e
        for e in res.events
        if e.event == "file_write" and e.target == "/home/user/.ssh/authorized_keys"
    ]
    shadow = [e for e in res.events if e.event == "file_read" and e.target == "/etc/shadow"]
    scp = [e for e in res.events if e.event == "process_start" and e.comm == "/usr/bin/scp"]
    assert accepts
    assert keys
    assert shadow
    assert scp
    assert keys[0].pid in res.malicious_pids
    assert scp[0].pid in res.malicious_pids


def test_supply_chain_pkg_exec_and_fake_registry():
    res = build_attack("supply_chain", seed=1)
    connects = {str(e.target) for e in res.events if e.event == "net_connect"}
    assert "registry.evil.example:443" in connects
    assert "github.com" not in connects
    execs = [e for e in res.events if e.event == "process_start" and e.comm == "/usr/local/bin/pkg"]
    assert execs
    pkgmgr = execs[0].ppid
    assert pkgmgr in res.malicious_pids
    tars = [e for e in res.events if e.event == "file_write" and e.target == "/tmp/pkg.tar"]
    assert tars
    assert tars[0].pid == pkgmgr
    c2 = [e for e in res.events if e.event == "net_connect" and e.target == "10.5.5.5:9000"]
    assert c2
    db = [e for e in res.events if e.event == "file_write" and e.target == "/home/user/.cache/pkg.db"]
    assert db


def test_exfiltration_beacon_net_nodes_malicious_in_graph():
    res = build_attack("exfiltration", seed=1)
    snap = build_snapshots(res.events, malicious_pids=res.malicious_pids)[-1]
    beacon_nodes = {n for n in snap.nodes if n.startswith("n:10.9.9.")}
    assert beacon_nodes
    assert beacon_nodes <= snap.malicious


def test_generate_dataset_layout(tmp_path):
    generate_dataset(tmp_path, seed=5, normal_runs=3, attack_runs_per_scenario=2)
    ev_files = sorted(p.name for p in tmp_path.glob("*.events.ndjson"))
    assert len(ev_files) == 3 + 5 * 2
    for i in range(3):
        assert f"normal_{i}.events.ndjson" in ev_files
    for name in ATTACK_NAMES:
        for j in range(2):
            assert f"attack_{name}_{j}.events.ndjson" in ev_files
    labels = sorted(p.name for p in tmp_path.glob("*.labels.json"))
    assert len(labels) == len(ev_files)


def test_generate_dataset_deterministic(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    generate_dataset(a, seed=11, normal_runs=2, attack_runs_per_scenario=1)
    generate_dataset(b, seed=11, normal_runs=2, attack_runs_per_scenario=1)
    for pa in sorted(p.name for p in a.iterdir() if p.name.endswith(".ndjson")):
        assert (a / pa).read_text() == (b / pa).read_text()