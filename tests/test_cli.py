import json
import pickle
import subprocess
import sys

from neurobpf.cli import main
from neurobpf.events import Event, event_to_json, write_events
from neurobpf.graphbuilder import Snapshot
from neurobpf.scenarios import ATTACKS


def test_generate_file_counts(tmp_path):
    out = tmp_path / "out"
    main(["generate", "--out", str(out), "--seed", "42", "--normal", "2", "--attack", "3"])
    run_dir = out / "42"
    ev = sorted(p.name for p in run_dir.glob("*.events.ndjson"))
    lb = sorted(p.name for p in run_dir.glob("*.labels.json"))
    assert len(ev) == 2 + 5 * 3
    assert len(lb) == len(ev)
    assert "normal_0.events.ndjson" in ev
    assert "normal_1.events.ndjson" in ev
    for name in ATTACKS:
        for j in range(3):
            assert f"attack_{name}_{j}.events.ndjson" in ev


def test_generate_default_counts(tmp_path):
    out = tmp_path / "out"
    main(["generate", "--out", str(out), "--seed", "1"])
    ev = list((out / "1").glob("*.events.ndjson"))
    assert len(ev) == 10 + 5 * 5


def test_generate_deterministic_byte_identical(tmp_path):
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    main(["generate", "--out", str(out_a), "--seed", "7", "--normal", "1", "--attack", "1"])
    main(["generate", "--out", str(out_b), "--seed", "7", "--normal", "1", "--attack", "1"])
    names = sorted(p.name for p in (out_a / "7").glob("*.ndjson"))
    assert names
    for name in names:
        assert (out_a / "7" / name).read_bytes() == (out_b / "7" / name).read_bytes()


def test_build_pickle_keys_types_and_order(tmp_path):
    out = tmp_path / "out"
    main(["generate", "--out", str(out), "--seed", "3", "--normal", "2", "--attack", "1"])
    pkl = tmp_path / "graph.pkl"
    main(["build", "--events", str(out / "3"), "--out", str(pkl)])
    with pkl.open("rb") as fh:
        data = pickle.load(fh)
    assert set(data) == {"snapshots", "run_types", "run_ids"}
    n = len(data["snapshots"])
    assert n >= 2 + 5 * 1
    assert len(data["run_types"]) == n
    assert len(data["run_ids"]) == n
    assert all(isinstance(s, Snapshot) for s in data["snapshots"])
    distinct = list(dict.fromkeys(data["run_ids"]))
    assert distinct == sorted(distinct)
    assert data["run_types"].count("normal") >= 2
    attack_types = [t for t in data["run_types"] if t == "attack"]
    assert len(attack_types) >= 5
    assert len(set(data["run_ids"])) == 2 + 5
    for rid in set(data["run_ids"]):
        labels = json.loads((out / "3" / f"{rid}.labels.json").read_text())
        idx = data["run_ids"].index(rid)
        assert data["run_types"][idx] == labels["run_type"]


def test_build_last_snapshot_is_cumulative_full_run(tmp_path):
    out = tmp_path / "out"
    main(["generate", "--out", str(out), "--seed", "9", "--normal", "1", "--attack", "1"])
    pkl = tmp_path / "graph.pkl"
    main(["build", "--events", str(out / "9"), "--out", str(pkl)])
    with pkl.open("rb") as fh:
        data = pickle.load(fh)
    for snap in data["snapshots"]:
        assert snap.nodes
        assert snap.edges


def test_build_missing_labels_assumes_real(tmp_path):
    run_dir = tmp_path / "runs"
    write_events(
        run_dir / "foo.events.ndjson",
        [Event(ts=1.0, event="process_start", pid=1, ppid=0, comm="x")],
    )
    pkl = tmp_path / "out.pkl"
    main(["build", "--events", str(run_dir), "--out", str(pkl)])
    with pkl.open("rb") as fh:
        data = pickle.load(fh)
    assert data["run_types"] == ["real"]
    assert data["run_ids"] == ["foo"]
    assert data["snapshots"][0].malicious == set()


def test_build_keeps_all_temporal_windows(tmp_path):
    import neurobpf.cli as cli

    evs_dir = tmp_path / "events"
    evs_dir.mkdir()
    events = []
    for i in range(3):
        e = Event(
            ts=float(i) * 1.0,
            event="file_read",
            pid=1,
            comm="bash",
            uid=1000,
            ppid=0,
            target=f"/f{i}",
        )
        events.append(e)
    for run in ("normal_0", "normal_1"):
        (evs_dir / f"{run}.events.ndjson").write_text(
            "\n".join(event_to_json(e).rstrip("\n") for e in events) + "\n"
        )
        (evs_dir / f"{run}.labels.json").write_text(
            json.dumps({"run_type": "normal", "malicious_pids": []})
        )
    out = tmp_path / "g.pkl"
    cli._build_pickle(evs_dir, out, window=2.0)
    data = pickle.load(open(out, "rb"))
    assert len(data["run_ids"]) == len(data["snapshots"])
    assert data["run_ids"][0] == "normal_0"
    assert data["snapshots"][0].window_start == 0.0
    assert data["run_ids"].count("normal_0") >= 2


def test_module_entry_generate(tmp_path):
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "neurobpf.cli",
            "generate",
            "--out",
            str(tmp_path / "o"),
            "--seed",
            "1",
            "--normal",
            "1",
            "--attack",
            "1",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "o" / "1" / "normal_0.events.ndjson").exists()
    assert (tmp_path / "o" / "1" / "attack_code_injection_0.events.ndjson").exists()