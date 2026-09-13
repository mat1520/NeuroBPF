import json

from neurobpf.benchmark.theia import (
    parse_theia_line,
    theia_events_to_ndjson,
    theia_ground_truth_to_pids,
    write_theia_run,
)
from neurobpf.events import event_from_json


def test_parse_theia_event_record():
    line = '{"local:subject":{"pid":123,"process":{"com":"bash"}},"local:op":"execve","subject":123}'
    ev = parse_theia_line(line)
    assert ev is not None
    assert ev.event == "process_start"
    assert ev.pid == 123
    assert ev.comm == "bash"


def test_parse_theia_unsupported_returns_none():
    assert parse_theia_line('{"local:op":"mem_read","a":1}') is None


def test_theia_events_roundtrip_through_event_json():
    e1 = parse_theia_line('{"local:subject":{"pid":1,"process":{"com":"init"}},"local:op":"execve","subject":1}')
    e2 = parse_theia_line('{"local:subject":{"pid":5,"process":{"com":"sshd"}},"local:op":"execve","subject":5}')
    text = theia_events_to_ndjson([e1, e2], "theia_0")
    lines = [event_from_json(l) for l in text.strip().splitlines()]
    assert [e.pid for e in lines] == [1, 5]


def test_write_theia_run_writes_pipeline_readable_files(tmp_path):
    e = parse_theia_line('{"local:subject":{"pid":9,"process":{"com":"bash"}},"local:op":"execve","subject":9}')
    write_theia_run(tmp_path, "theia_0", [e], [9])
    ev = tmp_path / "theia_0.events.ndjson"
    lab = tmp_path / "theia_0.labels.json"
    assert ev.exists() and lab.exists()
    assert event_from_json(ev.read_text().strip()).pid == 9
    assert json.loads(lab.read_text()) == {"run_type": "theia", "malicious_pids": [9]}


def test_theia_ground_truth_pids(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_text("head,pid\nrow,111\nrow,222")
    assert theia_ground_truth_to_pids(p) == [111, 222]