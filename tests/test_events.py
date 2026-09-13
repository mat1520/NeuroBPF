import json
from pathlib import Path

import pytest

from neurobpf.events import Event, event_from_json, event_to_json, read_events, write_events


def test_roundtrip_equality():
    e = Event(ts=1.5, event="process_start", pid=100, ppid=1, comm="systemd")
    assert event_from_json(event_to_json(e)) == e


def test_roundtrip_file_event_target_string():
    e = Event(
        ts=2.0,
        event="file_write",
        pid=100,
        target_type="file",
        target="/etc/passwd",
        result=0,
    )
    assert event_from_json(event_to_json(e)) == e


def test_roundtrip_net_event_target_string():
    e = Event(
        ts=3.0,
        event="net_connect",
        pid=100,
        target_type="net",
        target="203.0.113.7:1337",
    )
    assert event_from_json(event_to_json(e)) == e


def test_event_to_json_compact_sorted():
    e = Event(ts=1.0, event="process_start", pid=100)
    line = event_to_json(e)
    dumped = json.dumps(
        {
            "ts": 1.0,
            "event": "process_start",
            "pid": 100,
            "ppid": -1,
            "comm": "",
            "exe_path": "",
            "uid": 1000,
            "target_type": None,
            "target": None,
            "result": 0,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    assert line == dumped + "\n"
    assert line.endswith("\n")


def test_read_events_streams_in_order(tmp_path):
    path = tmp_path / "events.ndjson"
    events = [
        Event(ts=1.0, event="process_start", pid=100),
        Event(ts=2.0, event="file_read", pid=100, target_type="file", target="/etc/passwd"),
        Event(ts=3.0, event="net_connect", pid=100, target_type="net", target="1.2.3.4:443"),
    ]
    write_events(path, events)
    assert [e for e in read_events(path)] == events


def test_read_events_skips_blank_lines(tmp_path):
    path = tmp_path / "events.ndjson"
    path.write_text("\n" + event_to_json(Event(ts=1.0, event="process_start", pid=100)) + "\n\n")
    events = list(read_events(path))
    assert len(events) == 1
    assert events[0].event == "process_start"


def test_read_events_malformed_line_raises(tmp_path):
    path = tmp_path / "events.ndjson"
    path.write_text("not json\n")
    with pytest.raises(ValueError):
        list(read_events(path))