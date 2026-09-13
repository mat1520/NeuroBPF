import json
import sys
from pathlib import Path

SCHEMA_KEYS = {
    "ts",
    "event",
    "pid",
    "ppid",
    "comm",
    "exe_path",
    "uid",
    "target_type",
    "target",
    "result",
}
EVENT_KINDS = {
    "process_start",
    "process_exit",
    "file_read",
    "file_write",
    "net_connect",
    "net_accept",
}
TARGET_TYPES = {None, "file", "process", "net"}


def is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def main():
    sample = Path(__file__).resolve().parent / "sample_events.ndjson"
    lines = [ln for ln in sample.read_text().splitlines() if ln.strip()]
    if not lines:
        raise SystemExit("sample_events.ndjson is empty")
    kinds_seen = set()
    for ln in lines:
        ev = json.loads(ln)
        if not isinstance(ev, dict):
            raise SystemExit(f"not an object: {ln}")
        if set(ev) != SCHEMA_KEYS:
            raise SystemExit(
                "key mismatch: "
                f"got {sorted(set(ev))}, want {sorted(SCHEMA_KEYS)}"
            )
        if not isinstance(ev["ts"], (int, float)) or isinstance(ev["ts"], bool):
            raise SystemExit("ts must be a float seconds: " + ln)
        if ev["event"] not in EVENT_KINDS:
            raise SystemExit("unknown event kind: " + ln)
        if not is_int(ev["pid"]) or not is_int(ev["ppid"]):
            raise SystemExit("pid/ppid must be int: " + ln)
        if not isinstance(ev["comm"], str) or not isinstance(ev["exe_path"], str):
            raise SystemExit("comm/exe_path must be str: " + ln)
        if not is_int(ev["uid"]) or not is_int(ev["result"]):
            raise SystemExit("uid/result must be int: " + ln)
        if ev["target_type"] not in TARGET_TYPES:
            raise SystemExit("bad target_type: " + ln)
        if ev["target_type"] is None and ev["target"] is not None:
            raise SystemExit("null target_type must have null target: " + ln)
        kinds_seen.add(ev["event"])
    if kinds_seen != EVENT_KINDS:
        raise SystemExit(f"missing event kinds: {sorted(EVENT_KINDS - kinds_seen)}")
    print(f"schema OK: {len(lines)} events, {len(kinds_seen)} kinds")
    return 0


if __name__ == "__main__":
    sys.exit(main())