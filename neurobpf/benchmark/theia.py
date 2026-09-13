import csv
import json
from pathlib import Path

from neurobpf.events import Event, event_to_json

_OP_TO_EVENT = {
    "execve": "process_start",
    "open": "file_read",
    "read": "file_read",
    "write": "file_write",
    "unlink": "file_unlink",
    "connect": "net_connect",
}


def parse_theia_line(line):
    rec = None
    try:
        rec = json.loads(line)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(rec, dict):
        return None
    op = rec.get("local:op") or rec.get("op") or ""
    if op not in _OP_TO_EVENT:
        return None
    subject = rec.get("subject")
    raw = subject if isinstance(subject, dict) else rec.get("local:subject")
    if isinstance(raw, dict) and "subject" in raw:
        raw = raw["subject"]
    if not isinstance(raw, dict):
        return None
    pid = raw.get("pid")
    if pid is None:
        return None
    proc = raw.get("process") if isinstance(raw.get("process"), dict) else {}
    comm = proc.get("com", "") if isinstance(proc, dict) else ""
    args = rec.get("arguments")
    target = ""
    if isinstance(args, dict):
        target = args.get("subject path") or args.get("dst") or args.get("path") or ""
    name = _OP_TO_EVENT[op]
    return Event(
        ts=float(rec.get("ts", 0.0)),
        event=name,
        pid=int(pid),
        comm=str(comm),
        target_type="process" if name == "process_start" else ("net" if "net" in name else "file"),
        target=str(target) or None,
    )


def theia_events_to_ndjson(events, run_id):
    return "\n".join(event_to_json(e).rstrip("\n") for e in events) + "\n"


def write_theia_run(out_dir, run_id, events, malicious_pids):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{run_id}.events.ndjson").write_text(theia_events_to_ndjson(events, run_id))
    (out_dir / f"{run_id}.labels.json").write_text(
        json.dumps({"run_type": "theia", "malicious_pids": list(malicious_pids)})
    )


def theia_ground_truth_to_pids(path):
    pids = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            pid = row.get("pid")
            if pid is not None:
                try:
                    pids.append(int(pid))
                except ValueError:
                    continue
    return pids