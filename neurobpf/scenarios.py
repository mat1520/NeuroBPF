import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from neurobpf.events import Event, write_events

HOME = "/home/user"
BACKGROUND = [("systemd", 1), ("sshd", 1), ("cron", 1), ("bash", 1), ("editor", 1)]
HELPERS = ["ls", "cat", "grep", "ps", "sort", "sleep"]
COMMON_FILES = ["/etc/passwd", "/etc/crontab", "/proc/stat", "/etc/hostname"]
LOG_FILES = ["/var/log/syslog", "/var/log/auth.log"]
STD_NET = [
    ("8.8.8.8", 53),
    ("1.1.1.1", 53),
    ("93.184.216.34", 443),
    ("192.168.10.1", 22),
]


class PidSource:
    def __init__(self, base_pid: int = 100):
        self.next_pid = base_pid

    def next(self) -> int:
        pid = self.next_pid
        self.next_pid += 1
        return pid

    def allocate(self) -> int:
        return self.next()


@dataclass
class ScenarioResult:
    events: list[Event]
    malicious_pids: set[int]
    name: str
    run_type: Literal["normal", "attack"]


def write_run(out_dir: Path, run_id: str, result: ScenarioResult) -> None:
    out_dir = Path(out_dir)
    write_events(out_dir / f"{run_id}.events.ndjson", result.events)
    labels = {
        "run_type": result.run_type,
        "name": result.name,
        "malicious_pids": sorted(result.malicious_pids),
    }
    (out_dir / f"{run_id}.labels.json").write_text(json.dumps(labels))


class NormalSimulator:
    def __init__(self, seed: int):
        self.seed = seed
        self.source = PidSource()
        self._rng = random.Random(seed)

    def generate(self, duration: float = 60.0) -> list[Event]:
        rng = self._rng
        events = []
        t = 0.0
        bg_pids = {}
        for comm, ppid in BACKGROUND:
            pid = self.source.next()
            bg_pids[comm] = pid
            events.append(Event(ts=t, event="process_start", pid=pid, ppid=ppid, comm=comm))
        cron_pid = bg_pids["cron"]
        events.append(
            Event(ts=t, event="file_read", pid=cron_pid, target_type="file", target="/etc/crontab")
        )
        while t < duration:
            t += rng.uniform(0.5, 3.0)
            if t >= duration:
                break
            comm = rng.choice(HELPERS)
            child = self.source.next()
            parent = bg_pids[rng.choice(["bash", "cron", "sshd"])]
            events.append(Event(ts=t, event="process_start", pid=child, ppid=parent, comm=comm))
            events.append(
                Event(
                    ts=t,
                    event="file_read",
                    pid=child,
                    target_type="file",
                    target=rng.choice(COMMON_FILES),
                )
            )
            roll = rng.random()
            if roll < 0.15:
                events.append(
                    Event(
                        ts=t,
                        event="file_write",
                        pid=child,
                        target_type="file",
                        target=rng.choice(LOG_FILES),
                    )
                )
            elif roll < 0.3:
                ip, port = rng.choice(STD_NET)
                events.append(
                    Event(ts=t, event="net_connect", pid=child, target_type="net", target=f"{ip}:{port}")
                )
            events.append(Event(ts=t, event="process_exit", pid=child, ppid=parent, comm=comm))
        return events

    def run(self, duration: float = 60.0) -> ScenarioResult:
        return ScenarioResult(
            events=self.generate(duration),
            malicious_pids=set(),
            name="normal",
            run_type="normal",
        )