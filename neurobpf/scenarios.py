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


ATTACKS = ["code_injection", "persistence", "exfiltration", "lateral_movement", "supply_chain"]
BG_DURATION = 12.0
HIGH_PORT_BASE = 40000
HIGH_PORT_SPAN = 25000


def _attack_code_injection(source, t, pids):
    events = []
    appserver = source.next()
    sh = source.next()
    payload = source.next()
    events.append(
        Event(ts=t, event="process_start", pid=appserver, ppid=pids["systemd"], comm="appserver")
    )
    events.append(Event(ts=t + 0.2, event="process_start", pid=sh, ppid=appserver, comm="sh"))
    events.append(
        Event(
            ts=t + 0.3,
            event="file_write",
            pid=sh,
            target_type="file",
            target="/tmp/.payload",
            result=0o755,
        )
    )
    events.append(
        Event(
            ts=t + 0.6,
            event="process_start",
            pid=payload,
            ppid=sh,
            comm="/tmp/.payload",
            exe_path="/tmp/.payload",
        )
    )
    events.append(
        Event(
            ts=t + 0.8,
            event="net_connect",
            pid=payload,
            target_type="net",
            target="203.0.113.7:1337",
        )
    )
    return events, {appserver, sh, payload}


def _attack_persistence(source, t, pids):
    events = []
    sh = source.next()
    bin_pid = source.next()
    targets = [
        f"{HOME}/.bashrc",
        "/etc/crontab",
        f"{HOME}/.config/systemd/user/p.service",
        "/tmp/.bin",
    ]
    events.append(
        Event(ts=t, event="process_start", pid=sh, ppid=pids["systemd"], comm="sh")
    )
    for i, target in enumerate(targets):
        events.append(
            Event(
                ts=t + 0.1 + i * 0.15,
                event="file_write",
                pid=sh,
                target_type="file",
                target=target,
            )
        )
    events.append(
        Event(
            ts=t + 1.0,
            event="process_start",
            pid=bin_pid,
            ppid=sh,
            comm="/tmp/.bin",
            exe_path="/tmp/.bin",
        )
    )
    return events, {sh, bin_pid}


def _attack_exfiltration(source, t, pids, seed):
    events = []
    svc = source.next()
    events.append(
        Event(ts=t, event="process_start", pid=svc, ppid=pids["bash"], comm="svc")
    )
    for i in range(5):
        port = HIGH_PORT_BASE + (seed * 7919 + i * 104729) % HIGH_PORT_SPAN
        host = f"10.9.9.{1 + (seed + i * 7) % 250}"
        events.append(
            Event(
                ts=t + i * 0.3,
                event="net_connect",
                pid=svc,
                target_type="net",
                target=f"{host}:{port}",
            )
        )
    events.append(
        Event(
            ts=t + 1.8,
            event="net_connect",
            pid=svc,
            target_type="net",
            target="198.51.100.66:53",
        )
    )
    return events, {svc}


def _attack_lateral_movement(source, t, pids):
    events = []
    scp = source.next()
    sshd = pids["sshd"]
    events.append(
        Event(ts=t, event="net_accept", pid=sshd, target_type="net", target="10.1.1.7:22")
    )
    events.append(
        Event(
            ts=t + 0.2,
            event="process_start",
            pid=scp,
            ppid=sshd,
            comm="/usr/bin/scp",
            exe_path="/usr/bin/scp",
        )
    )
    events.append(
        Event(
            ts=t + 0.3,
            event="file_write",
            pid=scp,
            target_type="file",
            target=f"{HOME}/.ssh/authorized_keys",
        )
    )
    events.append(
        Event(
            ts=t + 0.4,
            event="file_read",
            pid=scp,
            target_type="file",
            target="/etc/shadow",
        )
    )
    return events, {sshd, scp}


def _attack_supply_chain(source, t, pids):
    events = []
    pkgmgr = source.next()
    pkg = source.next()
    events.append(
        Event(ts=t, event="process_start", pid=pkgmgr, ppid=pids["bash"], comm="pkgmgr")
    )
    events.append(
        Event(
            ts=t + 0.1,
            event="net_connect",
            pid=pkgmgr,
            target_type="net",
            target="registry.evil.example:443",
        )
    )
    events.append(
        Event(
            ts=t + 0.2,
            event="file_write",
            pid=pkgmgr,
            target_type="file",
            target="/tmp/pkg.tar",
        )
    )
    events.append(
        Event(
            ts=t + 0.3,
            event="file_write",
            pid=pkgmgr,
            target_type="file",
            target="/usr/local/bin/pkg",
        )
    )
    events.append(
        Event(
            ts=t + 0.8,
            event="process_start",
            pid=pkg,
            ppid=pkgmgr,
            comm="/usr/local/bin/pkg",
            exe_path="/usr/local/bin/pkg",
        )
    )
    events.append(
        Event(
            ts=t + 1.0,
            event="net_connect",
            pid=pkg,
            target_type="net",
            target="10.5.5.5:9000",
        )
    )
    events.append(
        Event(
            ts=t + 1.2,
            event="file_write",
            pid=pkg,
            target_type="file",
            target=f"{HOME}/.cache/pkg.db",
        )
    )
    return events, {pkgmgr, pkg}


_ATTACK_BUILDERS = {
    "code_injection": _attack_code_injection,
    "persistence": _attack_persistence,
    "exfiltration": _attack_exfiltration,
    "lateral_movement": _attack_lateral_movement,
    "supply_chain": _attack_supply_chain,
}


def build_attack(name: str, seed: int, background: NormalSimulator | None = None) -> ScenarioResult:
    if name not in ATTACKS:
        raise ValueError(f"unknown attack: {name!r}")
    bg = background if background is not None else NormalSimulator(seed)
    events = bg.generate(BG_DURATION)
    source = bg.source
    pids = {e.comm: e.pid for e in events if e.event == "process_start"}
    t = (events[-1].ts if events else BG_DURATION) + 0.5
    attacker = _ATTACK_BUILDERS[name]
    if name == "exfiltration":
        attack_events, malicious = attacker(source, t, pids, seed)
    else:
        attack_events, malicious = attacker(source, t, pids)
    events.extend(attack_events)
    return ScenarioResult(events=events, malicious_pids=malicious, name=name, run_type="attack")


def generate_dataset(
    out_dir: Path, seed: int, normal_runs: int = 10, attack_runs_per_scenario: int = 5
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(normal_runs):
        result = NormalSimulator(seed=seed + i * 7).run(60.0)
        write_run(out_dir, f"normal_{i}", result)
    for j, name in enumerate(ATTACKS):
        for k in range(attack_runs_per_scenario):
            result = build_attack(name, seed=seed + 1000 + j * 31 + k * 13)
            write_run(out_dir, f"attack_{name}_{k}", result)