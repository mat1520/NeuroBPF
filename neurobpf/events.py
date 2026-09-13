import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

VALID_TARGETS = {None, "file", "process", "net"}


@dataclass
class Event:
    """A single system-call level event captured from the provenance probe."""

    ts: float
    event: str
    pid: int
    ppid: int = -1
    comm: str = ""
    exe_path: str = ""
    uid: int = 1000
    target_type: str | None = None
    target: str | int | None = None
    result: int = 0

    def __post_init__(self):
        if self.target_type not in VALID_TARGETS:
            raise ValueError(f"invalid target_type: {self.target_type!r}")


def event_to_json(e: Event) -> str:
    return json.dumps(asdict(e), sort_keys=True, separators=(",", ":")) + "\n"


def event_from_json(line: str) -> Event:
    try:
        data = json.loads(line)
        return Event(**data)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"malformed event line: {line!r}") from exc


def write_events(path: Path, events: Iterable[Event]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for e in events:
            fh.write(event_to_json(e))


def read_events(path: Path) -> Iterator[Event]:
    with Path(path).open() as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            yield event_from_json(line)