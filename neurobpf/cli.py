import argparse
import json
import pickle
from pathlib import Path

from neurobpf.events import read_events
from neurobpf.graphbuilder import build_snapshots
from neurobpf.scenarios import generate_dataset


def _cmd_generate(args):
    out_dir = Path(args.out) / str(args.seed)
    generate_dataset(
        out_dir,
        seed=args.seed,
        normal_runs=args.normal,
        attack_runs_per_scenario=args.attack,
    )


def _cmd_build(args):
    run_paths = sorted(Path(args.events).rglob("*.events.ndjson"))
    snapshots = []
    run_types = []
    run_ids = []
    for path in run_paths:
        events = list(read_events(path))
        labels_path = path.with_name(path.name.replace(".events.ndjson", ".labels.json"))
        if labels_path.exists():
            labels = json.loads(labels_path.read_text())
            run_type = labels["run_type"]
            malicious = set(labels["malicious_pids"])
        else:
            run_type = "real"
            malicious = set()
        snapshots.append(build_snapshots(events, malicious_pids=malicious)[-1])
        run_types.append(run_type)
        run_ids.append(path.name.replace(".events.ndjson", ""))
    data = {"snapshots": snapshots, "run_types": run_types, "run_ids": run_ids}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fh:
        pickle.dump(data, fh)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="neurobpf")
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="generate a synthetic dataset of runs")
    gen.add_argument("--out", required=True, help="output directory")
    gen.add_argument("--seed", type=int, default=0, help="random seed")
    gen.add_argument("--normal", type=int, default=10, help="number of normal runs")
    gen.add_argument("--attack", type=int, default=5, help="attack runs per scenario")
    gen.set_defaults(func=_cmd_generate)
    bld = sub.add_parser("build", help="build provenance graph snapshots from runs")
    bld.add_argument("--events", required=True, help="directory containing run events")
    bld.add_argument("--out", required=True, help="output pickle path")
    bld.set_defaults(func=_cmd_build)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()