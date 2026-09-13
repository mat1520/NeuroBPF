import argparse
import json
import pickle
from pathlib import Path

from neurobpf.events import read_events
from neurobpf.gnn.dataset import load_pickled_graphs
from neurobpf.gnn.detect import annotate as annotate_graphs
from neurobpf.gnn.detect import evaluate
from neurobpf.gnn.features import FEAT_DIM
from neurobpf.gnn.train import compute_threshold, load_model, save_model, train_gae
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


def _build_pickle(events_dir, out_path):
    run_paths = sorted(Path(events_dir).rglob("*.events.ndjson"))
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
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fh:
        pickle.dump(data, fh)


def _cmd_build(args):
    _build_pickle(args.events, args.out)


def _normal_graphs(graphs, run_types):
    normal = [g for g, t in zip(graphs, run_types) if t == "normal"]
    return normal if normal else graphs


def _train_command(args):
    graphs, run_types, _ = load_pickled_graphs(args.graphs)
    normal = _normal_graphs(graphs, run_types)
    model = train_gae(
        normal,
        hidden_dim=args.hidden_dim,
        z_dim=args.z_dim,
        epochs=args.epochs,
        seed=args.seed,
    )
    save_model(args.out, model, {"source": str(args.graphs), "seed": args.seed})
    return model, normal


def _cmd_train(args):
    _train_command(args)


def _cmd_detect(args):
    graphs, run_types, _ = load_pickled_graphs(args.graphs)
    model = load_model(args.model, in_dim=FEAT_DIM)
    threshold = compute_threshold(model, _normal_graphs(graphs, run_types))
    result = evaluate(model, graphs, run_types, threshold)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))


def _cmd_annotate(args):
    graphs, run_types, run_ids = load_pickled_graphs(args.graphs)
    model = load_model(args.model, in_dim=FEAT_DIM)
    threshold = compute_threshold(model, _normal_graphs(graphs, run_types))
    entries = annotate_graphs(model, graphs, run_ids, run_types, threshold)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        (out_dir / f"{entry['run_id']}.json").write_text(json.dumps(entry, indent=2))


def _cmd_demo(args):
    out = Path(args.out)
    seed_dir = out / str(args.seed)
    generate_dataset(
        seed_dir,
        seed=args.seed,
        normal_runs=args.normal,
        attack_runs_per_scenario=args.attack,
    )
    pkl = out / "graphs.pkl"
    _build_pickle(seed_dir, pkl)
    graphs, run_types, run_ids = load_pickled_graphs(pkl)
    normal = _normal_graphs(graphs, run_types)
    model = train_gae(
        normal,
        hidden_dim=args.hidden_dim,
        z_dim=args.z_dim,
        epochs=args.epochs,
        seed=args.seed,
    )
    save_model(out / "model.pt", model, {"source": str(pkl), "seed": args.seed})
    threshold = compute_threshold(model, normal)
    attack = [(g, t, r) for g, t, r in zip(graphs, run_types, run_ids) if t == "attack"]
    if not attack:
        attack = [(g, t, r) for g, t, r in zip(graphs, run_types, run_ids)]
    attack_graphs = [a[0] for a in attack]
    attack_types = [a[1] for a in attack]
    result = evaluate(model, attack_graphs, attack_types, threshold)
    (out / "detection.json").write_text(json.dumps(result, indent=2))
    entries = annotate_graphs(model, graphs, run_ids, run_types, threshold)
    ann_dir = out / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        (ann_dir / f"{entry['run_id']}.json").write_text(json.dumps(entry, indent=2))
    print(f"AUC {result['auc_roc']:.2f} | recall@k {result['recall_at_topk']:.2f}")


def _add_train_args(parser):
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--z-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)


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
    trn = sub.add_parser("train", help="train the graph autoencoder on normal runs")
    trn.add_argument("--graphs", required=True, help="pickled graph dataset")
    trn.add_argument("--out", required=True, help="output model path")
    _add_train_args(trn)
    trn.set_defaults(func=_cmd_train)
    det = sub.add_parser("detect", help="score graphs and report detection metrics")
    det.add_argument("--graphs", required=True, help="pickled graph dataset")
    det.add_argument("--model", required=True, help="trained model path")
    det.add_argument("--out", required=True, help="output detection JSON path")
    det.set_defaults(func=_cmd_detect)
    ann = sub.add_parser("annotate", help="write per-run annotated replay JSON")
    ann.add_argument("--graphs", required=True, help="pickled graph dataset")
    ann.add_argument("--model", required=True, help="trained model path")
    ann.add_argument("--out", required=True, help="output directory")
    ann.set_defaults(func=_cmd_annotate)
    dem = sub.add_parser("demo", help="end-to-end generate/train/detect/annotate demo")
    dem.add_argument("--out", required=True, help="output directory")
    dem.add_argument("--normal", type=int, default=6)
    dem.add_argument("--attack", type=int, default=3)
    _add_train_args(dem)
    dem.set_defaults(func=_cmd_demo)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()