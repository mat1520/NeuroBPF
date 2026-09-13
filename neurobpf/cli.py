import argparse
import json
import pickle
from pathlib import Path

from neurobpf.events import read_events
from neurobpf.gnn.dataset import load_pickled_graphs, split_by_run
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


def _build_pickle(events_dir, out_path, window=5.0):
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
        run_id = path.name.replace(".events.ndjson", "")
        for snap in build_snapshots(events, window=window, malicious_pids=malicious):
            snapshots.append(snap)
            run_types.append(run_type)
            run_ids.append(run_id)
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


def _val_normal(graphs, run_types, seed):
    normal = _normal_graphs(graphs, run_types)
    if not normal:
        return []
    train, val, _ = split_by_run(normal, seed=seed)
    return val if val else train


def _train_command(args):
    graphs, run_types, _ = load_pickled_graphs(args.graphs)
    normal = _normal_graphs(graphs, run_types)
    train, val, _ = split_by_run(normal, seed=args.seed)
    model = train_gae(
        train,
        val_graphs=val,
        hidden_dim=args.hidden_dim,
        z_dim=args.z_dim,
        epochs=args.epochs,
        seed=args.seed,
    )
    save_model(args.out, model, {"source": str(args.graphs), "seed": args.seed})
    return model, val


def _cmd_train(args):
    _train_command(args)


def _cmd_detect(args):
    graphs, run_types, _ = load_pickled_graphs(args.graphs)
    model = load_model(args.model, in_dim=FEAT_DIM)
    threshold = compute_threshold(model, _val_normal(graphs, run_types, 0))
    result = evaluate(model, graphs, run_types, threshold)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))


def _cmd_annotate(args):
    graphs, run_types, run_ids = load_pickled_graphs(args.graphs)
    model = load_model(args.model, in_dim=FEAT_DIM)
    threshold = compute_threshold(model, _val_normal(graphs, run_types, 0))
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
    train, val, _ = split_by_run(normal, seed=args.seed)
    model = train_gae(
        train,
        val_graphs=val,
        hidden_dim=args.hidden_dim,
        z_dim=args.z_dim,
        epochs=args.epochs,
        seed=args.seed,
    )
    save_model(out / "model.pt", model, {"source": str(pkl), "seed": args.seed})
    threshold = compute_threshold(model, val or train)
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
    if args.serve:
        _cmd_serve(argparse.Namespace(
            annotated=str(ann_dir), web=str(args.web or "web/dist"),
            port=args.port, delay=args.delay,
        ))


def _cmd_serve(args):
    import uvicorn

    from neurobpf.server import create_app

    annotated_dir = Path(args.annotated)
    web_dir = Path(args.web) if args.web else None
    app = create_app(annotated_dir, loop_delay=args.delay, web_dir=web_dir)
    uvicorn.run(app, host="127.0.0.1", port=args.port)


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
    dem = sub.add_parser("demo", help="end-to-end generate/train/detect/annotate/serve demo")
    dem.add_argument("--out", required=True, help="output directory")
    dem.add_argument("--normal", type=int, default=6)
    dem.add_argument("--attack", type=int, default=3)
    dem.add_argument("--serve", action="store_true", help="serve the annotated replay over web")
    dem.add_argument("--web", default="web/dist", help="path to built frontend (default web/dist)")
    dem.add_argument("--port", type=int, default=8899)
    dem.add_argument("--delay", type=float, default=0.7)
    _add_train_args(dem)
    dem.set_defaults(func=_cmd_demo)
    srv = sub.add_parser("serve", help="serve annotated replay + frontend on a local port")
    srv.add_argument("--annotated", required=True, help="directory of annotated run JSON files")
    srv.add_argument("--web", default=None, help="path to built frontend (optional)")
    srv.add_argument("--port", type=int, default=8899)
    srv.add_argument("--delay", type=float, default=0.7)
    srv.set_defaults(func=_cmd_serve)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()