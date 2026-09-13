import json
import pickle
import re
from pathlib import Path

from neurobpf.cli import main


def _make_graphs(tmp_path):
    out = tmp_path / "data"
    main(["generate", "--out", str(out), "--seed", "3", "--normal", "2", "--attack", "1"])
    pkl = tmp_path / "graph.pkl"
    main(["build", "--events", str(out / "3"), "--out", str(pkl)])
    return pkl


def test_train_subcommand_writes_model(tmp_path):
    pkl = _make_graphs(tmp_path)
    model_path = tmp_path / "model.pt"
    main(
        [
            "train",
            "--graphs",
            str(pkl),
            "--out",
            str(model_path),
            "--hidden-dim",
            "16",
            "--z-dim",
            "16",
            "--epochs",
            "10",
            "--seed",
            "0",
        ]
    )
    assert model_path.exists()
    import torch

    data = torch.load(model_path, weights_only=False)
    assert data["meta"]["hidden_dim"] == 16
    assert data["meta"]["z_dim"] == 16


def test_detect_subcommand_writes_json(tmp_path):
    pkl = _make_graphs(tmp_path)
    model_path = tmp_path / "model.pt"
    main(["train", "--graphs", str(pkl), "--out", str(model_path), "--epochs", "8"])
    out_json = tmp_path / "detection.json"
    main(["detect", "--graphs", str(pkl), "--model", str(model_path), "--out", str(out_json)])
    assert out_json.exists()
    data = json.loads(out_json.read_text())
    assert {"auc_roc", "recall_at_topk", "per_run_recall"} <= set(data)
    assert 0.0 <= data["auc_roc"] <= 1.0


def test_annotate_subcommand_writes_per_run(tmp_path):
    pkl = _make_graphs(tmp_path)
    model_path = tmp_path / "model.pt"
    main(["train", "--graphs", str(pkl), "--out", str(model_path), "--epochs", "8"])
    out_dir = tmp_path / "ann"
    main(["annotate", "--graphs", str(pkl), "--model", str(model_path), "--out", str(out_dir)])
    files = sorted(p.name for p in out_dir.glob("*.json"))
    with pkl.open("rb") as fh:
        data = pickle.load(fh)
    assert len(files) == len(data["run_ids"])
    for rid in data["run_ids"]:
        entry = json.loads((out_dir / f"{rid}.json").read_text())
        assert set(entry) == {"run_id", "run_type", "snapshots"}
        assert entry["run_id"] == rid


def test_demo_prints_auc_line_and_writes_files(tmp_path, capsys):
    out = tmp_path / "demo"
    main(
        [
            "demo",
            "--out",
            str(out),
            "--seed",
            "3",
            "--normal",
            "2",
            "--attack",
            "1",
            "--epochs",
            "12",
        ]
    )
    captured = capsys.readouterr().out
    match = re.search(r"AUC \d+\.\d+ \| recall@k \d+\.\d+", captured)
    assert match, captured
    assert (out / "model.pt").exists()
    assert (out / "detection.json").exists()
    assert (out / "annotations").is_dir()