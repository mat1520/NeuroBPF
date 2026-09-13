import json
from pathlib import Path
from xml.etree import ElementTree

from scripts.figures import generate_figures, graph_svg, results_svg


def _write_report(dirpath):
    summary = {
        "gnn auc_roc": {"mean": 0.80, "std": 0.05},
        "gnn recall_at_1pct": {"mean": 0.90, "std": 0.04},
        "ocsvm auc_roc": {"mean": 0.55, "std": 0.02},
        "ocsvm recall_at_1pct": {"mean": 0.42, "std": 0.03},
    }
    per_seed = [{"seed": 0, "method": "gnn", "auc_roc": 0.8}, {"seed": 1, "method": "gnn", "auc_roc": 0.8}] * 2
    (dirpath / "report.json").write_text(
        json.dumps({"per_seed": per_seed, "summary": summary})
    )


def _write_annotated(dirpath):
    annotations = dirpath / "annotations"
    annotations.mkdir()
    data = {
        "run_id": "attack_x",
        "snapshots": [
            {
                "ts": 0.0,
                "nodes": [
                    {"id": "p:1", "label": "proc", "kind": "process", "score": 0.95, "anomalous": True},
                    {"id": "f:1", "label": "file", "kind": "file", "score": 0.1, "anomalous": False},
                ],
                "edges": [{"src": "p:1", "dst": "f:1", "label": "WRITE"}],
            }
        ],
    }
    (annotations / "attack_x.json").write_text(json.dumps(data))
    return annotations


def test_results_svg_well_formed_and_has_methods(tmp_path):
    _write_report(tmp_path)
    report = json.loads((tmp_path / "report.json").read_text())
    svg = results_svg(report["summary"], seeds=1, epochs=50)
    ElementTree.fromstring(svg)
    assert "<svg" in svg
    assert "AUC ROC" in svg
    assert "gnn" in svg


def test_graph_svg_well_formed(tmp_path):
    annotations = _write_annotated(tmp_path)
    data = json.loads((annotations / "attack_x.json").read_text())
    svg = graph_svg(data["snapshots"][-1], title="test")
    ElementTree.fromstring(svg)
    assert "anómalos" in svg
    assert 'stroke="#f43f5e"' in svg


def test_generate_figures_writes_both_files(tmp_path):
    _write_report(tmp_path)
    annotations = _write_annotated(tmp_path)
    out = generate_figures(tmp_path / "report.json", annotations, tmp_path / "assets")
    assert (out / "results.svg").is_file()
    assert (out / "graph.svg").is_file()
    ElementTree.fromstring((out / "results.svg").read_text())
    ElementTree.fromstring((out / "graph.svg").read_text())


def test_generate_figures_deterministic(tmp_path):
    _write_report(tmp_path)
    annotations = _write_annotated(tmp_path)
    out1 = generate_figures(tmp_path / "report.json", annotations, tmp_path / "a1")
    out2 = generate_figures(tmp_path / "report.json", annotations, tmp_path / "a2")
    assert (out1 / "results.svg").read_text() == (out2 / "results.svg").read_text()
    assert (out1 / "graph.svg").read_text() == (out2 / "graph.svg").read_text()