import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from neurobpf.server import create_app


def _write_run(directory, run_id, run_type, snapshots):
    (directory / f"{run_id}.json").write_text(
        json.dumps({"run_id": run_id, "run_type": run_type, "snapshots": snapshots})
    )


def _sample_snapshot(index):
    return {
        "ts": float(index + 1),
        "nodes": [
            {"id": f"p:{index}", "label": "bash", "kind": "process", "score": 0.5, "anomalous": index == 0}
        ],
        "edges": [{"src": f"p:{index}", "dst": f"f:/x{index}", "label": "WRITE"}],
    }


def _receive_frames(websocket, count):
    return [json.loads(websocket.receive_text()) for _ in range(count)]


@pytest.fixture
def annotated_dir(tmp_path):
    directory = tmp_path / "annotated"
    directory.mkdir()
    _write_run(directory, "run_a", "attack", [_sample_snapshot(0), _sample_snapshot(1)])
    _write_run(directory, "run_b", "normal", [_sample_snapshot(0), _sample_snapshot(1), _sample_snapshot(2)])
    return directory


def test_health_returns_ok(annotated_dir):
    with TestClient(create_app(annotated_dir, loop_delay=0.0)) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_root_returns_json_info_without_web_dir(annotated_dir):
    with TestClient(create_app(annotated_dir, loop_delay=0.0)) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_static_mount_serves_index(tmp_path, annotated_dir):
    web_dir = tmp_path / "web"
    web_dir.mkdir()
    (web_dir / "index.html").write_text("<html>hello</html>", encoding="utf-8")
    with TestClient(create_app(annotated_dir, loop_delay=0.0, web_dir=web_dir)) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "hello" in response.text


def test_ws_replay_protocol_and_run_ordering(annotated_dir):
    with TestClient(create_app(annotated_dir, loop_delay=0.0)) as client:
        with client.websocket_connect("/ws/snapshots") as websocket:
            first = json.loads(websocket.receive_text())
            assert set(first) == {"run_id", "run_type", "snapshot_index", "snap_total", "nodes", "edges"}
            assert first["run_id"] == "run_a"
            assert first["run_type"] == "attack"
            assert first["snapshot_index"] == 0
            assert first["snap_total"] == 2
            assert first["nodes"] == _sample_snapshot(0)["nodes"]
            assert first["edges"] == _sample_snapshot(0)["edges"]

            rest = _receive_frames(websocket, 6)
            expected = [
                ("run_a", 1, 2, "attack"),
                ("done", None, None, None),
                ("run_b", 0, 3, "normal"),
                ("run_b", 1, 3, "normal"),
                ("run_b", 2, 3, "normal"),
                ("done", None, None, None),
            ]
            for frame, (run_id, index, total, run_type) in zip(rest, expected):
                if run_id == "done":
                    assert frame == {"done": True}
                else:
                    assert frame["run_id"] == run_id
                    assert frame["run_type"] == run_type
                    assert frame["snapshot_index"] == index
                    assert frame["snap_total"] == total
                    assert frame["nodes"] == _sample_snapshot(index)["nodes"]
                    assert frame["edges"] == _sample_snapshot(index)["edges"]


def test_ws_loops_back_to_first_run(annotated_dir):
    with TestClient(create_app(annotated_dir, loop_delay=0.0)) as client:
        with client.websocket_connect("/ws/snapshots") as websocket:
            first_cycle = _receive_frames(websocket, 7)
            second_cycle = _receive_frames(websocket, 7)
    assert second_cycle == first_cycle
    assert first_cycle[2] == {"done": True}


def test_ws_reports_error_when_no_runs(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with TestClient(create_app(empty, loop_delay=0.0)) as client:
        with client.websocket_connect("/ws/snapshots") as websocket:
            frame = json.loads(websocket.receive_text())
    assert frame == {"error": "no annotated runs"}


def _write_detection(directory):
    p = directory / "detection.json"
    p.write_text(
        json.dumps(
            {
                "auc_roc": 0.74,
                "recall_at_topk": 0.5,
                "per_run_recall": {"run_a": 1.0},
                "threshold_report": {"tp": 2, "fp": 1, "fn": 1, "tn": 99, "tpr": 0.5, "fpr": 0.01},
            }
        )
    )
    return p


def _write_report(directory):
    p = directory / "report.json"
    p.write_text(
        json.dumps(
            {
                "per_seed": [],
                "summary": {
                    "gnn auc_roc": {"mean": 0.79, "std": 0.05},
                    "ocsvm auc_roc": {"mean": 0.55, "std": 0.02},
                },
            }
        )
    )
    return p


def test_api_experiment_returns_detection(tmp_path, annotated_dir):
    det = _write_detection(tmp_path)
    with TestClient(create_app(annotated_dir, loop_delay=0.0, detection_path=det)) as client:
        res = client.get("/api/experiment")
    body = res.json()
    assert body["auc_roc"] == 0.74
    assert body["recall_at_topk"] == 0.5
    assert body["threshold_report"]["tpr"] == 0.5


def test_api_experiment_404_when_missing(annotated_dir):
    with TestClient(create_app(annotated_dir, loop_delay=0.0)) as client:
        res = client.get("/api/experiment")
    assert res.status_code == 404


def test_api_runs_lists_runs(annotated_dir):
    with TestClient(create_app(annotated_dir, loop_delay=0.0)) as client:
        res = client.get("/api/runs")
    runs = res.json()["runs"]
    assert [r["id"] for r in runs] == ["run_a", "run_b"]
    assert [r["run_type"] for r in runs] == ["attack", "normal"]
    assert [r["n_snapshots"] for r in runs] == [2, 3]


def test_api_run_detail_returns_snapshots_and_stats(annotated_dir):
    with TestClient(create_app(annotated_dir, loop_delay=0.0)) as client:
        res = client.get("/api/runs/run_a")
    body = res.json()
    assert body["run_id"] == "run_a"
    assert len(body["snapshots"]) == 2
    assert len(body["stats"]) == 2
    stat = body["stats"][0]
    assert stat["n_nodes"] == 1
    assert stat["n_nodes"] == len(body["snapshots"][0]["nodes"])
    assert stat["n_anomalous"] == 1
    assert stat["max_score"] == 0.5


def test_api_run_missing_returns_404(annotated_dir):
    with TestClient(create_app(annotated_dir, loop_delay=0.0)) as client:
        res = client.get("/api/runs/nope")
    assert res.status_code == 404


def test_api_report_returns_series_when_present(tmp_path, annotated_dir):
    rep = _write_report(tmp_path)
    with TestClient(create_app(annotated_dir, loop_delay=0.0, report_path=rep)) as client:
        res = client.get("/api/report")
    body = res.json()
    assert body["summary"]["gnn auc_roc"]["mean"] == 0.79
    assert body["summary"]["ocsvm auc_roc"]["std"] == 0.02


def test_api_report_404_when_missing(annotated_dir):
    with TestClient(create_app(annotated_dir, loop_delay=0.0)) as client:
        res = client.get("/api/report")
    assert res.status_code == 404


def test_detection_discovered_in_parent_dir(tmp_path, annotated_dir):
    _write_detection(tmp_path)
    annotated = tmp_path / "annotations"
    annotated.mkdir()
    with TestClient(create_app(annotated, loop_delay=0.0)) as client:
        res = client.get("/api/experiment")
    assert res.status_code == 200
    assert res.json()["auc_roc"] == 0.74


def test_module_entry_serves_health_on_configured_port(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = dict(os.environ, NEUROBPF_PORT=str(port))
    repo_root = Path(__file__).resolve().parents[1]
    proc = None
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "neurobpf.server"],
            cwd=repo_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = time.time() + 30
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            try:
                response = httpx.get(f"http://127.0.0.1:{port}/health", timeout=0.5)
            except httpx.HTTPError:
                time.sleep(0.1)
                continue
            assert response.status_code == 200
            assert response.json() == {"ok": True}
            return
        stderr = proc.stderr.read() if proc.stderr else ""
        pytest.fail(f"server did not become healthy: {stderr[:2000]!r}")
    finally:
        if proc is not None:
            proc.terminate()
            proc.wait(timeout=10)