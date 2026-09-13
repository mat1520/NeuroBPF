import asyncio
import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.staticfiles import StaticFiles

DEFAULT_ANNOTATED_DIR = "annotated"
DEFAULT_WEB_DIR = "web/dist"


def _load_runs(annotated_dir: Path):
    runs = []
    for path in sorted(annotated_dir.glob("*.json"), key=lambda p: p.name):
        data = json.loads(path.read_text())
        runs.append((data["run_id"], data["run_type"], data["snapshots"]))
    return runs


def _discover(base_dir: Path, filename: str):
    direct = base_dir / filename
    if direct.is_file():
        return direct
    parent = base_dir.parent / filename
    if parent.is_file():
        return parent
    return None


def _json_or_404(path):
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail=f"no such file: {path}")
    return json.loads(path.read_text())


def _snapshot_stats(snap):
    nodes = snap.get("nodes", [])
    scores = [n.get("score", 0.0) for n in nodes
              if isinstance(n.get("score"), (int, float))]
    return {
        "ts": snap.get("ts", 0.0),
        "n_nodes": len(nodes),
        "n_edges": len(snap.get("edges", [])),
        "n_anomalous": sum(1 for n in nodes if n.get("anomalous")),
        "max_score": max(scores) if scores else 0.0,
    }


async def _replay(websocket: WebSocket, runs: list, loop_delay: float) -> None:
    index = 0
    while True:
        run_id, run_type, snapshots = runs[index]
        size = len(snapshots)
        for i, snap in enumerate(snapshots):
            await websocket.send_json(
                {
                    "run_id": run_id,
                    "run_type": run_type,
                    "snapshot_index": i,
                    "snap_total": size,
                    "nodes": snap["nodes"],
                    "edges": snap["edges"],
                }
            )
            if loop_delay > 0:
                await asyncio.sleep(loop_delay)
        await websocket.send_json({"done": True})
        index = (index + 1) % len(runs)


def _resolve_file_arg(path_or_file: Path | str | None, filename: str, fallback_dir: Path):
    if path_or_file is None:
        return _discover(fallback_dir, filename)
    p = Path(path_or_file)
    if p.is_file():
        return p
    return _discover(p, filename)


def create_app(
    annotated_dir: Path | str,
    loop_delay: float = 0.7,
    web_dir: Path | None = None,
    detection_path: Path | str | None = None,
    report_path: Path | str | None = None,
) -> FastAPI:
    app = FastAPI()
    base = Path(annotated_dir)
    detection_file = _resolve_file_arg(detection_path, "detection.json", base)
    report_file = _resolve_file_arg(report_path, "report.json", base)

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.get("/api/experiment")
    async def experiment():
        return _json_or_404(detection_file)

    @app.get("/api/runs")
    async def runs():
        rows = []
        for rid, rtype, snapshots in _load_runs(base):
            last = snapshots[-1] if snapshots else {}
            rows.append(
                {
                    "id": rid,
                    "run_type": rtype,
                    "n_snapshots": len(snapshots),
                    "n_nodes_final": len(last.get("nodes", [])),
                    "anomalous_final": sum(1 for n in last.get("nodes", []) if n.get("anomalous")),
                }
            )
        return {"runs": rows}

    @app.get("/api/runs/{run_id}")
    async def run_detail(run_id: str):
        runs_by_id = {rid: (rtype, snaps) for rid, rtype, snaps in _load_runs(base)}
        if run_id not in runs_by_id:
            raise HTTPException(status_code=404, detail=f"no such run: {run_id}")
        rtype, snapshots = runs_by_id[run_id]
        return {
            "run_id": run_id,
            "run_type": rtype,
            "snapshots": snapshots,
            "stats": [_snapshot_stats(s) for s in snapshots],
        }

    @app.get("/api/report")
    async def report():
        return _json_or_404(report_file)

    @app.websocket("/ws/snapshots")
    async def snapshots(websocket: WebSocket):
        await websocket.accept()
        runs = _load_runs(Path(annotated_dir))
        if not runs:
            await websocket.send_json({"error": "no annotated runs"})
            await websocket.close()
            return
        await _replay(websocket, runs, loop_delay)

    if web_dir is not None and (Path(web_dir) / "index.html").is_file():
        app.mount("/", StaticFiles(directory=Path(web_dir), html=True), name="web")
    else:

        @app.get("/")
        async def root():
            return {"message": "NeuroBPF replay server running"}

    return app


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("NEUROBPF_PORT", "8899"))
    app = create_app(Path(DEFAULT_ANNOTATED_DIR), web_dir=Path(DEFAULT_WEB_DIR))
    uvicorn.run(app, host="127.0.0.1", port=port)