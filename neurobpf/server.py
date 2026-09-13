import asyncio
import json
import os
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles

DEFAULT_ANNOTATED_DIR = "annotated"
DEFAULT_WEB_DIR = "web/dist"


def _load_runs(annotated_dir: Path):
    runs = []
    for path in sorted(annotated_dir.glob("*.json"), key=lambda p: p.name):
        data = json.loads(path.read_text())
        runs.append((data["run_id"], data["run_type"], data["snapshots"]))
    return runs


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


def create_app(annotated_dir: Path | str, loop_delay: float = 0.7, web_dir: Path | None = None) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"ok": True}

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