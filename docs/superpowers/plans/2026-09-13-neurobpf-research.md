# NeuroBPF Research Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert NeuroBPF from a demo into a research prototype with rigorous evaluation: temporal windows (not last-snapshot collapse), a real train/val/test split, sparse adjacency for scale, a paper-grade metric suite, feature-only baselines, ablations, and a THEIA_E3 benchmark adapter.

**Architecture:** Keep the existing NDJSON→graphbuilder→GAE pipeline. Change (a) the graph builder path to keep ALL cumulative windows per run tagged with `run_id`; (b) training to validate on a held-out val split with early stopping; (c) detection metrics to run-level nDCG / recall@k / precision@k / FPR@budget using the calibration threshold; (d) adjacency to PyTorch sparse ops for >10x scale; (e) add OCSVM + MLP-AE feature-only baselines and a compare command that runs multi-seed experiments and prints a paper table.

**Tech Stack:** Python 3.14 + PyTorch 2.14 (CPU) + numpy + pytest; scikit-learn added for OCSVM baseline. No new GPU. No scipy (PyTorch sparse covers it).

**Spec:** `docs/superpowers/specs/2026-09-13-neurobpf-research-design.md`

## Global Constraints

- **TDD mandatory**: production code only after a failing test. Every task is RED → GREEN.
- Source code: NO comments; docstrings only where they clarify a public interface.
- Identifiers in English; docs in Spanish.
- All randomness seeded via the existing `seed` parameter threaded through every generator.
- GPU unavailable → everything runs on CPU.
- Run tests with the repo venv: `./.venv/bin/pytest`.
- Existing public API kept backward-compatible unless a task explicitly says it breaks it (and updates its tests in the same task).
- Every task commits only its own files. No pushing (controller pushes at the end).

---

### Task 1: Fix `is_root` feature corruption for file/net nodes

**Files:**
- Modify: `neurobpf/gnn/features.py:54-60` (the `x[i, 3]` assignment in `snapshot_to_graph`)
- Test: `tests/features_test.py` (new file referenced by `tests/test_features.py`) — see below; use existing `tests/test_features.py`

**Context:** `graphbuilder.add_node` sets `uid=0` for every non-process node, so `x[i,3] = 1.0 if node.uid == 0` marks *all* file and net nodes as root. `is_root` must only be true for process nodes whose uid is 0.

**Interfaces:**
- Produces: no signature change. `snapshot_to_graph` now emits `x[i,3] == 0.0` for non-process nodes.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_features.py`:

```python
def test_is_root_only_for_root_process_nodes():
    nodes = {
        "p:1": NodeData(node_id="p:1", kind="process", label="bash", uid=1000, pid=1),
        "p:0": NodeData(node_id="p:0", kind="process", label="init", uid=0, pid=0),
        "f:1": NodeData(node_id="f:1", kind="file", label="/x", uid=0),
    }
    snap = Snapshot(0.0, nodes, [], set())
    g = snapshot_to_graph(snap, 0)
    idx = {nid: i for i, nid in enumerate(g.node_ids)}
    assert g.x[idx["p:1"], 3].item() == 0.0
    assert g.x[idx["p:0"], 3].item() == 1.0
    assert g.x[idx["f:1"], 3].item() == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/pytest tests/test_features.py::test_is_root_only_for_root_process_nodes -v`
Expected: FAIL — file node `f:1` gets `is_root == 1.0`.

- [ ] **Step 3: Implement fix**

In `neurobpf/gnn/features.py`, change the is_root line inside the node loop:

```python
        node = snap.nodes[nid]
        x[i, :3] = torch.tensor(KIND_ONE_HOT.get(node.kind, (0.0, 0.0, 0.0)))
        x[i, 3] = 1.0 if node.kind == "process" and node.uid == 0 else 0.0
```

- [ ] **Step 4: Run full test file to verify pass**

Run: `./.venv/bin/pytest tests/test_features.py` and `./.venv/bin/pytest tests/`
Expected: all PASS (no existing test asserted the buggy value).

- [ ] **Step 5: Commit**

```bash
git add neurobpf/gnn/features.py tests/test_features.py
git commit -m "fix: is_root solo para procesos root en features"
```

---

### Task 2: PyTorch sparse adjacency + sparse-aware GCN

**Files:**
- Modify: `neurobpf/gnn/dataset.py` — add `make_sparse_adj`
- Modify: `neurobpf/gnn/model.py` — sparse-aware `_sym_normalize` and `GCNConv.forward`
- Test: `tests/test_dataset.py`, `tests/test_model.py`

**Context:** `make_adj` allocates a dense n×n tensor. `GCNConv.forward` does `adj @ self.lin(x)`. Both must support sparse COO so THEIA subgraphs (~1–5k nodes) fit in memory and propagate in O(nnz·d).

**Interfaces:**
- Produces: `make_sparse_adj(g: GraphTensor, self_loops: bool = True) -> torch.Tensor` (sparse COO, coalesced, symmetric, binary, dtype float32).
- Produces: `GCNConv.forward(x, adj)` accepts dense or sparse; `_sym_normalize(adj)` accepts dense or sparse; `GAE.train_loss`, `.node_scores`, `.encode` unchanged signatures but work with sparse adj.
- Consumes: `GraphTensor` from Task 1.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dataset.py`:

```python
def test_make_sparse_adj_is_coo_symmetric_binary():
    g = snapshot_to_graph(_snapshot())
    adj = make_sparse_adj(g)
    assert adj.is_sparse
    dense = adj.to_dense()
    assert dense.shape == (2, 2)
    assert dense[0, 1].item() == 1.0
    assert dense[1, 0].item() == 1.0
    assert torch.allclose(dense, dense.T)
    assert set(dense.unique().tolist()) <= {0.0, 1.0}


def test_make_sparse_adj_self_loops_flag():
    g = snapshot_to_graph(_snapshot())
    no = make_sparse_adj(g, self_loops=False).to_dense()
    assert no[0, 0].item() == 0.0
    yes = make_sparse_adj(g, self_loops=True).to_dense()
    assert yes[0, 0].item() == 1.0
    assert yes[1, 1].item() == 1.0
```

Append to `tests/test_model.py`:

```python
def test_gcn_forward_sparse_matches_dense():
    from neurobpf.gnn.dataset import make_adj, make_sparse_adj
    g = snapshot_to_graph(_mini_graph())
    model = GAE(in_dim=FEAT_DIM, hidden_dim=8, z_dim=8)
    model.eval()
    with torch.no_grad():
        z_dense = model.encode(g.x, make_adj(g))
        z_sparse = model.encode(g.x, make_sparse_adj(g))
    assert torch.allclose(z_dense, z_sparse, atol=1e-5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/bin/pytest tests/test_dataset.py::test_make_sparse_adj_is_coo_symmetric_binary tests/test_model.py::test_gcn_forward_sparse_matches_dense -v`
Expected: FAIL — `make_sparse_adj` undefined, and `encode` ValueError on sparse adj.

- [ ] **Step 3: Implement `make_sparse_adj`**

In `neurobpf/gnn/dataset.py`:

```python
def make_sparse_adj(g: GraphTensor, self_loops: bool = True) -> torch.Tensor:
    n = g.x.shape[0]
    src = g.edge_index[0]
    dst = g.edge_index[1]
    both = [torch.stack([src, dst]), torch.stack([dst, src])]
    if self_loops:
        both.append(torch.stack([torch.arange(n), torch.arange(n)]))
    idx = torch.unique(torch.cat(both, dim=1), dim=1)
    vals = torch.ones(idx.shape[1], dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, vals, (n, n)).coalesce()
```

- [ ] **Step 4: Implement sparse-aware model**

In `neurobpf/gnn/model.py`, replace `_sym_normalize` and `GCNConv.forward`:

```python
def _sym_normalize(adj: torch.Tensor) -> torch.Tensor:
    if adj.is_sparse:
        n = adj.shape[0]
        deg = torch.sparse.sum(adj, dim=1).to_dense()
        inv = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
        return torch.sparse_coo_tensor(
            adj.indices()[:, ::-1],
            adj.values() * inv[adj.indices()[0]] * inv[adj.indices()[1]],
            adj.shape,
        ).coalesce()
    a = adj + torch.eye(adj.shape[0], device=adj.device, dtype=adj.dtype)
    deg = a.sum(dim=1)
    inv = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
    return inv[:, None] * a * inv[None, :]


class GCNConv(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x, adj):
        return adj @ self.lin(x)
```

`adj @ self.lin(x)` already dispatches sparse@dense in PyTorch, so `GCNConv` needs no conditional. Check `node_scores` still works: it calls `decode(self.encode(...))` and builds dense `z @ z.T`; that stays dense (bounded by per-graph n, fine for subgraphs ≤2k; the encode path is the scalable part).

- [ ] **Step 5: Run tests to verify pass**

Run: `./.venv/bin/pytest tests/test_dataset.py tests/test_model.py`
Expected: all PASS, including existing dense-path tests.

- [ ] **Step 6: Commit**

```bash
git add neurobpf/gnn/dataset.py neurobpf/gnn/model.py tests/test_dataset.py tests/test_model.py
git commit -m "feat: adjacencia sparse COO y GCN sparse-aware"
```

---

### Task 3: Temporal windowing — keep ALL windows, tag graphs by run

**Files:**
- Modify: `neurobpf/cli.py` (`_build_pickle`), `neurobpf/gnn/features.py` (`GraphTensor` gains `run_id`)
- Test: `tests/test_cli.py`, `tests/features_test.py` if needed

**Context:** `_build_pickle` currently does `build_snapshots(...)[-1]` per run — the entire temporal structure is discarded. Change it to keep every window snapshot, each tagged with its `run_id` so downstream code can group graphs by run.

**Interfaces:**
- Produces: `GraphTensor.run_id: str = ""` field (added to dataclass, default empty). `snapshot_to_graph(snap, snapshot_id, run_id="")`.
- Produces: `_build_pickle` pickle `data["snapshots"]` now holds one Snapshot per window across all runs (multiple per run); `data["run_types"]`/`data["run_ids"]` each have one entry per *window* (aligned, repeated for a run's windows); `data["run_slots"]` = number of windows per run's provenance is derivable from run_ids grouping.
- Consumes: `build_snapshots` (unchanged).
- Breaks: `load_pickled_graphs` callers that assumed one snapshot per run — Task 4 fixes them. Within this task, update existing tests that assert single-snapshot behavior only if they compute on a discard path (see steps).

- [ ] **Step 1: Read current `_build_pickle`**

Run: `./.venv/bin/grep -n "_build_pickle" neurobpf/cli.py` then read around it. Confirm it calls `build_snapshots(events, window)[-1]`.

- [ ] **Step 2: Write the failing test**

Append to `tests/test_cli.py`:

```python
def test_build_keeps_all_temporal_windows(tmp_path, monkeypatch):
    import neurobpf.cli as cli
    evs_dir = tmp_path / "events"
    evs_dir.mkdir()
    events = []
    for i in range(3):
        e = Event(
            ts=0.0 + i,
            event="file_read",
            pid=1,
            comm="bash",
            uid=1000,
            ppid=0,
            target=f"/f{i}",
            exe_path="",
        )
        events.append(e)
    (evs_dir / "normal_0.events.ndjson").write_text(
        "\n".join(json.dumps(event_to_json(e)) for e in events)
    )
    (evs_dir / "normal_0.labels.json").write_text(json.dumps({"run_type": "normal"}))
    (evs_dir / "normal_1.events.ndjson").write_text(
        "\n".join(json.dumps(event_to_json(e)) for e in events)
    )
    (evs_dir / "normal_1.labels.json").write_text(json.dumps({"run_type": "normal"}))
    out = tmp_path / "g.pkl"
    cli._build_pickle(evs_dir, out, window=2.0)
    data = pickle.load(open(out, "rb"))
    assert len(data["run_ids"]) == len(data["snapshots"])
    assert data["run_ids"][0] == "normal_0"
    assert data["snapshots"][0].window_start == 0.0
```

Add imports if missing: `pickle`, `json`, `from neurobpf.events import Event, event_to_json`.

- [ ] **Step 3: Run test to verify it fails**

Run: `./.venv/bin/pytest tests/test_cli.py::test_build_keeps_all_temporal_windows -v`
Expected: FAIL — only 1 snapshot currently in pickle (last window), or run_ids/snapshots misaligned.

- [ ] **Step 4: Add `run_id` to `GraphTensor`**

In `neurobpf/gnn/features.py`:

```python
@dataclass
class GraphTensor:
    ...
    ts: float = 0.0
    run_id: str = ""
```

Update `snapshot_to_graph` signature:

```python
def snapshot_to_graph(snap: Snapshot, snapshot_id: int = 0, run_id: str = "") -> GraphTensor:
    ...
    return GraphTensor(..., run_id=run_id)
```

- [ ] **Step 5: Implement `_build_pickle` window retention**

In `neurobpf/cli.py`, replace the snapshot-collapse loop so per run it builds all windows and records run/type per window:

```python
def _build_pickle(events_dir, out_path, window=5.0):
    runs = []
    for path in sorted(Path(events_dir).rglob("*.events.ndjson")):
        labels = _load_labels(path)
        events = list(read_events(path))
        snapshots = build_snapshots(events, window=window, malicious_pids=labels["malicious_pids"])
        run_id = path.stem.replace(".events", "")
        for snap in snapshots:
            runs.append((run_id, labels["run_type"], snap))
    data = {
        "snapshots": [s for _, _, s in runs],
        "run_types": [t for _, t, _ in runs],
        "run_ids": [r for r, _, _ in runs],
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(out_path).open("wb") as fh:
        pickle.dump(data, fh)
```

Match the existing `_load_labels` helper and `read_events` import names exactly as they already exist in `cli.py` (read the file first — do not invent names). If labels lack `malicious_pids` the existing default logic (`run_type = "real"`, empty malicious) must be preserved.

- [ ] **Step 6: Update `load_pickled_graphs` to pass run_id**

In `neurobpf/gnn/dataset.py`:

```python
def load_pickled_graphs(path):
    with Path(path).open("rb") as fh:
        data = pickle.load(fh)
    graphs = [
        snapshot_to_graph(snap, i, run_id)
        for i, (snap, run_id) in enumerate(zip(data["snapshots"], data["run_ids"]))
    ]
    return graphs, data["run_types"], data["run_ids"]
```

- [ ] **Step 7: Run full suite; fix any test that assumed one snapshot per run**

Run: `./.venv/bin/pytest tests/ -x`
Expected: existing failures limited to tests that assumed 1 snapshot/run (e.g. `test_cli_gnn.py`, `test_dataset.py::test_load_pickled_graphs_roundtrip` if it builds multi-run data). Update those tests minimally: where a test constructs `data["snapshots"]` with a single snapshot per run it still works (run_ids length 1); where generation now yields multiple windows, assert counts via `len(set(run_ids))` or `sum(1 for r in run_ids if r == "normal_0")`.

- [ ] **Step 8: Verify the previously-written test passes and commit**

Run: `./.venv/bin/pytest tests/test_cli.py::test_build_keeps_all_temporal_windows tests/test_dataset.py -v`
Then:

```bash
git add neurobpf/cli.py neurobpf/gnn/features.py neurobpf/gnn/dataset.py tests/test_cli.py tests/test_dataset.py
git commit -m "feat: retención de ventanas temporales con run_id por grafo"
```

---

### Task 4: Real train/val/test split + early stopping + val-calibrated threshold

**Files:**
- Modify: `neurobpf/gnn/train.py`, `neurobpf/gnn/dataset.py` (add `split_by_run`), `neurobpf/cli.py` (train/detect use split)
- Test: `tests/test_train.py`, `tests/test_dataset.py`, `tests/test_cli_gnn.py`

**Context:** `train_gae` validates on `train_graphs[-1]` (leak). `compute_threshold` calibrates on training graphs. Need: train/val/test split at run granularity — split train (normal), val (normal), test (normal + attack) so attacks never appear in train, and no graph trained on is validated on or thresholded from.

**Interfaces:**
- Produces: `split_by_run(graphs, train_frac=0.6, val_frac=0.2, seed=0) -> (train, val, test)` where each is a list of `GraphTensor` and the run assignments are stratified by `run_type` at the run level (all windows of a run go to exactly one split).
- Produces: `train_gae(train_graphs, val_graphs, ...)` — adds required `val_graphs` param; early stopping uses val loss; signature otherwise unchanged.
- Produces: `compute_threshold(model, val_graphs, percentile=99.0)` — unchanged signature but callers switch to passing val graphs.
- Breaks: callers passing 3 args to `train_gae`; all call sites updated in-cli: `cli._normal_graphs` split; `test_cli_gnn.py` train/detect steps updated to exercise the split path.

- [ ] **Step 1: Write the failing split test**

Append to `tests/test_dataset.py`:

```python
def _windows_runs():
    graphs = []
    for r in range(6):
        snap = _snapshot()
        for w in range(3):
            g = snapshot_to_graph(snap, r * 3 + w, run_id=f"run{r}")
            graphs.append(g)
    return graphs


def test_split_by_run_no_mixed_runs():
    graphs = _windows_runs()
    train, val, test = split_by_run(graphs, train_frac=0.5, val_frac=0.25, seed=0)
    tr_ids = {g.run_id for g in train}
    va_ids = {g.run_id for g in val}
    te_ids = {g.run_id for g in test}
    assert tr_ids & va_ids == set()
    assert tr_ids & te_ids == set()
    assert va_ids & te_ids == set()
    assert all(g.run_id in tr_ids for g in train)
    assert len(set.union(tr_ids, va_ids, te_ids)) == 6
```

- [ ] **Step 2: Run to verify it fails**

Run: `./.venv/bin/pytest tests/test_dataset.py::test_split_by_run_no_mixed_runs -v`
Expected: FAIL — `split_by_run` undefined.

- [ ] **Step 3: Implement `split_by_run`**

In `neurobpf/gnn/dataset.py`:

```python
def split_by_run(graphs, train_frac=0.6, val_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    runs = sorted({g.run_id for g in graphs})
    rng.shuffle(runs)
    n_train = max(1, int(round(len(runs) * train_frac)))
    n_val = max(0, int(round(len(runs) * val_frac)))
    train_runs = set(runs[:n_train])
    val_runs = set(runs[n_train:n_train + n_val])
    test_runs = set(runs[n_train + n_val:])
    train = [g for g in graphs if g.run_id in train_runs]
    val = [g for g in graphs if g.run_id in val_runs]
    test = [g for g in graphs if g.run_id in test_runs]
    return train, val, test
```

Add `import numpy as np` if not present in dataset.py.

- [ ] **Step 4: Write failing early-stopping test**

Append to `tests/test_train.py`:

```python
def test_train_uses_val_graphs_for_early_stop():
    graphs = [snapshot_to_graph(_mini_graph(n_extra=i), i, run_id=f"r{i}") for i in range(4)]
    train = graphs[:2]
    val = graphs[2:]
    model = train_gae(train, val_graphs=val, hidden_dim=8, z_dim=8, epochs=5, seed=0)
    assert model is not None
```

- [ ] **Step 5: Run to verify it fails**

Run: `./.venv/bin/pytest tests/test_train.py::test_train_uses_val_graphs_for_early_stop -v`
Expected: FAIL — `train_gae` takes unexpected keyword `val_graphs`.

- [ ] **Step 6: Implement early stopping on val**

In `neurobpf/gnn/train.py`, change the signature and the validation block:

```python
def train_gae(
    train_graphs,
    val_graphs=None,
    hidden_dim=64,
    z_dim=64,
    epochs=150,
    lr=1e-3,
    patience=20,
    seed=0,
):
    torch.manual_seed(seed)
    in_dim = train_graphs[0].x.shape[1]
    model = GAE(in_dim=in_dim, hidden_dim=hidden_dim, z_dim=z_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = None
    best_state = None
    stalled = 0
    for _ in range(epochs):
        model.train()
        total = 0.0
        for g in train_graphs:
            adj = make_adj(g)
            optimizer.zero_grad()
            loss = model.train_loss(g.x, adj)
            loss.backward()
            optimizer.step()
            total += loss.item()
        num = None
        for g in (val_graphs if val_graphs else []):
            with torch.no_grad():
                l = model.train_loss(g.x, make_adj(g)).item()
            num = l if num is None else max(num, l)
        if num is not None and (best_loss is None or num < best_loss):
            best_loss = num
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stalled = 0
        elif num is not None:
            stalled += 1
            if stalled >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model
```

Keeps existing behavior when `val_graphs=None` except it no longer over-fits to `train_graphs[-1]`.

- [ ] **Step 7: Update existing `train_gae` callers/tests**

In `tests/test_train.py` and `tests/test_model.py`, replace bare `train_gae(graphs, ...)` calls that had no `val_graphs` only if their asserted semantics relied on `train_graphs[-1]` validation — a straightforward inspection shows only `test_train_seed_coverage_or_single_graph` and the `_models()` helper call it; with `val_graphs=None` the path still returns a trained eval-mode model. Leave them, but confirm `_models()` still passes (loss decrease relies on training, which still happens over the same graphs).

- [ ] **Step 8: Update CLI train/detect to split + val threshold**

Read `neurobpf/cli.py` `train`, `detect`, `annotate`, `demo` subcommands. Change them to:
- Load graphs → `split_by_run(graphs, seed=seed)` only on *normal* runs for train/val where available.
- `train_gae(train_part, val_graphs=val_part, ...)`.
- `compute_threshold(model, val_part)` in detect/annotate/demo (fall back to normal graphs if val empty).
Make minimal edits so the `detect` CLI evaluates **test** runs for the headline AUC and keeps backward-compatible keys.

- [ ] **Step 9: Run full suite and fix breakages**

Run: `./.venv/bin/pytest tests/ -x`
Expected: update `test_cli_gnn.py` assertions on `run_ids` length and split-driven behavior if they hard-code single-run pkl outputs. Keep productions identical to CLI changes.

- [ ] **Step 10: Commit**

```bash
git add neurobpf/gnn/dataset.py neurobpf/gnn/train.py neurobpf/cli.py tests/test_dataset.py tests/test_train.py tests/test_cli_gnn.py
git commit -m "feat: split train/val/test por run con early stopping real"
```

---

### Task 5: Paper-grade metric suite — nDCG, precision@k, FPR@budget, threshold-aware evaluate

**Files:**
- Modify: `neurobpf/gnn/detect.py`
- Test: `tests/test_detect.py`

**Context:** `evaluate()` ignores `threshold`. Need run-level metrics: group per-run graphs by `run_id`, aggregate per-node scores as the max z-score across windows, labels as any-malicious, then compute over the pooled test set: ROC-AUC, recall@k, precision@k, nDCG, FPR@budget. Threshold used to report flagged-positive/false-negative counts at the calibrated operating point.

**Interfaces:**
- Produces:
  - `recall_at_k(scores, labels, k)` (reuse existing `_recall_at_k_for_run` logic renamed and exposed).
  - `precision_at_k(scores, labels, k) -> float`
  - `ndcg_at_k(scores, labels, k) -> float` with binary gain `2^rel - 1`, DCG normalized by ideal DCG.
  - `fpr_at_budget(scores, labels, budget) -> float` = fraction of benign nodes among the top-`budget` alerts.
  - `threshold_report(scores, labels, threshold) -> dict` with `{"tp", "fp", "fn", "tn", "tpr", "fpr"}` at score ≥ threshold.
  - `run_level_evaluate(model, graphs, threshold) -> dict` grouping by `g.run_id`; returns `{"auc_roc", "recall_at_1pct", "recall_at_5pct", "precision_at_1pct", "ndcg_at_5pct", "fpr_budget_1", "fpr_budget_5", "fpr_budget_10", "threshold_report", "per_run_recall", "n_runs"}`.
  - `evaluate()` kept for backward compat (still pooling), now also populating `"threshold_report"` via `threshold_report`.
- Consumes: `node_anomaly_scores`, `_standardize` (internal), `GraphTensor`.

- [ ] **Step 1: Write the failing metric tests**

Append to `tests/test_detect.py`:

```python
def test_precision_and_recall_at_k():
    scores = [0.9, 0.8, 0.7, 0.3]
    labels = [True, False, True, False]
    assert precision_at_k(scores, labels, 2) == 0.5
    assert recall_at_k(scores, labels, 2) == 0.5


def test_ndcg_at_k():
    scores = [0.9, 0.8, 0.7, 0.3, 0.2]
    labels = [False, True, True, False, True]
    k = 3
    order = np.argsort(scores)[::-1][:k]
    ranks = [labels[i] for i in order]
    ideal = sorted([l for l in labels], reverse=True)[:k]
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(ranks))
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal))
    assert abs(ndcg_at_k(scores, labels, k) - dcg / idcg) < 1e-9


def test_fpr_at_budget():
    scores = [0.9, 0.8, 0.7, 0.6, 0.1]
    labels = [True, False, False, True, False]
    assert fpr_at_budget(scores, labels, 2) == 1.0
    assert fpr_at_budget(scores, labels, 4) == 0.5
```

- [ ] **Step 2: Run to verify they fail**

Run: `./.venv/bin/pytest tests/test_detect.py -k "precision or ndcg or fpr_at_budget" -v`
Expected: FAIL — names undefined.

- [ ] **Step 3: Implement metrics**

Append to `neurobpf/gnn/detect.py`:

```python
def precision_at_k(scores, labels, k):
    order = np.argsort(scores)[::-1][:k]
    return float(sum(1 for i in order if labels[i]) / k)


def recall_at_k(scores, labels, k):
    if k == 0:
        return 1.0
    order = np.argsort(scores)[::-1][:k]
    return float(sum(1 for i in order if labels[i]) / k)


def ndcg_at_k(scores, labels, k):
    order = np.argsort(scores)[::-1]
    gains = np.asarray([1.0 if labels[i] else 0.0 for i in order], dtype=float)[:k]
    ideal = np.sort(np.asarray(labels, dtype=float))[::-1][:k]
    dcg = float(np.sum(gains / np.log2(np.arange(2, len(gains) + 2))))
    idcg = float(np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2))))
    return dcg / idcg if idcg > 0 else 0.0


def fpr_at_budget(scores, labels, budget):
    order = np.argsort(scores)[::-1][:budget]
    flagged = [labels[i] for i in order]
    benign = sum(1 for f in flagged if not f)
    total_benign = sum(1 for l in labels if not l)
    return float(benign / total_benign) if total_benign else 0.0


def threshold_report(scores, labels, threshold):
    flags = np.asarray(scores) >= threshold
    y = np.asarray(labels, dtype=bool)
    tp = int((flags & y).sum())
    fp = int((flags & ~y).sum())
    fn = int((~flags & y).sum())
    tn = int((~flags & ~y).sum())
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "tpr": float(tp / (tp + fn)) if tp + fn else 0.0,
        "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
    }
```

- [ ] **Step 4: Implement `run_level_evaluate`**

Append to `neurobpf/gnn/detect.py`:

```python
def run_level_evaluate(model, graphs, threshold):
    runs = {}
    for i, g in enumerate(graphs):
        runs.setdefault(g.run_id, []).append(i)
    run_scores = []
    run_labels = []
    per_run = {}
    for rid, idxs in runs.items():
        raw = np.stack([node_anomaly_scores(model, graphs[i]) for i in idxs])
        score = np.max(raw, axis=0)
        labels = np.any(np.stack([np.asarray(graphs[i].malicious, dtype=bool) for i in idxs]), axis=0)
        run_scores.append(score)
        run_labels.append(labels)
        if labels.sum() > 0:
            k = labels.sum()
            per_run[rid] = recall_at_k(score, labels, k)
    scores = np.concatenate(run_scores) if run_scores else np.array([])
    labels = np.concatenate(run_labels) if run_labels else np.array([])
    n_pos = int(labels.sum())
    n_total = len(labels)
    k1 = max(int(round(n_total * 0.01)), 1)
    k5 = max(int(round(n_total * 0.05)), 1)
    return {
        "auc_roc": _roc_auc(scores, labels),
        "recall_at_1pct": recall_at_k(scores, labels, k1),
        "recall_at_5pct": recall_at_k(scores, labels, k5),
        "precision_at_1pct": precision_at_k(scores, labels, k1),
        "ndcg_at_5pct": ndcg_at_k(scores, labels, k5),
        "fpr_budget_1": fpr_at_budget(scores, labels, 1),
        "fpr_budget_5": fpr_at_budget(scores, labels, 5),
        "fpr_budget_10": fpr_at_budget(scores, labels, 10),
        "threshold_report": threshold_report(scores, labels, threshold),
        "per_run_recall": per_run,
        "n_runs": len(runs),
        "n_pos": n_pos,
    }
```

- [ ] **Step 5: Add threshold to existing `evaluate` output**

In `evaluate()`, after computing metrics, add:

```python
    return {
        "auc_roc": float(auc),
        "recall_at_topk": recall_at_topk,
        "per_run_recall": run_recalls,
        "threshold_report": threshold_report(all_scores, all_labels, threshold),
    }
```

- [ ] **Step 6: Write run-level integration test**

Append to `tests/test_detect.py`:

```python
def test_run_level_evaluate_groups_by_run_id():
    from neurobpf.gnn.detect import run_level_evaluate
    graphs, scores = _eval_inputs()
    for g, r in zip(graphs, ["a", "b", "b"]):
        g.run_id = r
    model = _fake_model_for(graphs, scores)
    res = run_level_evaluate(model, graphs, threshold=0.5)
    assert res["n_runs"] == 2
    assert "auc_roc" in res
    assert set(res["threshold_report"]) >= {"tp", "fp", "fn", "tn", "tpr", "fpr"}
```

Update `_eval_inputs`'s graphs to accept setting `.run_id` (GraphTensor has the field from Task 3).

- [ ] **Step 7: Run full suite**

Run: `./.venv/bin/pytest tests/ -x`
Expected: PASS (existing `evaluate` output now has extra key; existing tests use `>=` set checks, compatible).

- [ ] **Step 8: Commit**

```bash
git add neurobpf/gnn/detect.py tests/test_detect.py
git commit -m "feat: métricas de paper nDCG/precision/fpr-budget con threshold"
```

---

### Task 6: Feature-only baselines — OCSVM + MLP autoencoder

**Files:**
- Create: `neurobpf/gnn/baselines.py`
- Modify: `pyproject.toml` (add scikit-learn)
- Test: `tests/test_baselines.py`

**Context:** To show graph topology adds value, compare against feature-only detectors on the same per-node feature vectors (`GraphTensor.x`), with the same run-level protocol.

**Interfaces:**
- Produces:
  - `ocsvm_scores(features: np.ndarray) -> np.ndarray`: fit `sklearn.svm.OneClassSVM(nu=0.1)`; return `-decision_function`.
  - `mlp_ae_scores(features: torch.Tensor, hidden=32, epochs=100, seed=0) -> np.ndarray`: train a small ReLU autoencoder (dim→hidden→dim, MSE loss) and return per-row reconstruction error (mean squared over the 12 features).
  - `baseline_run_level(model_fn, graphs) -> dict`: same protocol as `run_level_evaluate` but scoring with `model_fn(g.x)`.
- Consumes: `GraphTensor`, `run_level_evaluate`'s grouping helper (reuse `run_level_evaluate` after replacing the score source: extract the grouping into a private `_group_scores(node_score_fn, graphs)` in `detect.py`, or implement grouping here — plan says: add `_group_run_scores(scores_by_run, graphs)` helper in `detect.py` and reuse it inside `run_level_evaluate`).

- [ ] **Step 1: Add dependency and install**

```bash
./.venv/bin/pip install scikit-learn
```

Add `"scikit-learn>=1.5"` to `pyproject.toml` dependencies.

- [ ] **Step 2: Add grouping helper to detect.py**

In `neurobpf/gnn/detect.py`, refactor the run grouping so baselines can reuse it:

```python
def _group_run_metrics(scores_by_graph, graphs):
    runs = {}
    for i, g in enumerate(graphs):
        runs.setdefault(g.run_id, []).append(i)
    run_scores = []
    run_labels = []
    per_run = {}
    for rid, idxs in runs.items():
        score = np.max(np.stack([np.asarray(scores_by_graph[i]) for i in idxs]), axis=0)
        labels = np.any(np.stack([np.asarray(graphs[i].malicious, dtype=bool) for i in idxs]), axis=0)
        run_scores.append(score)
        run_labels.append(labels)
        if labels.sum() > 0:
            per_run[rid] = recall_at_k(score, labels, int(labels.sum()))
    scores = np.concatenate(run_scores)
    labels = np.concatenate(run_labels)
    return scores, labels, per_run
```

Rewrite `run_level_evaluate` to call `_group_run_metrics([node_anomaly_scores(model, g) for g in graphs], graphs)`. Update existing test expectations (same values, keep `n_runs`/`n_pos` computed from `runs`).

- [ ] **Step 3: Write the failing baseline tests**

Create `tests/test_baselines.py`:

```python
import numpy as np
import torch

from neurobpf.gnn.baselines import mlp_ae_scores, ocsvm_scores


def test_ocsvm_scores_shape_and_anomalous_scores_higher():
    rng = np.random.default_rng(0)
    feats = np.concatenate([
        rng.normal(0, 0.1, size=(50, 5)),
        rng.normal(10, 0.1, size=(3, 5)),
    ], axis=0).astype(float)
    s = ocsvm_scores(feats)
    assert s.shape == (53,)
    assert s[-3:].mean() > s[:-3].mean()


def test_mlp_ae_scores_outliers_above_normal():
    rng = np.random.default_rng(1)
    feats = np.concatenate([
        rng.normal(0, 0.1, size=(50, 5)),
        rng.normal(10, 0.1, size=(3, 5)),
    ], axis=0).astype(float)
    s = mlp_ae_scores(torch.tensor(feats, dtype=torch.float32), epochs=150, seed=0)
    assert s.shape == (53,)
    assert s[-3:].mean() > s[:-3].mean() * 1.2
```

- [ ] **Step 4: Run to verify they fail**

Run: `./.venv/bin/pytest tests/test_baselines.py -v`
Expected: FAIL — module missing.

- [ ] **Step 5: Implement `baselines.py`**

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.svm import OneClassSVM


class _AutoEncoder(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.enc = nn.Linear(in_dim, hidden)
        self.dec = nn.Linear(hidden, in_dim)

    def forward(self, x):
        h = F.relu(self.enc(x))
        return self.dec(h)


def ocsvm_scores(features):
    clf = OneClassSVM(nu=0.1)
    clf.fit(features)
    return -clf.decision_function(features)


def mlp_ae_scores(features, hidden=32, epochs=100, seed=0):
    torch.manual_seed(seed)
    x = torch.as_tensor(features, dtype=torch.float32)
    model = _AutoEncoder(x.shape[1], hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = F.mse_loss(model(x), x)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        err = F.mse_loss(model(x), x, reduction="none").mean(dim=1).numpy()
    return err
```

- [ ] **Step 6: Run baseline tests pass**

Run: `./.venv/bin/pytest tests/test_baselines.py -v`
Expected: PASS.

- [ ] **Step 7: Run full suite + commit**

Run: `./.venv/bin/pytest tests/`
Then:

```bash
git add neurobpf/gnn/baselines.py neurobpf/gnn/detect.py pyproject.toml tests/test_baselines.py tests/test_detect.py
git commit -m "feat: baselines feature-only OCSVM y MLP autoencoder"
```

---

### Task 7: Ablations — topology ablation via adjacency shuffling

**Files:**
- Modify: `neurobpf/gnn/ablation.py` (new)
- Test: `tests/test_ablation.py`

**Context:** To attribute detection power to topology, ablate the GNN by shuffling adjacency rows per graph — the same model architecture, same embeddings, but the graph structure is destroyed. Compare vs the intact topology.

**Interfaces:**
- Produces: `shuffled_graphs(graphs, seed=0) -> list[GraphTensor]`: for each graph copy, permute `edge_index` nodes via a fixed random permutation of node ids (keep x and malicious untouched). Return new `GraphTensor`s with `run_id` preserved.
- Consumes: `GraphTensor`, `from neurobpf.gnn.features import GraphTensor`; test uses `run_level_evaluate`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ablation.py`:

```python
import torch

from neurobpf.gnn import ablation
from neurobpf.gnn.dataset import make_adj
from neurobpf.gnn.features import FEAT_DIM, snapshot_to_graph
from neurobpf.gnn.model import GAE
from neurobpf.graphbuilder import EdgeData, NodeData, Snapshot


def _graph():
    nodes = {"hub": NodeData("hub", "process", "hub", 1000, 1)}
    edges = []
    for k in range(4):
        nid = f"leaf{k}"
        nodes[nid] = NodeData(nid, "process", nid, 1000, k + 2)
        edges.append(EdgeData("hub", nid, "EXEC", 1.0))
    return Snapshot(0.0, nodes, edges, {"leaf1", "leaf3"})


def test_shuffled_graphs_preserve_nodes_but_permute_adjacency():
    g = snapshot_to_graph(_graph(), 0, run_id="r0")
    out = ablation.shuffled_graphs([g], seed=0)
    assert len(out) == 1
    s = out[0]
    assert s.node_ids == g.node_ids
    assert s.malicious == g.malicious
    assert not torch.equal(s.edge_index, g.edge_index)


def test_shuffled_deterministic_per_seed():
    g = snapshot_to_graph(_graph(), 0, run_id="r0")
    a = ablation.shuffled_graphs([g], seed=7)[0]
    b = ablation.shuffled_graphs([g], seed=7)[0]
    assert torch.equal(a.edge_index, b.edge_index)
```

- [ ] **Step 2: Run to verify they fail**

Run: `./.venv/bin/pytest tests/test_ablation.py -v`
Expected: FAIL — module `neurobpf.gnn.ablation` missing.

- [ ] **Step 3: Implement `ablation.py`**

```python
import torch

from neurobpf.gnn.features import GraphTensor


def shuffled_graphs(graphs, seed=0):
    out = []
    for gi, g in enumerate(graphs):
        n = g.x.shape[0]
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed + gi))
        assert (perm == torch.arange(n)).sum().item() < n
        import copy
        s = copy.copy(g)
        s.edge_index = perm[g.edge_index]
        out.append(s)
    return out
```

- [ ] **Step 4: Run to verify pass + self-review the assertion**

Run: `./.venv/bin/pytest tests/test_ablation.py -v`
Expected: PASS. Confirm the `assert perm != identity` guard cannot loop (if a collision occurs the model would not degrade; acceptable for large graphs; for tiny graphs the guard documents the limitation).

- [ ] **Step 5: Commit**

```bash
git add neurobpf/gnn/ablation.py tests/test_ablation.py
git commit -m "feat: ablación de topología con shuffling de adjacencia"
```

---

### Task 8: `compare` command — multi-seed experiment table

**Files:**
- Modify: `neurobpf/cli.py` (add `compare` + `_cmd_compare` + parser)
- Test: `tests/test_cli_compare.py`

**Context:** Papers need multi-seed runs with mean±std over the paper metrics. `compare` runs the full pipeline (train on train-split, evaluate on test-split) for GNN-intact, GNN-shuffled, OCSVM, MLP-AE across N seeds and writes a report.

**Interfaces:**
- Produces: `cli.compare(graphs_path, out_path, seeds=5, hidden_dim=128, z_dim=128, epochs=300) -> None` writes `{"per_seed": [{seed, method, auc_roc, recall_at_5pct, ndcg_at_5pct, fpr_budget_5, recall_at_1pct}], "summary": {"<method>|<metric>": {"mean", "std"}}}`.
- Produces: `_cmd_compare(args)` — CLI wrapper matching the `main()` argparse pattern (existing `_add_train_args` style).
- Consumes: `split_by_run`, `train_gae`, `compute_threshold`, `_group_run_metrics` (shifted pooling helper defined in Task 6), `run_level_evaluate` (Task 5), `ocsvm_scores`/`mlp_ae_scores` (Task 6), `shuffled_graphs` (Task 7), plus `_paper_metrics(scores, labels)` — a small helper defined in detect.py (Task 5) that computes the four reported metrics from pooled scores+labels so test/summary logic lives in one place.

**IMPORTANT** (corrects an earlier draft): baseline scoring reuses `node_anomaly_scores`'s pool path through `run_level_evaluate`. For the two feature-only baselines, build per-graph scores and pass them to `_group_run_metrics` directly, then also run `threshold_report` for completeness — do NOT invent a separate `run_level_evaluate_over_scores` function.

- [ ] **Step 1: Write the failing CLI test**

Create `tests/test_cli_compare.py`:

```python
import json
import pickle

import neurobpf.cli as cli
from neurobpf.graphbuilder import EdgeData, NodeData, Snapshot


def _snap(malicious):
    nodes = {"hub": NodeData("hub", "process", "hub", 1000, 1)}
    edges = []
    for k in range(4):
        nid = f"leaf{k}"
        nodes[nid] = NodeData(nid, "process", nid, 1000, k + 2)
        edges.append(EdgeData("hub", nid, "EXEC", 1.0))
    return Snapshot(0.0, nodes, edges, malicious)


def _pkl(tmp_path):
    runs = []
    for i in range(6):
        runs.append((f"n{i}", "normal", _snap(set())))
    for i in range(3):
        runs.append((f"a{i}", "attack", _snap({"leaf1"})))
    data = {
        "snapshots": [s for _, _, s in runs],
        "run_types": [t for _, t, _ in runs],
        "run_ids": [r for r, _, _ in runs],
    }
    p = tmp_path / "g.pkl"
    with p.open("wb") as fh:
        pickle.dump(data, fh)
    return p


def test_compare_writes_report(tmp_path):
    p = _pkl(tmp_path)
    out = tmp_path / "rep.json"
    cli.compare(p, out, seeds=1)
    rep = json.loads(out.read_text())
    assert set(rep) == {"per_seed", "summary"}
    assert rep["per_seed"][0]["method"] in {"gnn", "gnn_shuffled", "ocsvm", "mlp_ae"}
```

- [ ] **Step 2: Run to verify it fails**

Run: `./.venv/bin/pytest tests/test_cli_compare.py -v`
Expected: FAIL — `cli.compare` missing.

- [ ] **Step 3: Add `_paper_metrics` + `_group_run_metrics` wiring in detect.py (if not yet done in Task 6)**

In Task 6 you added `_group_run_metrics(scores_by_graph, graphs)`. Add this to its module:

```python
def _paper_metrics(scores, labels, n_total=None):
    n_total = n_total if n_total is not None else len(scores)
    k1 = max(int(round(n_total * 0.01)), 1)
    k5 = max(int(round(n_total * 0.05)), 1)
    return {
        "auc_roc": _roc_auc(scores, labels),
        "recall_at_1pct": recall_at_k(scores, labels, k1),
        "recall_at_5pct": recall_at_k(scores, labels, k5),
        "ndcg_at_5pct": ndcg_at_k(scores, labels, k5),
        "fpr_budget_5": fpr_at_budget(scores, labels, 5),
    }


def run_level_evaluate(model, graphs, threshold):
    scores_by_graph = [node_anomaly_scores(model, g) for g in graphs]
    scores, labels, per_run = _group_run_metrics(scores_by_graph, graphs)
    res = _paper_metrics(scores, labels)
    res["threshold_report"] = threshold_report(scores, labels, threshold)
    res["per_run_recall"] = per_run
    res["n_runs"] = len({g.run_id for g in graphs})
    res["n_pos"] = int(labels.sum())
    return res
```

Replace the Run-level body you sketched earlier with this (the pooling moved into `_group_run_metrics`). Update the Task 5 integration test to keep asserting the same keys.

- [ ] **Step 4: Implement `compare` in cli.py**

```python
import numpy as np

from neurobpf.gnn.ablation import shuffled_graphs
from neurobpf.gnn.baselines import mlp_ae_scores, ocsvm_scores
from neurobpf.gnn.dataset import split_by_run
from neurobpf.gnn.detect import _paper_metrics, run_level_evaluate, threshold_report
from neurobpf.gnn.train import compute_threshold


def compare(graphs_path, out_path, seeds=5, hidden_dim=128, z_dim=128, epochs=300):
    graphs, run_types, run_ids = load_pickled_graphs(graphs_path)
    per_seed = []
    for seed in range(seeds):
        train_all, val_all, test = split_by_run(graphs, seed=seed)
        model = train_gae(
            train_all,
            val_graphs=val_all,
            hidden_dim=hidden_dim,
            z_dim=z_dim,
            epochs=epochs,
            seed=seed,
        )
        thr = compute_threshold(model, val_all)
        gnn = run_level_evaluate(model, test, thr)
        shuff_model = train_gae(
            shuffled_graphs(train_all, seed=seed),
            val_graphs=shuffled_graphs(val_all, seed=seed),
            hidden_dim=hidden_dim,
            z_dim=z_dim,
            epochs=epochs,
            seed=seed,
        )
        gnn_shuffled = run_level_evaluate(shuff_model, test, thr)
        for name, m in (("gnn", gnn), ("gnn_shuffled", gnn_shuffled)):
            per_seed.append({
                "seed": seed, "method": name,
                "auc_roc": m["auc_roc"],
                "recall_at_1pct": m["recall_at_1pct"],
                "recall_at_5pct": m["recall_at_5pct"],
                "ndcg_at_5pct": m["ndcg_at_5pct"],
                "fpr_budget_5": m["fpr_budget_5"],
            })
        for name, score_fn in (("ocsvm", ocsvm_scores), ("mlp_ae", mlp_ae_scores)):
            feats = np.stack([g.x.numpy() for g in test])
            scores_by_graph = [score_fn(g.x.numpy()) for g in test]
            sc, lab, _ = _group_run_metrics(scores_by_graph, test)
            m = _paper_metrics(sc, lab)
            per_seed.append({
                "seed": seed, "method": name,
                "auc_roc": m["auc_roc"],
                "recall_at_1pct": m["recall_at_1pct"],
                "recall_at_5pct": m["recall_at_5pct"],
                "ndcg_at_5pct": m["ndcg_at_5pct"],
                "fpr_budget_5": m["fpr_budget_5"],
            })
    summary = _summarize(per_seed)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(out_path).open("w") as fh:
        json.dump({"per_seed": per_seed, "summary": summary}, fh, indent=2)


def _summarize(per_seed):
    rows = {}
    for row in per_seed:
        key = f"{row['method']}"
        for metric in ("auc_roc", "recall_at_1pct", "recall_at_5pct", "ndcg_at_5pct", "fpr_budget_5"):
            values = np.asarray([r[metric] for r in per_seed if r["method"] == key])
            rows[f"{key} {metric}"] = {"mean": float(values.mean()), "std": float(values.std())}
    return rows
```

Then add the argparse subcommand in `main()`:

```python
    cmp = sub.add_parser("compare", help="multi-seed experiment table (GNN vs baselines vs ablations)")
    cmp.add_argument("--graphs", required=True, help="pickled graph dataset")
    cmp.add_argument("--out", required=True, help="output report JSON path")
    cmp.add_argument("--seeds", type=int, default=5)
    _add_train_args(cmp)
    cmp.set_defaults(func=_cmd_compare)
```

And:

```python
def _cmd_compare(args):
    compare(args.graphs, args.out, seeds=args.seeds,
            hidden_dim=args.hidden_dim, z_dim=args.z_dim, epochs=args.epochs)
```

- [ ] **Step 5: Run to verify pass**

Run: `./.venv/bin/pytest tests/test_cli_compare.py -v`
Expected: PASS.

- [ ] **Step 6: Run full suite, fix breakages**

Run: `./.venv/bin/pytest tests/ -x`

- [ ] **Step 7: Commit**

```bash
git add neurobpf/cli.py neurobpf/gnn/detect.py tests/test_cli_compare.py
git commit -m "feat: comando compare multi-seed con tabla de experimentos"
```

---

### Task 9: THEIA_E3 benchmark adapter (schema + sample fixture)

**Files:**
- Create: `neurobpf/benchmark/__init__.py`
- Create: `neurobpf/benchmark/theia.py`
- Create: `tests/test_theia_adapter.py`

**Context:** THEIA_E3 (Linux, ~12GB) releases DARPA audit logs. PIDSMaker provides ground truth. This task implements the adapter surface and validates it on a small hand-made fixture (no 12GB download in CI). The adapter's output feeds the existing `build` command: one NDJSON line per `Event` (via `event_to_json`), plus a labels file — exactly the format `cli._build_pickle` reads.

**Interfaces:**
- Produces:
  - `parse_theia_line(line: str) -> Event | None`: parse a single theia-form audit record into our `Event` (keyword args, default-safe). Return `None` for unsupported ops or records without a pid.
  - `theia_events_to_ndjson(events: list[Event], run_id: str) -> str`: NDJSON text, one `event_to_json` line per event (starts a "run" the way `generate` does).
  - `write_theia_run(out_dir, run_id, events, malicious_pids)`: writes `<run_id>.events.ndjson` + `<run_id>.labels.json` with `{"run_type": "theia", "malicious_pids": [...]}`.
  - `theia_ground_truth_to_pids(path) -> list[int]`: PIDSMaker-style CSV (`...,pid,...`) → malicious pids.
- Consumes: `neurobpf/events.py` (`Event`, `event_to_json`).
- Constraint: no scipy/sklearn; pure stdlib + numpy.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_theia_adapter.py`:

```python
import json

from neurobpf.benchmark.theia import (
    parse_theia_line,
    theia_events_to_ndjson,
    theia_ground_truth_to_pids,
    write_theia_run,
)
from neurobpf.events import event_from_json


def test_parse_theia_event_record():
    line = '{"local:subject":{"pid":123,"process":{"com":"bash"}},"local:op":"execve","subject":123}'
    ev = parse_theia_line(line)
    assert ev is not None
    assert ev.event == "process_start"
    assert ev.pid == 123
    assert ev.comm == "bash"


def test_parse_theia_unsupported_returns_none():
    assert parse_theia_line('{"local:op":"mem_read","a":1}') is None


def test_theia_events_roundtrip_through_event_json():
    e1 = parse_theia_line('{"local:subject":{"pid":1,"process":{"com":"init"}},"local:op":"execve","subject":1}')
    e2 = parse_theia_line('{"local:subject":{"pid":5,"process":{"com":"sshd"}},"local:op":"execve","subject":5}')
    text = theia_events_to_ndjson([e1, e2], "theia_0")
    lines = [event_from_json(l) for l in text.strip().splitlines()]
    assert [e.pid for e in lines] == [1, 5]


def test_write_theia_run_writes_pipeline_readable_files(tmp_path):
    e = parse_theia_line('{"local:subject":{"pid":9,"process":{"com":"bash"}},"local:op":"execve","subject":9}')
    write_theia_run(tmp_path, "theia_0", [e], [9])
    ev = tmp_path / "theia_0.events.ndjson"
    lab = tmp_path / "theia_0.labels.json"
    assert ev.exists() and lab.exists()
    assert event_from_json(ev.read_text().strip()).pid == 9
    assert json.loads(lab.read_text()) == {"run_type": "theia", "malicious_pids": [9]}


def test_theia_ground_truth_pids(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_text("head,pid\nrow,111\nrow,222")
    assert theia_ground_truth_to_pids(p) == [111, 222]
```

- [ ] **Step 2: Run to verify they fail**

Run: `./.venv/bin/pytest tests/test_theia_adapter.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `theia.py`**

Note the exact `Event` field order: `ts, event, pid, ppid=-1, comm="", exe_path="", uid=1000, target_type=None, target=None, result=0`. Use keyword args.

```python
import csv
import json
from pathlib import Path

from neurobpf.events import Event, event_to_json

_OP_TO_EVENT = {
    "execve": "process_start",
    "open": "file_read",
    "read": "file_read",
    "write": "file_write",
    "unlink": "file_unlink",
    "connect": "net_connect",
}


def parse_theia_line(line: str) -> Event | None:
    rec = None
    try:
        rec = json.loads(line)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(rec, dict):
        return None
    op = rec.get("local:op") or rec.get("op") or ""
    if op not in _OP_TO_EVENT:
        return None
    subject = rec.get("subject")
    raw = subject.get("local:subject") if isinstance(subject, dict) else subject
    if isinstance(raw, dict) and "subject" in raw:
        raw = raw["subject"]
    if not isinstance(raw, dict):
        return None
    pid = raw.get("pid")
    if pid is None:
        return None
    proc = raw.get("process") if isinstance(raw.get("process"), dict) else {}
    comm = proc.get("com", "") if isinstance(proc, dict) else ""
    args = rec.get("arguments")
    target = ""
    if isinstance(args, dict):
        target = args.get("subject path") or args.get("dst") or args.get("path") or ""
    name = _OP_TO_EVENT[op]
    return Event(
        ts=float(rec.get("ts", 0.0)),
        event=name,
        pid=int(pid),
        comm=str(comm),
        target_type="process" if name == "process_start" else ("net" if "net" in name else "file"),
        target=str(target) or None,
    )


def theia_events_to_ndjson(events, run_id):
    return "\n".join(event_to_json(e).rstrip("\n") for e in events) + "\n"


def write_theia_run(out_dir, run_id, events, malicious_pids):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{run_id}.events.ndjson").write_text(theia_events_to_ndjson(events, run_id))
    (out_dir / f"{run_id}.labels.json").write_text(json.dumps(
        {"run_type": "theia", "malicious_pids": list(malicious_pids)}
    ))


def theia_ground_truth_to_pids(path):
    pids = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            pid = row.get("pid")
            if pid is not None:
                try:
                    pids.append(int(pid))
                except ValueError:
                    continue
    return pids
```

- [ ] **Step 4: Run to verify pass**

Run: `./.venv/bin/pytest tests/test_theia_adapter.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add neurobpf/benchmark/__init__.py neurobpf/benchmark/theia.py tests/test_theia_adapter.py
git commit -m "feat: adapter benchmark THEIA_E3 con fixture de test"
```

---

### Task 10: Final verification — full suite, e2e demo, artifacts

**Files:**
- Run coverage: `./.venv/bin/pytest tests/ | tail -1`
- Run e2e: `./.venv/bin/python -m neurobpf.cli demo --out /tmp/opencode/research_demo --seed 7 --normal 4 --attack 3` — must still produce AUC line, `.pkl`, `detection.json`, annotated JSON, and WebSocket-servable `annotated/`.
- Run `compare`: `./.venv/bin/python -m neurobpf.cli compare --graphs <demo pkl> --seeds 3 --out /tmp/opencode/research_compare.json` — verify table populated.
- Update README (Spanish) briefly: temporal windows, split, metric names, compare command, benchmark adapter existence.

**Interfaces:**
- Produces: README research-prototype section.

- [ ] **Step 1: Full suite green**

Run: `./.venv/bin/pytest tests/ -q | tail -1`
Expected: no failures.

- [ ] **Step 2: Demo e2e still works**

Run: the demo command above. Confirm it exits 0 and prints `AUC ... | recall@k ...`; confirm `annotated/` non-empty.

- [ ] **Step 3: compare command e2e**

Run: `./.venv/bin/python -m neurobpf.cli compare --graphs ... --seeds 3 --out /tmp/opencode/research_compare.json`
Expected: JSON with `per_seed` and `summary`, 4 methods.

- [ ] **Step 4: Update README in Spanish**

Add a "Prototipo de investigación" section covering: all temporal windows retained per run; train/val/test split by run with val early-stopping; metric suite (ROC-AUC, recall@k, precision@k, nDCG, FPR@budget); baselines OCSVM/MLP-AE; topology ablation via adjacency shuffling; `compare` multi-seed command; THEIA_E3 adapter (needs 12GB download, parser + ground-truth reader provided).

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: sección prototipo de investigación en README"
```

- [ ] **Step 6: Push (controller)**

```bash
git push
```

---

## Self-Review Checklist

- Spec coverage: Task 1 covers `is_root` bug (spec: features fix). Task 2 sparse (spec §2). Task 3 temporal (spec §1). Task 4 split + early stop + val threshold (spec §1). Task 5 metrics (spec §4). Task 6 baselines (spec §3). Task 7 ablations (spec §3). Task 8 compare multi-seed (spec §4). Task 9 THEIA adapter (spec §5). Task 10 README (spec's deliverables). No gaps.
- Placeholder scan: `compare` keeps one stub (`run_level_evaluate_over_scores`) flagged inline with exact refactor instructions; all code blocks complete.
- Type consistency: `make_sparse_adj` produced Task 2, consumed in Task 2 model tests and implicitly in cli later; `split_by_run` produced Task 4 consumed Task 8; `run_id` field produced Task 3 consumed Tasks 4/5/8; `_group_run_metrics` produced Task 6 consumed Task 8; `nDCG`/`fpr_at_budget` produced Task 5 consumed in `_summarize` (Task 8) and tested.