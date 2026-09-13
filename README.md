# NeuroBPF

Pipeline de detección de intrusiones a nivel de kernel: captura eBPF + motor de escenarios sintéticos → grafo de procedencia temporal → GNN de autoencoder de reconstrucción (entrenada solo con comportamiento normal) → visualizador 2D/3D en vivo.

El resultado es una herramienta de demostración (portfolio) que detecta ataques **desconocidos** —aquellos no vistos durante el entrenamiento— modelando el comportamiento normal y destacando las desviaciones como anomalías por nodo.

## Arquitectura

```
 ┌──────────────┐    NDJSON     ┌─────────────────┐    pickle    ┌────────────────────┐
 │ eBPF (C)     │──────────────▶│ provenance      │─────────────▶│ GNN autoencoder     │
 │ collector    │  eventos de   │ snapshots       │  grafos      │ (solo entrenamiento │
 │ kernel       │  syscalls     │ temporales      │  por run     │  con datos normales)│
 └──────────────┘               └─────────────────┘              └────────┬───────────┘
                                                            scores anomalía │
 ┌──────────────┐    NDJSON     ┌─────────────────┐              ┌──────────▼───────────┐
 │ motor de     │──────────────▶│  (mismo camino) │─────────────▶│ anotación por run    │
 │ escenarios   │  5 ataques +  │                 │              │ (z-score vs umbral   │
 │ sintéticos   │  normal       │                 │              │  percentil 99)       │
 └──────────────┘               └─────────────────┘              └──────────┬───────────┘
                                                                            │ WebSocket
                                                          ┌─────────────────▼─────────┐
                                                          │ FastAPI replay +           │
                                                          │ visualizador React 2D/3D   │
                                                          └───────────────────────────┘
```

Los datos reales (eBPF) y los sintéticos comparten el mismo esquema NDJSON, por lo que el resto del pipeline es idéntico para ambos.

## Detección

- **GNN**: autoencoder de reconstrucción (pila de GCN + producto interno). Se entrena exclusivamente con runs normales; reconstruye el grafo de procedencia y aprende a "explicar" el comportamiento normal.
- **Score por nodo**: error de reconstrucción BCE de las aristas del grafo dado el nodo.
- **Umbral**: percentil 99 de la distribución de scores **normalizados (z-score por run)** de los runs normales de entrenamiento. Nodo cuyo z-score supera el umbral → anómalo. La normalización por run hace el umbral invariante a la escala del grafo (un run de ataque acumulado no marca todo el grafo).
- **Métricas**: AUC ROC y recall@top-k por run, con el score de anomalía calculado sobre datos de ataque no vistos.

## Prototipo de investigación

Protocolo experimental con métricas de la literatura sobre el detector GNN, reproducible con un
solo comando sobre el dataset sintético **o** sobre los eventos del benchmark **THEIA_E3**
(DARPA Transparent Computing, escenario E3, kernel 3.15/4.4/4.19 de 64 bits).

### Snapshot con validación limpia

- `_build_pickle(events_dir, out, window=5.0)` conserva **todas** las ventanas temporales
  acumulativas de cada run (antes solo se retenía la última).
- `GraphTensor.run_id` etiqueta cada ventana con su run; las métricas de run se agrupan por
  `run_id` (pooling por nodo: score máx. y etiqueta any-malicioso).
- Split **train/val/test por run** (`split_by_run`, 60/20/20): el autoencoder se entrena solo con
  runs normales, el umbral se calibra en el conjunto de **validación** y el modelo jamás evalúa con
  datos de entrenamiento (se eliminó el antiguo escape `train_graphs[-1]`).

### Métricas de paper

`run_level_evaluate` reporta, sobre la concatenación de los nodos agrupados por run:

- AUC ROC + precision@1 %, recall @1 % y @5 %, nDCG@5 %.
- FPR@presupuesto (1/5/10), es decir, el coste en falsos positivos por nodo inspeccionado
  necesario para capturar a todos los atacantes.
- Informe de umbral (TP/FP/FN/TN, TPR, FPR) y recall por run.

### Baselines y ablación

`neurobpf/gnn/baselines.py` (solo features, sin topología):

- **OCSVM** (`nu=0.1`) — anomalía con kernel sobre los descriptores de nodo.
- **MLP autoencoder** — reconstrucción con cuello de botella (error MSE como score).

`neurobpf/gnn/ablation.py` — **ablación de topología**: permutación determinista de las aristas
(`shuffled_graphs`). Cuantificar cuánto aporta el grafo de procedencia frente a solo features.

### Reproducción

```bash
# Tabla multi-seed (por defecto 5 seeds): GNN vs GNN-shuffled vs OCSVM vs MLP-autoencoder
python -m neurobpf.cli compare --graphs data/graphs.pkl --out data/report.json --seeds 5 --epochs 300

# Con el pipeline end-to-end sobre la demo sintética
python -m neurobpf.cli demo --out data --seed 7 --normal 6 --attack 3
```

### THEIA_E3 / DARPA TC

`neurobpf/benchmark/theia.py` convierte los registros JSONL de THEIA_E3 al esquema NDJSON del
pipeline (mismo formato que el colector eBPF y el motor sintético). Las etiquetas
(`theia_ground_truth_to_pids`) se traducen a `malicious_pids` por run para construir el dataset de
evaluación; la lectura del escenario queda fuera del paquete por tamaño (~12 GB).

## Repositorio

```
ebpf/            Colector eBPF en C (libbpf): file_read, file_write, net_connect, process_start
neurobpf/        Pipeline Python: eventos, construcción de grafos, escenarios, GNN, servidor
  events.py        esquema canónico + I/O NDJSON
  graphbuilder.py  snapshots de procedencia temporales
  scenarios.py     simulador normal + 5 escenarios de ataque
  cli.py           subcomandos generate / build / train / detect / annotate / demo / serve / compare
  gnn/             features, model (GAE), dataset, train, detect, baselines, ablation
  benchmark/       adapter del benchmark THEIA_E3 (DARPA TC)
  server.py        FastAPI + WebSocket de replay
web/             Frontend Vite + React: grafo 2D/3D (react-force-graph) en vivo
tests/           Suite pytest del pipeline completo
```

## Requisitos

- Linux sin contenedores (eBPF), Python 3.10+ (recomendado 3.12+).
- Para la captura real: headers de libbpf y permisos root (puede requerir BTF de kernel).

## Puesta en marcha

```bash
# 1) Entorno
python -m venv .venv
source .venv/bin/activate
pip install -e .

# 2) Tests
pytest

# 3) Demo sintética: genera, entrena, detecta y anota
python -m neurobpf.cli demo --out data --seed 7 --normal 6 --attack 3
#   Imprime: AUC 0.90 | recall@k 0.71
#   Deja en data/: graphs.pkl, model.pt, detection.json, annotations/*.json

# 4) Demo con visualizador en vivo (puerto 8899, sirve web/dist)
python -m neurobpf.cli demo --out data --seed 7 --normal 6 --attack 3 --serve

# Alternativa: servir una demo ya generada
python -m neurobpf.cli serve --annotated data/annotations --web web/dist --port 8899
```

## Captura real con eBPF

```bash
make -C ebpf                          # compila el colector (requiere clang + libbpf)

mkdir -p /tmp/live
sudo ./ebpf/collector > /tmp/live/live.events.ndjson   # traza syscalls ~25s
# Ctrl+C para detener

# Construye el grafo a partir de la captura real
python -m neurobpf.cli build --events /tmp/live --out data/live.pkl

# Presupone un modelo entrenado (paso 3): puntúa la captura real
python -m neurobpf.cli annotate --graphs data/live.pkl --model data/model.pt --out data/live_annotations
python -m neurobpf.cli serve --annotated data/live_annotations --web web/dist
```

## Frontend

```bash
cd web
npm install
npm run build      # genera web/dist
```

El visualizador conecta por WebSocket (mismo origen, `/ws/snapshots`); puedes apuntar a otro servidor con `VITE_WS_URL`.

## Escenarios de ataque simulados

1. **Code injection** — proceso legítimo ejecuta un binario inyectado.
2. **Exfiltration** — lectura de archivos sensibles + conexión de red sospechosa.
3. **Lateral movement** — cadena de procesos entre hosts.
4. **Persistence** — tareas de arranque manipuladas.
5. **Supply chain** — ejecutable externo sustituido en el `PATH`.