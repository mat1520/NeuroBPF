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

## Repositorio

```
ebpf/            Colector eBPF en C (libbpf): file_read, file_write, net_connect, process_start
neurobpf/        Pipeline Python: eventos, construcción de grafos, escenarios, GNN, servidor
  events.py        esquema canónico + I/O NDJSON
  graphbuilder.py  snapshots de procedencia temporales
  scenarios.py     simulador normal + 5 escenarios de ataque
  cli.py           subcomandos generate / build / train / detect / annotate / demo / serve
  gnn/             features, model (GAE), dataset, train, detect
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