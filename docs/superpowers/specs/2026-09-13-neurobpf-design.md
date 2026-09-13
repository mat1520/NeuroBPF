# NeuroBPF — Diseño

Detección de Intrusiones en el Kernel con eBPF y Redes Neuronales de Grafos (GNN).

## 1. Propósito

Proyecto de portafolio / demostración técnica que combina:

- Captura de eventos del kernel en tiempo real con **eBPF** (C + libbpf, sin ralentizar el SO).
- Construcción de un **grafo de procedencia temporal** (procesos, archivos, conexiones de red).
- Detección de anomalías con una **GNN** tipo *autoencoder de reconstrucción*, entrenada solo con comportamiento normal; detecta ataques **desconocidos**.
- **Visualización 2D/3D** del grafo que resalta la "ramificación anómala" de un proceso infectado (pieza visual para LinkedIn).

Objetivo medible: sobre escenarios sintéticos etiquetados de ataque, la GNN debe obtener AUC-ROC ≥ 0.80 a nivel de nodo con modelos entrenados exclusivamente con ventanas normales.

## 2. Arquitectura

```
┌────────────────────────┐   ┌───────────────────────────────┐
│ Scenario Engine (Py)   │   │ eBPF Collector (C + libbpf)    │
│ flujos NDJSON          │   │ tracepoints/kprobes → NDJSON   │
│ etiquetados            │   └───────────────┬───────────────┘
└───────────┬────────────┘                   │
            ▼                                ▼
      formato canónico NDJSON de eventos (compartido)
            ▼
┌─────────────────────────────────────────────────────────────┐
│ Graph Builder (Py)  eventos → grafo de procedencia temporal  │
│ nodos: proceso | archivo | red  aristas: exec/read/write/... │
└───────────────────────────────┬─────────────────────────────┘
                                ▼
       ┌────────────────────────────────────────────┐
       │ GNN Detector (PyTorch, CPU)                  │
       │ autoencoder de grafos entrenado SOLO con     │
       │ ventanas normales; error de reconstrucción   │
       │ por nodo → score de anomalía                 │
       └────────────────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────┐
│ Live Visualizer (React + react-force-graph, 2D/3D)           │
│ flujo desde servidor FastAPI + WebSocket;                  │
│ ramas anómalas resaltadas; modo replay para demo/GIF        │
└─────────────────────────────────────────────────────────────┘
```

## 3. Subsistemas y orden de construcción

Cada subsistema es un entregable demo-able e independiente:

1. **Scenario Engine + Graph Builder** — fundación; genera eventos NDJSON etiquetados y produce grafo de procedencia. Define el esquema canónico.
2. **GNN Detector** — autoencoder de grafos entrenado solo con datos normales; evaluado con los 5 escenarios de ataque.
3. **Live Visualizer** — animación 2D/3D del grafo de procedencia con anomalías resaltadas; usa datos sintéticos sin necesidad de kernel.
4. **eBPF Collector** — captura real en este host (sudo) emitiendo el MISMO formato NDJSON para alimentar el mismo pipeline.

## 4. Esquema canónico de eventos (NDJSON, 1 línea = 1 evento)

```json
{
  "ts": 1700000000.123,
  "event": "process_start",
  "pid": 2001,
  "ppid": 1000,
  "comm": "bash",
  "exe_path": "/usr/bin/bash",
  "uid": 1000,
  "target_type": null,
  "target": null,
  "result": 0
}
```

Tipos de evento (`event`):
- `process_start` (pid, ppid, comm, exe_path) — arista **EXEC** ppid→pid, crea nodo proceso.
- `process_exit` (pid) — marca fin del proceso.
- `file_read` / `file_write` (path) — arista **READ/WRITE** pid→nodo archivo.
- `file_unlink` (path) — arista **UNLINK** pid→archivo.
- `file_rename` (old, new) — arista **UNLINK** pid→old y **WRITE** pid→new.
- `net_connect` (addr, port) — arista **CONNECT** pid→nodo red(addr:port) (saliente).
- `net_accept` (addr, port) — arista **ACCEPT** pid→nodo red (entrante).

Reglas:
- `ts` flotante en segundos, creciente en el stream.
- `pid` identifica el nodo de proceso; `ppid` la arista EXEC.
- El grafo vive en ventanas temporales discretas (más abajo).

## 5. Grafo de procedencia y ventanas

- Nodos: `proceso` (pid, comm, exe_path), `archivo` (path), `red` ("ip:port").
- Aristas dirigidas: EXEC, READ, WRITE, UNLINK, CONNECT, ACCEPT (cada tipo = etiqueta).
- **Ventana temporal**: W = 5 s. Cada ventana produce un *snapshot* del grafo. Para contexto temporal, la GNN usa el snapshot de la ventana actual (los grafos pequeños se resuelven con features de nodo y estructura).
- Normalización de features de nodo dentro de cada snapshot: tipo (one-hot 3), grado in/out (normalizados), contador de eventos por tipo (normalizado), uid (binarizado 0=xid=0 u otro). F total ≈ 12.

## 6. GNN: autoencoder de reconstrucción (detección de anomalías)

- **Encoder**: GCN de 2 capas (dim. oculta 64, activación ReLU), concat embed desde features + estructura → z_i ∈ R^64.
- **Decoder**: producto interno — reconstruye adyacencia `A_hat_ij = sigmoid(z_i · z_j)`.
- **Loss de entrenamiento** (solo grafos NORMALES): BCE sobre aristas existentes + pares negativos muestreados.
- **Score de anomalía por nodo**: error de reconstrucción BCE de sus aristas incidentes (ejecutar el modelo "con" el nodo dado sus features). Score de subgrafo = media.
- **Umbral**: percentil 99 de la distribución de scores normales de entrenamiento.
- Nodo con score > umbral → anómalo. Top-k se resaltan en la visualización.
- Entrenamiento: PyTorch en CPU (GPU no disponible por mismatch de driver). Snapshot de ~20–100 nodos; baterías de escenarios normales → cientos de snapshots.
- Métricas de evaluación: AUC-ROC por nodo sobre test, precisión@k, y comparación de detección por escenario.

## 7. Scenario Engine (5 escenarios de ataque + tráfico normal)

Genera streams NDJSON etiquetados (cada corrida = un archivo de eventos + un archivo de etiquetas). El motor ejecuta "guiones" de comportamiento con parámetros aleatorios.

**Tráfico normal** (baseline de entrenamiento): procesos de sistema y usuario corriendo tareas benignas — shell + herramientas frecuentes leyendo/escribiendo config, logs, compilación, uso de red hacia puertos estándar (HTTPS/DNS/SSH).

Escenarios (cada ataque se inyecta sobre un fondo normal; los pids del ataque se etiquetan como `malicious`):

1. **Code injection**: binario legítimo (p. ej. python/server) → fork + exec de shell; el shell escribe y ejecuta un hijo extraño (p. ej. `/tmp/.payload`) con perms raros. Cadena: A exec→ B(shell), B write→ /tmp/.x, B exec→ C(payload), C connect→ red.
2. **Persistence**: proceso escribe en crontab/`.bashrc`/`/etc/systemd/system/x.service` y crea un archivo binario oculto con perms 0777 en /tmp.
3. **Exfiltration beacon**: proceso normal de repente abre ráfaga de conexiones salientes a IPs nuevas en puertos altos (comportamiento de beaconing).
4. **Lateral movement**: ssh/scp hacia IP interna + nuevo árbol de proceso "remoto" activo (varias conexiones entrantes ACCEPT sobre una IP), escritura de clave SSH.
5. **Supply-chain**: gestor de paquetes (simulado) descarga de un repositorio "falso" y ejecuta el paquete descargado tras escribirlo.

Cada escenario incluye variantes (parámetros aleatorios) para que la GNN no memorice un solo patrón. Las etiquetas se generan con el grafo (por constructo, no por firma).

## 8. eBPF Collector

- C + **libbpf** (skeleton generado con `bpftool btf dump`/CO-RE). Fuentes `ebpf/collector.bpf.c` + `ebpf/collector.c` + Makefile (clang).
- Hooks: `tracepoint/syscalls` o kprobes/fentry sobre `execve`, `openat`, `write`, `connect`, `accept`, `unlink`, `rename*` — emiten al espacio de usuario vía ring buffer (libbpf `ring_buffer`).
- El user-space escribe los eventos en el **mismo formato NDJSON canónico**.
- Requiere root: se ejecuta con `sudo` del usuario. Es la última pieza; el pipeline completo funciona sin él (con datos sintéticos).

## 9. Visualizer

- Servidor **FastAPI**: carga snapshots anotados (con scores de anomalía) y los sirve vía **WebSocket** en secuencia (replay).
- Frontend **React + Vite + react-force-graph**: modo 2D (d3-force) y 3D (three.js), toggle; animación laminada de snapshots; ramas/nodos con score alto se resaltan en rojo con pulso; panel con evento actual.
- `npm run build` debe pasar para que el entregable sea estático y portátil; el demo se corre con `npm run dev` apuntando al servidor local.

## 10. Stack

- Python 3.14 (host) + PyTorch 2.14 **CPU** (venv dedicado en repo, gestión con pip). NumPy. pytest para tests.
- Frontend: Node 26 + Vite + React + react-force-graph.
- eBPF: C + libbpf + clang 22 + Makefile. Host: CachyOS kernel 7.2, BTF disponible.
- Sin GPU (driver NVIDIA en mismatch: `nvml` inválido) → todo en CPU.

## 11. Layout del repo

```
NeuroBPF/
├── README.md
├── pyproject.toml
├── neurobpf/
│   ├── __init__.py
│   ├── events.py          # dataclasses + IO NDJSON canónico
│   ├── scenarios.py       # scenario engine (normal + 5 ataques)
│   ├── graphbuilder.py    # eventos → snapshots de grafo
│   ├── gnn/
│   │   ├── __init__.py
│   │   ├── model.py       # GAE: encoder GCN + decoder producto interno
│   │   ├── train.py       # entrenar solo-normal, guardar modelo + umbral
│   │   └── detect.py      # scores por nodo, evaluación, anotación
│   ├── server.py          # FastAPI + WebSocket (replay)
│   └── cli.py             # CLI: generate / build / train / detect / serve
├── tests/                 # pytest para events/scenarios/graphbuilder/model/detect
├── ebpf/
│   ├── collector.bpf.c
│   ├── collector.c
│   ├── collector.h        # schema de evento compartido C
│   ├── Makefile
│   └── README.md
├── web/
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       ├── GraphView.jsx   # capa 2D/3D con react-force-graph
│       └── ws.js           # cliente WebSocket
└── data/                  # corridas generadas/snapshots/modelos (gitignored)
```

## 12. Restricciones globales

- **TDD obligatorio**: ningún `producción sin test que falle primero`. Cada subsistema con pruebas pytest (o build, en el caso del frontend/eBPF).
- Los módulos de código fuente se escriben **sin comentarios** salvo docstrings de uso de API cuando aporten claridad de interfaz.
- Idioma del código e identificadores: **inglés**. Documentación (README, docs): **español** (mercado/LinkedIn del autor).
- Version floor: torch ≥ 2.10, python ≥ 3.12 (host 3.14), react-force-graph ≥ 1.43.
- Nada de data real del host en el repo; `data/` en `.gitignore`.
- La GNN se entrega entrenable y evaluada sobre datos sintéticos (reproducible, `seed` fija).