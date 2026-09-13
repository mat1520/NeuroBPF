# NeuroBPF — Diseño del Pipeline de Investigación

## Problema actual

El pipeline funciona end-toendo (eBPF → grafos → GNN → visualización), pero tiene 4 problemas que impiden usarlo como prototipo de paper:

1. **Colapso temporal**: `_build_pickle` descarta ventanas temporales y usa solo la última snapshot acumulativa. La GNN nunca ve evolución temporal.
2. **Sin train/test split**: validación en `train_graphs[-1]` (mismo que entrena), threshold calibrado en training, AUC incluye datos de entrenamiento.
3. **Métricas no reflejan detección**: `evaluate()` ignora el `threshold` — AUC/recall@k son independientes del umbral. El flag `anomalous` usa z-score vs threshold pero las métricas no lo miden.
4. **No hay baselines**: no se puede comparar contra métodos simples (OCSVM, MLP-AE) para demostrar valor de la topología.

## Diseño propuesto

### 1. Pipeline temporal con train/val/test split

**Split por runs** (no por nodos):
- 60% train (solo normales), 20% val (normales), 20% test (mixto: normales + ataques)
- Stratificado por tipo de ataque en test
- Los runs de test jamás se tocan durante entrenamiento

**Temporal por ventanas**:
- Cada run genera múltiples grafos (uno por ventana de 5s acumulativa)
- Un run de 60s → 12 grafos. Un run de 12s → 2-3 grafos
- La GNN entrena/val/testea sobre grafos individuales, no sobre runs

**Early stopping real**:
- Validación en `val_graphs` (nunca en train)
- Patience sobre val loss

**Threshold en val**:
- Calibrar z-scores con `val_graphs`, no con train
- Usar percentil 99 de z-scores de val

### 2. Sparse CSR + escalabilidad

Reemplazar `make_adj` (densa O(n²)) por sparse CSR:

```python
# dataset.py
def make_sparse_adj(g: GraphTensor) -> csr_matrix:
    n = g.x.shape[0]
    src, dst = g.edge_index
    data = np.ones(len(src), dtype=np.float32)
    adj = csr_matrix((data, (src.numpy(), dst.numpy())), shape=(n, n))
    adj = adj + adj.T  # simetrizar
    adj.data[:] = 1.0  # binarizar
    return adj
```

**Node scores sparse**:
- Propagación: `h = adj @ lin(x)` → sparse × dense = dense (funciona con PyTorch sparse)
- Reconstruction: `z @ z.T` sigue siendo densa pero solo para grafo individual
- BCE por nodo: fila de la matriz de reconstrucción

**Benchmark THEIA_E3** (~12GB, Linux, ~1-5k nodos):
- Grafo completo de THEIA cabe en sparse (1k nodos → 1M entries sparse vs 1M entries dense — pero sparse propaga en O(nnz·d) vs O(n²·d))
- Adapter: convertir formato DARPA TC → NDJSON → graphbuilder

### 3. Baselines y ablaciones

**Baselines** (features-only, sin topología):
- **OCSVM**: One-Class SVM sobre vector de features promedio por nodo
- **MLP-AE**: Autoencoder simple sobre features, anomaly score = reconstruction error
- **détach**: Reconstrucción de atributos sin topología (feat only)

**Ablaciones**:
- GNN-con-topology vs GNN-sin-topology (shuffle adjacency)
- Pooling: mean vs max sobre embeddings de nodos vecinos
- Temporal: con ventanas vs sin ventanas (último snapshot)

### 4. Métricas de paper

Implementar en `detect.py`:

| Métrica | Descripción | Uso |
|---------|-------------|-----|
| ROC-AUC | Área bajo curva ROC | Métrica principal |
| Recall@k | Top-k más scoreados que son malicious | k = top-1%, top-5% |
| nDCG | Normalized Discounted Cumulative Gain | Calidad de ranking |
| FPR@budget | FPR cuando se emiten N alerts | budgets: 1, 5, 10 alerts |
| Precision@k | De los top-k, cuántos son malicious | Reportar junto con recall |

**Evaluación**:
- Por-seed con desviación estándar (5 seeds mínimo)
- Curvas de rendimiento por tipo de ataque
- Tabla comparativa: GNN vs baselines vs ablaciones

### 5. Benchmark adapter (THEIA_E3)

**Formato de entrada**:
- THEIA_E3 viene como eventos comprimidos (tar.gz)
- PIDSMaker provee ground truth en CSV
- Adapter convierte: DARPA event → `Event` dataclass → NDJSON

**Extracción de subgrafos**:
- Grafo completo de THEIA puede tener >10k nodos
- Extraer subgrafos por ventana temporal (misma métrica que sintético)
- Mantener proporción train/val/test

**Ground truth**:
- Usar PIDSMaker's `orthrus` labels (conservador, másPreciso)
- Mapear PIDs a nodos de proceso
- Propagar a archivos/net como hace nuestro graphbuilder

## Archivos a modificar

| Archivo | Cambio |
|---------|--------|
| `neurobpf/gnn/dataset.py` | `make_sparse_adj`, `split_graphs`, `normalize_x` real |
| `neurobpf/gnn/train.py` | Early stopping en val, threshold en val |
| `neurobpf/gnn/detect.py` | Métricas: nDCG, FPR@budget, precision@k. Usar threshold en evaluate |
| `neurobpf/gnn/features.py` | Fix `is_root` bug para file/net nodes |
| `neurobpf/gnn/model.py` | Soporte sparse adj en forward/train_loss/node_scores |
| `neurobpf/gnn/baselines.py` | **Nuevo**: OCSVM, MLP-AE, détach |
| `neurobpf/gnn/ablation.py` | **Nuevo**: GNN-sin-topology, shuffle adj |
| `neurobpf/benchmark/` | **Nuevo**: adapter THEIA_E3 → NDJSON |
| `neurobpf/cli.py` | Subcomandos: `split`, `benchmark`, `compare` |
| `tests/test_*.py` | Tests para cada cambio |

## Orden de implementación

1. Fix `is_root` bug + test
2. Sparse CSR + test
3. Train/val/test split + test
4. Temporal windowing (usar todas las snapshots) + test
5. Early stopping real + threshold en val + test
6. Métricas paper (nDCG, FPR@budget, precision@k) + test
7. Baselines (OCSVM, MLP-AE) + test
8. Ablaciones + test
9. Benchmark adapter THEIA_E3 + test
10. CLI subcomandos + test
11. Suite completa + demo + commit
