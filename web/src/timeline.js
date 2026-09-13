export function snapshotStats(snap) {
  const nodes = snap?.nodes || [];
  const scores = nodes.filter((n) => typeof n.score === 'number').map((n) => n.score);
  return {
    ts: snap?.ts ?? 0,
    nNodes: nodes.length,
    nEdges: (snap?.edges || []).length,
    nAnomalous: nodes.filter((n) => n.anomalous).length,
    maxScore: scores.length ? Math.max(...scores) : 0,
  };
}

export function buildTimeline(runDetail) {
  const snapshots = runDetail?.snapshots || [];
  const serverStats = runDetail?.stats;
  return snapshots.map((snap, index) => {
    const stats = serverStats?.[index]
      ? {
          ts: serverStats[index].ts,
          nNodes: serverStats[index].n_nodes,
          nEdges: serverStats[index].n_edges,
          nAnomalous: serverStats[index].n_anomalous,
          maxScore: serverStats[index].max_score,
        }
      : snapshotStats(snap);
    return { index, ...stats };
  });
}

export function timelineBounds(points) {
  const toNum = (value) => (Number.isFinite(value) ? value : 0);
  const maxScore = Math.max(1e-6, ...points.map((p) => toNum(p.maxScore)));
  const maxAnom = Math.max(0, ...points.map((p) => toNum(p.nAnomalous)));
  return { maxScore, maxAnom };
}

export function runSeverity(experiment) {
  const rep = experiment?.threshold_report;
  if (!rep) return 'unknown';
  if (rep.tpr >= 0.5 && rep.fpr <= 0.05) return 'ok';
  return 'warn';
}

export function methodRows(report) {
  const summary = report?.summary || {};
  const rows = [];
  for (const [key, value] of Object.entries(summary)) {
    const space = key.lastIndexOf(' ');
    const method = space > 0 ? key.slice(0, space) : key;
    const metric = space > 0 ? key.slice(space + 1) : 'auc_roc';
    rows.push({
      method,
      metric,
      mean: value.mean,
      std: value.std,
    });
  }
  return rows.sort((a, b) => (a.method < b.method ? -1 : 1));
}

export function metricLabels(rows) {
  return [...new Set(rows.map((r) => r.metric))];
}

export function methodColor(method) {
  const palette = {
    gnn: '#22d3ee',
    gnn_shuffled: '#94a3b8',
    ocsvm: '#f472b6',
    mlp_ae: '#facc15',
  };
  return palette[method] || '#818cf8';
}

export function runTypeLabel(runType) {
  return { attack: 'ataque', normal: 'normal', real: 'real' }[runType] || runType;
}