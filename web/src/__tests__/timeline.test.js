import { describe, expect, test } from 'vitest';
import {
  buildTimeline,
  methodColor,
  methodRows,
  metricLabels,
  runSeverity,
  snapshotStats,
  timelineBounds,
} from '../timeline.js';

describe('snapshotStats', () => {
  test('counts nodes, edges, anomalies and max score', () => {
    const snap = {
      ts: 2.5,
      nodes: [
        { id: 'a', score: 0.2, anomalous: false },
        { id: 'b', score: 0.9, anomalous: true },
        { id: 'c' },
      ],
      edges: [{ src: 'a', dst: 'b' }],
    };
    expect(snapshotStats(snap)).toEqual({
      ts: 2.5,
      nNodes: 3,
      nEdges: 1,
      nAnomalous: 1,
      maxScore: 0.9,
    });
  });
});

describe('buildTimeline', () => {
  test('falls back to computing stats from snapshots', () => {
    const runDetail = {
      snapshots: [
        { ts: 1, nodes: [{ id: 'a', score: 0.4, anomalous: false }], edges: [] },
        { ts: 2, nodes: [{ id: 'a', score: 1.0, anomalous: true }], edges: [{ src: 'a', dst: 'b' }] },
      ],
    };
    const points = buildTimeline(runDetail);
    expect(points).toHaveLength(2);
    expect(points[0]).toMatchObject({ index: 0, ts: 1, nNodes: 1, nAnomalous: 0, maxScore: 0.4 });
    expect(points[1].maxScore).toBe(1.0);
    expect(points[1].nAnomalous).toBe(1);
  });

  test('uses server-provided stats when present', () => {
    const runDetail = {
      snapshots: [{ ts: 1, nodes: [] }, { ts: 2, nodes: [] }],
      stats: [
        { ts: 1, n_nodes: 5, n_edges: 4, n_anomalous: 0, max_score: 0.1 },
        { ts: 2, n_nodes: 6, n_edges: 5, n_anomalous: 2, max_score: 0.8 },
      ],
    };
    const points = buildTimeline(runDetail);
    expect(points[1]).toMatchObject({ index: 1, ts: 2, nNodes: 6, nEdges: 5, nAnomalous: 2, maxScore: 0.8 });
    expect(points[0].nNodes).toBe(5);
  });
});

describe('timelineBounds', () => {
  test('guards empty input and zero scores', () => {
    expect(timelineBounds([]).maxScore).toBeGreaterThan(0);
    expect(timelineBounds([{ maxScore: 0 }])).toEqual({ maxScore: 1e-6, maxAnom: 0 });
  });
});

describe('runSeverity', () => {
  test('ok when tpr high and fpr low', () => {
    expect(runSeverity({ threshold_report: { tpr: 0.8, fpr: 0.01 } })).toBe('ok');
    expect(runSeverity({ threshold_report: { tpr: 0.2, fpr: 0.3 } })).toBe('warn');
    expect(runSeverity({})).toBe('unknown');
  });
});

describe('methodRows', () => {
  const report = {
    summary: {
      'gnn auc_roc': { mean: 0.79, std: 0.05 },
      'gnn recall_at_5pct': { mean: 0.4, std: 0.1 },
      'ocsvm auc_roc': { mean: 0.55, std: 0.02 },
    },
  };
  test('splits method and metric keys', () => {
    const rows = methodRows(report);
    expect(rows).toContainEqual({ method: 'gnn', metric: 'auc_roc', mean: 0.79, std: 0.05 });
    expect(rows).toContainEqual({ method: 'ocsvm', metric: 'auc_roc', mean: 0.55, std: 0.02 });
  });
  test('collects distinct metrics and colors', () => {
    expect(metricLabels(methodRows(report)).sort()).toEqual(['auc_roc', 'recall_at_5pct']);
    expect(methodColor('gnn')).toBe('#22d3ee');
    expect(methodColor('unknown')).toBe('#818cf8');
  });
});