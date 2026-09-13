import { useState } from 'react';
import { methodColor, methodRows, metricLabels } from '../timeline.js';

export default function BaselinesBar({ report }) {
  const rows = methodRows(report);
  const metrics = metricLabels(rows);
  const [metric, setMetric] = useState(metrics[0] || 'auc_roc');
  if (metrics.length === 0) {
    return <p className="panel-hint">ejecuta <span className="mono">compare</span> para generar report.json</p>;
  }
  const group = rows.filter((r) => r.metric === metric).sort((a, b) => b.mean - a.mean);
  const max = Math.max(...group.map((r) => r.mean + r.std), 1e-6);

  return (
    <div className="baselines">
      <div className="seg" role="group" aria-label="métrica a comparar">
        {metrics.map((m) => (
          <button
            key={m}
            type="button"
            className={`seg-btn ${m === metric ? 'active' : ''}`}
            onClick={() => setMetric(m)}
            aria-pressed={m === metric}
          >
            {m}
          </button>
        ))}
      </div>
      <svg viewBox="0 0 440 150" role="img" aria-label={`comparativa de ${metric} entre métodos`}>
        {group.map((r, i) => {
          const y = 16 + i * 34;
          const w = (r.mean / max) * 340;
          return (
            <g key={r.method}>
              <text x={0} y={y + 4} className="bar-label">{r.method}</text>
              <rect x={0} y={y + 8} width={Math.max(2, w)} height="10" rx="2" fill={methodColor(r.method)} />
              <line
                x1={((r.mean - r.std) / max) * 340}
                y1={y + 6}
                x2={((r.mean + r.std) / max) * 340}
                y2={y + 6}
                className="bar-whisker"
              />
              <text x={w + 6} y={y + 17} className="bar-value">
                {r.mean.toFixed(2)} ± {r.std.toFixed(2)}
              </text>
            </g>
          );
        })}
      </svg>
      <div className="bar-foot">media ± desviación entre seeds</div>
    </div>
  );
}