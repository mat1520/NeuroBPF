import { useState } from 'react';
import { timelineBounds } from '../timeline.js';

const W = 760;
const H = 160;
const PAD = { top: 16, right: 16, bottom: 24, left: 40 };

export default function Timeline({ points, selectedIndex, onScrub }) {
  const [hoverIndex, setHoverIndex] = useState(null);
  if (!points || points.length === 0) return null;

  const { maxScore } = timelineBounds(points);
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;
  const x = (i) => PAD.left + (points.length === 1 ? plotW / 2 : (i / (points.length - 1)) * plotW);
  const yScore = (s) => PAD.top + plotH - (s / maxScore) * plotH * 0.85;
  const yAnom = () => PAD.top + 6;

  const nearRightEdge = (i, list) => x(i) + 90 > W - PAD.right;

  const xToIndex = (px) => {
    const t = Math.max(0, Math.min(1, (px - PAD.left) / plotW));
    return Math.round(t * (points.length - 1));
  };

  const line = points.map((p, i) => `${i === 0 ? 'M' : 'L'}${x(i).toFixed(1)},${yScore(p.maxScore).toFixed(1)}`).join(' ');
  const area = `${line} L${x(points.length - 1)},${PAD.top + plotH} L${x(0)},${PAD.top + plotH} Z`;
  const focus = hoverIndex ?? selectedIndex ?? 0;
  const focusPoint = points[focus];

  return (
    <div className="timeline" role="group" aria-label="línea temporal del run">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="slider"
        tabIndex={0}
        aria-valuemin={0}
        aria-valuemax={points.length - 1}
        aria-valuenow={selectedIndex ?? 0}
        aria-valuetext={`snapshot ${(selectedIndex ?? 0) + 1} de ${points.length}`}
        onKeyDown={(evt) => {
          if (evt.key === 'ArrowRight') onScrub(Math.min(points.length - 1, focus + 1));
          if (evt.key === 'ArrowLeft') onScrub(Math.max(0, focus - 1));
        }}
        onPointerDown={(evt) => {
          const rect = evt.currentTarget.getBoundingClientRect();
          const px = ((evt.clientX - rect.left) / rect.width) * W;
          onScrub(xToIndex(px));
        }}
        onPointerMove={(evt) => {
          const rect = evt.currentTarget.getBoundingClientRect();
          setHoverIndex(xToIndex(((evt.clientX - rect.left) / rect.width) * W));
        }}
        onPointerLeave={() => setHoverIndex(null)}
      >
        <line
          x1={PAD.left}
          y1={PAD.top + plotH}
          x2={PAD.left + plotW}
          y2={PAD.top + plotH}
          stroke="var(--color-border)"
        />
        <line
          x1={PAD.left}
          y1={PAD.top}
          x2={PAD.left}
          y2={PAD.top + plotH}
          stroke="var(--color-border)"
        />
        <text x={PAD.left - 8} y={PAD.top + 4} className="axis-label" textAnchor="end">
          score
        </text>

        <path d={area} className="timeline-area" aria-hidden="true" />
        <path d={line} className="timeline-line" aria-hidden="true" />
        {points.map((p, i) => (
          <g key={i}>
            <circle cx={x(i)} cy={yScore(p.maxScore)} r="2.5" className="timeline-dot" aria-hidden="true" />
            {p.nAnomalous > 0 && (
              <path
                d={`M${x(i)},${yAnom() + 6} L${x(i) + 4},${yAnom()} L${x(i)},${yAnom() - 6} L${x(i) - 4},${yAnom()} Z`}
                className="timeline-anom"
                aria-hidden="true"
              />
            )}
          </g>
        ))}
        <line x1={x(focus)} y1={PAD.top} x2={x(focus)} y2={PAD.top + plotH} className="timeline-cursor" aria-hidden="true" />
        {focusPoint && (
          <text
            x={nearRightEdge(focus, points, PAD, W) ? W - PAD.right : Math.min(W - PAD.right - 4, x(focus) + 4)}
            y={PAD.top - 2}
            textAnchor={nearRightEdge(focus, points, PAD, W) ? 'end' : 'start'}
            className="timeline-readout"
          >
            t={focusPoint.ts.toFixed(1)} · score {focusPoint.maxScore.toFixed(2)} · {focusPoint.nAnomalous} anómalos
          </text>
        )}
      </svg>
      <div className="timeline-legend" aria-hidden="true">
        <span><span className="swatch line" />score máx</span>
        <span><span className="swatch anom" />snapshot con anómalos</span>
        <span className="timeline-hint">arrastra · ← → para navegar</span>
      </div>
      <details className="timeline-table">
        <summary>datos del timeline</summary>
        <table>
          <thead>
            <tr>
              <th>snapshot</th>
              <th>t</th>
              <th>score máx</th>
              <th>anómalos</th>
              <th>nodos</th>
              <th>aristas</th>
            </tr>
          </thead>
          <tbody>
            {points.map((p) => (
              <tr key={p.index}>
                <td>{p.index + 1}</td>
                <td>{p.ts.toFixed(1)}</td>
                <td>{p.maxScore.toFixed(3)}</td>
                <td>{p.nAnomalous}</td>
                <td>{p.nNodes}</td>
                <td>{p.nEdges}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </div>
  );
}