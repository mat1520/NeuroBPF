import { useEffect, useMemo, useRef, useState } from 'react';
import { connectSnapshots } from './ws.js';
import GraphView from './GraphView.jsx';

const WS_PATH = import.meta.env.VITE_WS_URL || (() => {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${proto}//${window.location.host}/ws/snapshots`;
})();

const SPEEDS = [1, 2, 4];

const TYPE_LABEL = {
  attack: 'ataque',
  normal: 'normal',
  real: 'real',
};

export default function App() {
  const [frame, setFrame] = useState(null);
  const [mode, setMode] = useState('2d');
  const [live, setLive] = useState(false);
  const [paused, setPaused] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [selected, setSelected] = useState(null);
  const cursorRef = useRef();
  const pausedRef = useRef(false);
  const speedRef = useRef(1);

  useEffect(() => {
    pausedRef.current = paused;
  }, [paused]);
  useEffect(() => {
    speedRef.current = speed;
  }, [speed]);

  useEffect(() => {
    let cancelled = false;
    let cursor;

    (async () => {
      try {
        cursor = connectSnapshots(WS_PATH);
        cursorRef.current = cursor;
        setLive(true);
        while (!cancelled) {
          const { value, done } = await cursor.next();
          if (done || cancelled) break;
          if (pausedRef.current) continue;
          let latest = value;
          for (let i = 1; i < speedRef.current; i += 1) {
            const { value: v, done: d } = await cursor.next();
            if (d || cancelled) break;
            latest = v;
          }
          setFrame(latest);
        }
      } catch {
        setLive(false);
      }
    })();

    return () => {
      cancelled = true;
      if (cursor) cursor.close();
    };
  }, []);

  const stats = useMemo(() => {
    if (!frame) return { nodes: 0, edges: 0, anomalies: 0 };
    const nodes = frame.nodes || [];
    return {
      nodes: nodes.length,
      edges: (frame.edges || []).length,
      anomalies: nodes.filter((n) => n.anomalous).length,
    };
  }, [frame]);

  const runId = frame?.run_id ?? null;
  const runType = frame?.run_type ?? null;
  const snapIndex = frame?.snapshot_index;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <svg className="brand-mark" viewBox="0 0 24 24" aria-hidden="true">
            <path
              d="M12 2 L20 6 V14 L12 22 L4 14 V6 Z M12 7 L16 9.5 V14.5 L12 17 L8 14.5 V9.5 Z"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.4"
            />
            <circle cx="12" cy="12" r="1.6" fill="currentColor" />
          </svg>
          <div className="brand-text">
            <h1>Neuro<b>BPF</b></h1>
            <span className="brand-sub">kernel anomaly detection</span>
          </div>
        </div>

        <div className="run-meta" role="status" aria-live="polite">
          <span className={`run-badge ${runType || 'idle'}`}>
            {runId ? `${runId.split('_').pop()}·${runId.split('_')[1] || 'run'}` : '—'}
          </span>
          <span className="run-type">{runType ? TYPE_LABEL[runType] || runType : 'esperando datos'}</span>
          {snapIndex !== undefined && (
            <span className="snap-progress">{snapIndex + 1}/{frame.snap_total}</span>
          )}
        </div>

        <div className="stats" aria-label="métricas del snapshot">
          <Stat value={stats.nodes} label="nodos" />
          <Stat value={stats.edges} label="aristas" />
          <Stat value={stats.anomalies} label="anómalos" tone={stats.anomalies > 0 ? 'danger' : 'ok'} />
        </div>

        <div className="controls" aria-label="controles de reproducción">
          <button
            className="ctrl-btn"
            onClick={() => setPaused((p) => !p)}
            aria-label={paused ? 'reanudar' : 'pausar'}
            title={paused ? 'reanudar' : 'pausar'}
          >
            {paused ? (
              <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7z" fill="currentColor" /></svg>
            ) : (
              <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 5h4v14H6zM14 5h4v14h-4z" fill="currentColor" /></svg>
            )}
          </button>
          <div className="seg" role="group" aria-label="velocidad de reproducción">
            {SPEEDS.map((s) => (
              <button
                key={s}
                className={`seg-btn ${speed === s ? 'active' : ''}`}
                onClick={() => setSpeed(s)}
                aria-pressed={speed === s}
              >
                {s}×
              </button>
            ))}
          </div>
          <div className="seg" role="group" aria-label="modo de visualización">
            <button
              className={`seg-btn ${mode === '2d' ? 'active' : ''}`}
              onClick={() => setMode('2d')}
              aria-pressed={mode === '2d'}
            >
              2D
            </button>
            <button
              className={`seg-btn ${mode === '3d' ? 'active' : ''}`}
              onClick={() => setMode('3d')}
              aria-pressed={mode === '3d'}
            >
              3D
            </button>
          </div>
        </div>

        <div className={`status-pill ${live ? 'live' : 'off'}`}>
          <span className="status-dot" aria-hidden="true" />
          {live ? 'LIVE' : 'OFFLINE'}
        </div>
      </header>

      <main className="workspace">
        <section className="graph-stage" aria-label="grafo de procedencia">
          <GraphView frame={frame} mode={mode} onSelect={setSelected} selected={selected} />
          {!frame && (
            <div className="empty-state">
              <div className="empty-glyph" aria-hidden="true">
                <span />
                <span />
                <span />
              </div>
              <p>esperando snapshots del kernel...</p>
            </div>
          )}
        </section>

        <aside className="side-panel">
          <Legend />
          <Details node={selected} />
        </aside>
      </main>
    </div>
  );
}

function Stat({ value, label, tone }) {
  return (
    <div className="stat">
      <span className={`stat-value ${tone === 'danger' ? 'danger' : ''}`}>{value}</span>
      <span className="stat-label">{label}</span>
    </div>
  );
}

function Legend() {
  return (
    <section className="panel" aria-label="leyenda">
      <h2>Leyenda</h2>
      <div className="legend-block">
        <span className="legend-title">Nodos</span>
        <ul className="legend-list">
          <li><span className="node-glyph kind-process" aria-hidden="true" />proceso</li>
          <li><span className="node-glyph kind-file" aria-hidden="true" />archivo</li>
          <li><span className="node-glyph kind-net" aria-hidden="true" />red</li>
        </ul>
      </div>
      <div className="legend-block">
        <span className="legend-title">Aristas</span>
        <ul className="legend-list">
          <li><span className="edge-glyph exec" aria-hidden="true" />EXEC</li>
          <li><span className="edge-glyph read" aria-hidden="true" />READ</li>
          <li><span className="edge-glyph write" aria-hidden="true" />WRITE</li>
          <li><span className="edge-glyph connect" aria-hidden="true" />CONNECT</li>
          <li><span className="edge-glyph unlink" aria-hidden="true" />UNLINK</li>
        </ul>
      </div>
      <div className="legend-block">
        <span className="legend-title">Anomalía</span>
        <ul className="legend-list">
          <li><span className="node-glyph kind-anom" aria-hidden="true" />nodo anómalo</li>
        </ul>
      </div>
    </section>
  );
}

function Details({ node }) {
  return (
    <section className="panel" aria-label="detalle del nodo">
      <h2>Detalle</h2>
      {!node ? (
        <p className="panel-hint">haz clic en un nodo del grafo</p>
      ) : (
        <dl className="detail-list">
          <dt>id</dt>
          <dd className="mono">{node.id}</dd>
          <dt>etiqueta</dt>
          <dd>{node.label || node.id}</dd>
          <dt>tipo</dt>
          <dd>{node.kind || 'desconocido'}</dd>
          <dt>score anomalía</dt>
          <dd className="mono">{typeof node.score === 'number' ? node.score.toFixed(3) : '—'}</dd>
          <dt>estado</dt>
          <dd className={node.anomalous ? 'anomalous-text' : 'ok-text'}>
            {node.anomalous ? 'anómalo' : 'normal'}
          </dd>
        </dl>
      )}
    </section>
  );
}