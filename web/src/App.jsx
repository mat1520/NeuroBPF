import { useEffect, useState } from 'react';
import { connectSnapshots } from './ws.js';
import GraphView from './GraphView.jsx';

const WS_PATH = import.meta.env.VITE_WS_URL || (() => {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${proto}//${window.location.host}/ws/snapshots`;
})();

export default function App() {
  const [frame, setFrame] = useState(null);
  const [mode, setMode] = useState('2d');
  const [live, setLive] = useState(false);

  useEffect(() => {
    let cancelled = false;
    let cursor;

    (async () => {
      try {
        cursor = connectSnapshots(WS_URL);
        setLive(true);
        while (!cancelled) {
          const { value, done } = await cursor.next();
          if (done || cancelled) break;
          setFrame(value);
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

  const run = frame || {};

  return (
    <div className="app">
      <header className="app-header">
        <h1>NeuroBPF</h1>
        <div className="run-info">
          <span>{run.run_type || 'waiting'} run {run.run_id ?? '—'}</span>
          <span>snapshot {run.snapshot_index ?? '—'}/{run.snap_total ?? '—'}</span>
        </div>
        <div className="mode-toggle">
          <button className={mode === '2d' ? 'active' : ''} onClick={() => setMode('2d')}>
            2D
          </button>
          <button className={mode === '3d' ? 'active' : ''} onClick={() => setMode('3d')}>
            3D
          </button>
        </div>
        <span className={`status ${live ? 'on' : 'off'}`}>{live ? 'live' : 'offline'}</span>
      </header>
      <main className="graph-wrap">
        <GraphView frame={frame} mode={mode} />
      </main>
    </div>
  );
}