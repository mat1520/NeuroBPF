import { useEffect, useMemo, useRef, useState } from 'react';
import { fetchExperiment, fetchReport, fetchRun, fetchRuns } from './api.js';
import GraphView from './GraphView.jsx';
import KpiHeader from './components/KpiHeader.jsx';
import RunsTable from './components/RunsTable.jsx';
import Timeline from './components/Timeline.jsx';
import BaselinesBar from './components/BaselinesBar.jsx';
import NodeDetail from './components/NodeDetail.jsx';
import { buildTimeline, runTypeLabel } from './timeline.js';

export default function App() {
  const [experiment, setExperiment] = useState(null);
  const [runs, setRuns] = useState(null);
  const [report, setReport] = useState(null);
  const [runDetail, setRunDetail] = useState(null);
  const [selectedRunId, setSelectedRunId] = useState(null);
  const [snapshotIndex, setSnapshotIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [mode, setMode] = useState('2d');
  const [search, setSearch] = useState('');
  const [onlyAnomalies, setOnlyAnomalies] = useState(false);
  const [selected, setSelected] = useState(null);
  const playRef = useRef(false);

  useEffect(() => {
    playRef.current = playing;
  }, [playing]);

  useEffect(() => {
    let cancelled = false;
    Promise.all([fetchExperiment(), fetchRuns(), fetchReport()]).then(([exp, runList, rep]) => {
      if (cancelled) return;
      setExperiment(exp);
      const rows = runList?.runs || [];
      setRuns(rows);
      setReport(rep);
      setSelectedRunId((prev) => prev ?? rows[0]?.id ?? null);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!selectedRunId) return undefined;
    let cancelled = false;
    fetchRun(selectedRunId).then((detail) => {
      if (cancelled) return;
      setRunDetail(detail);
      setSnapshotIndex(0);
      setSelected(null);
    });
    return () => {
      cancelled = true;
    };
  }, [selectedRunId]);

  useEffect(() => {
    if (!playing || !runDetail) return undefined;
    const id = window.setInterval(() => {
      setSnapshotIndex((i) => {
        const next = i + 1;
        if (next >= runDetail.snapshots.length) {
          setPlaying(false);
          return 0;
        }
        return next;
      });
    }, 350);
    return () => window.clearInterval(id);
  }, [playing, runDetail]);

  const points = useMemo(() => (runDetail ? buildTimeline(runDetail) : []), [runDetail]);
  const frame = useMemo(() => runDetail?.snapshots?.[snapshotIndex] ?? null, [runDetail, snapshotIndex]);
  const count = useMemo(() => runDetail?.snapshots?.length ?? 0, [runDetail]);

  return (
    <div className="app">
      <KpiHeader experiment={experiment} />

      <main className="workspace">
        <aside className="sidebar">
          <section className="panel">
            <h2>Runs</h2>
            <RunsTable
              runs={runs}
              selectedId={selectedRunId}
              onSelect={(id) => {
                setSelectedRunId(id);
                setPlaying(false);
              }}
            />
          </section>
          <section className="panel">
            <h2>Baselines</h2>
            <BaselinesBar report={report} />
          </section>
        </aside>

        <section className="main-stage">
          <section className="panel timeline-panel">
            <div className="panel-row">
              <h2>
                {runDetail ? (
                  <>
                    <span className="mono">{runDetail.run_id}</span>
                    <span className={`run-badge ${runDetail.run_type}`}>{runTypeLabel(runDetail.run_type)}</span>
                  </>
                ) : (
                  'Timeline'
                )}
              </h2>
              <div className="seg" role="group" aria-label="controles de reproducción">
                <button
                  type="button"
                  className="seg-btn"
                  onClick={() => setPlaying((p) => !p)}
                  aria-label={playing ? 'pausar' : 'reproducir'}
                  disabled={!runDetail}
                >
                  {playing ? '❚❚' : '▶'}
                </button>
                <button
                  type="button"
                  className="seg-btn"
                  onClick={() => setPlaying(false)}
                  disabled={!runDetail}
                  aria-label="reiniciar"
                >
                  ⟲
                </button>
              </div>
              <span className="snap-progress">
                {count ? `${snapshotIndex + 1}/${count}` : ''}
              </span>
            </div>
            <Timeline points={points} selectedIndex={snapshotIndex} onScrub={setSnapshotIndex} />
          </section>

          <section className="panel graph-panel">
            <div className="panel-row">
              <div className="graph-controls">
                <input
                  className="search"
                  type="search"
                  value={search}
                  onChange={(evt) => setSearch(evt.target.value)}
                  placeholder="buscar nodo..."
                  aria-label="buscar nodo"
                />
                <label className="toggle">
                  <input
                    type="checkbox"
                    checked={onlyAnomalies}
                    onChange={(evt) => setOnlyAnomalies(evt.target.checked)}
                  />
                  solo anómalos
                </label>
              </div>
              <div className="seg" role="group" aria-label="modo de visualización">
                <button
                  type="button"
                  className={`seg-btn ${mode === '2d' ? 'active' : ''}`}
                  onClick={() => setMode('2d')}
                  aria-pressed={mode === '2d'}
                >
                  2D
                </button>
                <button
                  type="button"
                  className={`seg-btn ${mode === '3d' ? 'active' : ''}`}
                  onClick={() => setMode('3d')}
                  aria-pressed={mode === '3d'}
                >
                  3D
                </button>
              </div>
            </div>
            <div className="graph-frame">
              <GraphView
                frame={frame}
                mode={mode}
                selected={selected}
                onSelect={setSelected}
                onlyAnomalies={onlyAnomalies}
                searchText={search}
              />
              {!frame && <div className="empty-state"><p>selecciona un run para inspeccionar</p></div>}
            </div>
          </section>
        </section>

        <aside className="right-panel">
          <section className="panel">
            <h2>Detalle</h2>
            <NodeDetail frame={frame} node={selected} />
          </section>
        </aside>
      </main>
    </div>
  );
}