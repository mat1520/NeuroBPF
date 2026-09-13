import { runTypeLabel } from '../timeline.js';

export default function RunsTable({ runs, selectedId, onSelect }) {
  if (!runs || runs.length === 0) {
    return <p className="panel-hint">sin runs anotados</p>;
  }
  return (
    <ul className="runs-list">
      {runs.map((run) => {
        const active = run.id === selectedId;
        const cls = `run-row ${active ? 'active' : ''} ${run.run_type === 'attack' ? 'is-attack' : ''}`;
        return (
          <li key={run.id}>
            <button
              type="button"
              className={cls}
              onClick={() => onSelect(run.id)}
              aria-pressed={active}
              aria-label={`run ${run.id}, ${runTypeLabel(run.run_type)}`}
            >
              <span className="run-row-top">
                <span className="run-id mono">{run.id}</span>
                <span className={`run-badge ${run.run_type}`}>{runTypeLabel(run.run_type)}</span>
              </span>
              <span className="run-row-meta">
                <span>{run.n_snapshots} snapshots</span>
                {run.anomalous_final > 0 && (
                  <span className="anom-count">{run.anomalous_final} anómalos</span>
                )}
              </span>
            </button>
          </li>
        );
      })}
    </ul>
  );
}