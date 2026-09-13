const KIND_GLYPH = { process: '⌬', file: '▭', net: '◆' };

export default function NodeDetail({ frame, node, selected }) {
  const nodes = frame?.nodes || [];
  const edges = frame?.edges || [];
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const current = node instanceof Object && node.id != null ? node : null;
  const connections = current
    ? edges
        .filter((e) => e.src === current.id || e.dst === current.id)
        .map((e) => {
          const otherId = e.src === current.id ? e.dst : e.src;
          return { id: otherId, label: e.label, direction: e.src === current.id ? '→' : '←', node: byId.get(otherId) };
        })
    : [];

  return (
    <div className="node-detail">
      {!current ? (
        <p className="panel-hint">haz clic en un nodo del grafo</p>
      ) : (
        <>
          <div className="detail-head">
            <span className={`node-glyph kind-${current.kind || 'process'}`}>{KIND_GLYPH[current.kind] || '●'}</span>
            <div>
              <h3 className="mono">{current.label || current.id}</h3>
              <span className={`run-badge ${current.anomalous ? 'attack' : 'normal'}`}>
                {current.anomalous ? 'anómalo' : 'normal'}
              </span>
            </div>
          </div>
          <dl className="detail-list">
            <dt>id</dt>
            <dd className="mono">{current.id}</dd>
            <dt>tipo</dt>
            <dd>{current.kind || 'desconocido'}</dd>
            <dt>score anomalía</dt>
            <dd className="mono">{typeof current.score === 'number' ? current.score.toFixed(3) : '—'}</dd>
          </dl>
          <h4>conexiones ({connections.length})</h4>
          <ul className="connections">
            {connections.map((c) => (
              <li key={`${c.id}:${c.label}`}>
                <span className="mono conn-label">{c.label}</span>
                <span className="conn-dir" aria-hidden="true">{c.direction}</span>
                <span className={`mono conn-id ${c.node?.anomalous ? 'anomalous-text' : ''}`}>{c.id}</span>
                <span className={`node-glyph kind-${c.node?.kind || 'process'}`} aria-hidden="true">
                  {KIND_GLYPH[c.node?.kind] || '●'}
                </span>
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}