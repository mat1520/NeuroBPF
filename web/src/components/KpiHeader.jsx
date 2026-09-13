import { runSeverity } from '../timeline.js';

const SEVERITY_LABEL = { ok: 'operación estable', warn: 'umbral ajustable', unknown: 'sin experimento' };

export default function KpiHeader({ experiment }) {
  const rep = experiment?.threshold_report;
  const severity = runSeverity(experiment);
  return (
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
          <span className="brand-sub">kernel anomaly detection · research console</span>
        </div>
      </div>

      {experiment && rep && (
        <div className="kpis" aria-label="métricas del experimento">
          <Kpi label="AUC ROC" value={experiment.auc_roc?.toFixed(3)} />
          <Kpi label="recall@topk" value={experiment.recall_at_topk?.toFixed(3)} />
          <Kpi label="TPR" value={rep.tpr?.toFixed(3)} tone="ok" />
          <Kpi label="FPR @umbral" value={rep.fpr?.toFixed(4)} tone={rep.fpr > 0.05 ? 'warn' : 'ok'} />
        </div>
      )}

      <div className={`status-pill ${severity}`} role="status">
        <span className="status-dot" aria-hidden="true" />
        {SEVERITY_LABEL[severity]}
      </div>
    </header>
  );
}

function Kpi({ label, value, tone }) {
  return (
    <div className="kpi">
      <span className={`kpi-value ${tone || ''}`}>{value ?? '—'}</span>
      <span className="kpi-label">{label}</span>
    </div>
  );
}