const BASE = '/api';

async function get(path) {
  try {
    const res = await fetch(`${BASE}${path}`);
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

export const fetchExperiment = () => get('/experiment');
export const fetchRuns = () => get('/runs');
export const fetchRun = (id) => get(`/runs/${encodeURIComponent(id)}`);
export const fetchReport = () => get('/report');