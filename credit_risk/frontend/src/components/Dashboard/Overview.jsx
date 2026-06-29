import { useEffect, useState } from 'react';
import Spinner from '../shared/Spinner';

function formatNumber(n) {
  return n.toLocaleString('en-IN');
}

function formatPct(n) {
  return `${(n * 100).toFixed(1)}%`;
}

export default function Overview() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch('/api/dashboard/overview')
      .then((r) => {
        if (!r.ok) throw new Error('Failed to load overview');
        return r.json();
      })
      .then(setData)
      .catch((e) => setError(e.message));
  }, []);

  if (error) return <div className="error-banner">{error}</div>;
  if (!data) {
    return (
      <div className="center-loader">
        <Spinner />
      </div>
    );
  }

  const fair = !data.fairness_flags.regression && !data.fairness_flags.classification;

  return (
    <div className="overview-grid">
      <div className="card">
        <div className="stat-icon">◈</div>
        <div className="stat-value">{formatNumber(data.total_applicants)}</div>
        <div className="stat-label">loan applications assessed</div>
      </div>
      <div className="card">
        <div className="stat-icon">◈</div>
        <div className="stat-value" style={{ color: 'var(--accent-teal)' }}>
          {formatNumber(data.thin_file_count)}
        </div>
        <div className="stat-label">{data.thin_file_pct}% of portfolio</div>
        <div className="stat-extra">Assessed via alternate data</div>
      </div>
      <div className="card">
        <div className="stat-icon">◈</div>
        <div className="stat-value" style={{ color: 'var(--risk-medium)' }}>
          {formatPct(data.mean_default_probability)}
        </div>
        <div className="stat-label">portfolio average</div>
      </div>
      <div className="card">
        <div className="stat-icon">◈</div>
        <div className="stat-value" style={{ color: fair ? 'var(--risk-low)' : 'var(--risk-high)' }}>
          {fair ? '✓ Fair' : '⚠ Review'}
        </div>
        <div className="stat-label">Thin-file bias: not detected</div>
      </div>
    </div>
  );
}
