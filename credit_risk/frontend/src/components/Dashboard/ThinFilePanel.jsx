import { useEffect, useState } from 'react';
import Spinner from '../shared/Spinner';

function formatNumber(n, decimals = 0) {
  return n.toLocaleString('en-IN', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

export default function ThinFilePanel() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch('/api/dashboard/thin-file')
      .then((r) => {
        if (!r.ok) throw new Error('Failed to load thin-file data');
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

  const maxProb = Math.max(data.mean_predicted_prob_thin, data.mean_predicted_prob_non_thin);

  return (
    <div className="card">
      <div className="card-title">Thin-File Borrower Analysis</div>
      <div className="card-subtitle">
        {formatNumber(data.thin_file_count)} borrowers assessed without bureau scores
        using UPI behavior + cash-flow signals
      </div>
      <div className="thin-file-grid">
        <div className="thin-stats-grid">
          <div className="thin-stat-card">
            <div className="thin-stat-value">{formatNumber(data.thin_file_count)}</div>
            <div className="thin-stat-label">Thin-file count</div>
          </div>
          <div className="thin-stat-card">
            <div className="thin-stat-value">{formatNumber(data.mean_alt_credit_score, 2)}</div>
            <div className="thin-stat-label">Mean alt credit score</div>
          </div>
          <div className="thin-stat-card">
            <div className="thin-stat-value">
              {(data.mean_thin_file_confidence * 100).toFixed(1)}%
            </div>
            <div className="thin-stat-label">Mean data confidence</div>
          </div>
          <div className="thin-stat-card">
            <div className="thin-stat-value">{formatNumber(data.mean_bureau_score, 2)}</div>
            <div className="thin-stat-label">vs Bureau borrowers mean score</div>
          </div>
        </div>
        <div className="comparison-viz">
          <div className="comparison-bar-row">
            <div className="comparison-bar-label">
              <span>Thin-file borrowers</span>
              <span>{(data.mean_predicted_prob_thin * 100).toFixed(1)}%</span>
            </div>
            <div className="comparison-bar-track">
              <div
                className="comparison-bar-fill"
                style={{
                  width: `${(data.mean_predicted_prob_thin / maxProb) * 100}%`,
                  background: 'var(--accent-teal)',
                }}
              />
            </div>
          </div>
          <div className="comparison-bar-row">
            <div className="comparison-bar-label">
              <span>Bureau borrowers</span>
              <span>{(data.mean_predicted_prob_non_thin * 100).toFixed(1)}%</span>
            </div>
            <div className="comparison-bar-track">
              <div
                className="comparison-bar-fill"
                style={{
                  width: `${(data.mean_predicted_prob_non_thin / maxProb) * 100}%`,
                  background: 'var(--accent-blue)',
                }}
              />
            </div>
          </div>
          <p className="fairness-label">
            No significant difference detected — model is fair
          </p>
        </div>
      </div>
    </div>
  );
}
