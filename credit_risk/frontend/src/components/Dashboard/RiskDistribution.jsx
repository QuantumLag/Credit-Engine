import { useEffect, useState } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from 'recharts';
import Spinner from '../shared/Spinner';

const RISK_CONFIG = {
  High: { color: 'var(--risk-high)', label: 'High Risk' },
  Medium: { color: 'var(--risk-medium)', label: 'Medium Risk' },
  Low: { color: 'var(--risk-low)', label: 'Low Risk' },
};

function CustomTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const item = payload[0].payload;
  return (
    <div className="custom-tooltip">
      <div className="label">{item.label}</div>
      <div className="value">{item.count.toLocaleString('en-IN')} applicants</div>
    </div>
  );
}

export default function RiskDistribution() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch('/api/dashboard/overview')
      .then((r) => {
        if (!r.ok) throw new Error('Failed to load risk distribution');
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

  const total = data.total_applicants;
  const dist = data.risk_distribution;

  const chartData = ['High', 'Medium', 'Low'].map((tier) => ({
    tier,
    label: RISK_CONFIG[tier].label,
    count: dist[tier] || 0,
    fill: RISK_CONFIG[tier].color,
    pct: total ? ((dist[tier] || 0) / total) * 100 : 0,
  }));

  return (
    <div className="card">
      <div className="card-title">Risk Distribution</div>
      <div className="card-subtitle">Portfolio breakdown by predicted risk tier</div>
      <div className="risk-dist-grid">
        <div style={{ height: 280 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
              <XAxis
                dataKey="tier"
                tick={{ fill: 'var(--text-secondary)', fontSize: 12 }}
                axisLine={{ stroke: 'var(--border)' }}
                tickLine={false}
              />
              <YAxis
                tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
              <Bar dataKey="count" radius={[4, 4, 0, 0]} maxBarSize={80} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="risk-breakdown-list">
          {chartData.map((row) => (
            <div key={row.tier} className="risk-row">
              <span className="risk-dot" style={{ background: row.fill }} />
              <span className="risk-row-label">{row.label}</span>
              <div className="progress-bar-track">
                <div
                  className="progress-bar-fill"
                  style={{ width: `${row.pct}%`, background: row.fill }}
                />
              </div>
              <span className="risk-row-count">{row.count.toLocaleString('en-IN')}</span>
              <span className="risk-row-pct">{row.pct.toFixed(1)}%</span>
            </div>
          ))}
          <p className="risk-note">Class imbalance handled via SMOTE during training</p>
        </div>
      </div>
    </div>
  );
}
