import { useEffect, useState } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Cell,
} from 'recharts';
import Spinner from '../shared/Spinner';

const GROUP_COLORS = {
  'Credit signals': 'var(--accent-purple)',
  'UPI behavior': 'var(--accent-teal)',
  'Income & capacity': 'var(--accent-blue)',
  'Loan stress': 'var(--accent-amber)',
  'Employment profile': '#ec4899',
  'Thin-file signals': 'var(--accent-teal)',
  'Loan structure': 'var(--text-muted)',
  Geography: 'var(--text-muted)',
  Others: 'var(--text-muted)',
};

function CustomTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const item = payload[0].payload;
  return (
    <div className="custom-tooltip">
      <div className="label">{item.feature}</div>
      <div className="value">SHAP: {item.mean_abs_shap.toFixed(4)}</div>
      <div className="label" style={{ marginTop: 4 }}>{item.group}</div>
    </div>
  );
}

export default function FeatureImportance() {
  const [data, setData] = useState(null);
  const [tab, setTab] = useState('regression');
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch('/api/dashboard/feature-importance')
      .then((r) => {
        if (!r.ok) throw new Error('Failed to load feature importance');
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

  const features = (tab === 'regression' ? data.regression : data.classification)
    .slice(0, 10)
    .map((f) => ({
      ...f,
      group: f.group || 'Others',
    }))
    .reverse();

  const usedGroups = [...new Set(features.map((f) => f.group))];

  return (
    <div className="card">
      <div className="card-title">Feature Importance</div>
      <div className="card-subtitle">Top 10 features by mean absolute SHAP value</div>
      <div className="feature-tabs">
        <button
          className={`feature-tab ${tab === 'regression' ? 'active' : ''}`}
          onClick={() => setTab('regression')}
        >
          Regression
        </button>
        <button
          className={`feature-tab ${tab === 'classification' ? 'active' : ''}`}
          onClick={() => setTab('classification')}
        >
          Classification
        </button>
      </div>
      <div style={{ height: 360 }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={features}
            layout="vertical"
            margin={{ top: 0, right: 24, left: 8, bottom: 0 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" horizontal={false} />
            <XAxis
              type="number"
              tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              type="category"
              dataKey="feature"
              width={160}
              tick={{ fill: 'var(--text-secondary)', fontSize: 11, fontFamily: 'var(--font-mono)' }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
            <Bar dataKey="mean_abs_shap" radius={[0, 4, 4, 0]} maxBarSize={18}>
              {features.map((entry) => (
                <Cell key={entry.feature} fill={GROUP_COLORS[entry.group] || GROUP_COLORS.Others} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="group-legend">
        {usedGroups.map((group) => (
          <div key={group} className="legend-item">
            <span
              className="legend-dot"
              style={{ background: GROUP_COLORS[group] || GROUP_COLORS.Others }}
            />
            {group}
          </div>
        ))}
      </div>
    </div>
  );
}
