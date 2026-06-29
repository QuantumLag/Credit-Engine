import { useEffect, useState } from 'react';

const DRIVER_COLOR = 'var(--risk-high)';
const MITIGANT_COLOR = 'var(--accent-teal)';

function WaterfallBar({ item, maxAbs, index, type }) {
  const [width, setWidth] = useState(0);
  const pct = maxAbs > 0 ? (Math.abs(item.shap_value) / maxAbs) * 45 : 0;
  const color = type === 'driver' ? DRIVER_COLOR : MITIGANT_COLOR;

  useEffect(() => {
    const timer = setTimeout(() => setWidth(pct), 60 * index);
    return () => clearTimeout(timer);
  }, [pct, index]);

  return (
    <div className="waterfall-row">
      <div className="waterfall-feature-col">
        <div className="waterfall-feature-name">{item.feature}</div>
        <div className="waterfall-feature-value">{item.human_readable}</div>
      </div>
      <div className="waterfall-bar-area">
        <div className="waterfall-baseline" />
        <div
          className={`waterfall-bar ${type}`}
          style={{
            width: `${width}%`,
            background: color,
            opacity: type === 'driver' && item.magnitude === 'high' ? 1 : 0.75,
          }}
        />
      </div>
      <div className="waterfall-shap-value" style={{ color }}>
        {item.shap_value > 0 ? '+' : ''}{item.shap_value.toFixed(3)}
      </div>
    </div>
  );
}

export default function ShapWaterfall({ applicant }) {
  const drivers = applicant.top_risk_drivers || [];
  const mitigants = applicant.top_risk_mitigants || [];
  const allItems = [...drivers, ...mitigants];
  const maxAbs = Math.max(...allItems.map((i) => Math.abs(i.shap_value)), 0.001);

  return (
    <div className="card">
      <div className="waterfall-title">What drove this score</div>
      <div className="waterfall-subtitle">← Reduces risk &nbsp;&nbsp; Increases risk →</div>
      <div className="waterfall-chart">
        {mitigants.map((item, i) => (
          <WaterfallBar key={`m-${item.feature}`} item={item} maxAbs={maxAbs} index={i} type="mitigant" />
        ))}
        {drivers.map((item, i) => (
          <WaterfallBar
            key={`d-${item.feature}`}
            item={item}
            maxAbs={maxAbs}
            index={mitigants.length + i}
            type="driver"
          />
        ))}
      </div>
    </div>
  );
}
