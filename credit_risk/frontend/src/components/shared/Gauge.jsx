const RISK_COLORS = {
  High: 'var(--risk-high)',
  Medium: 'var(--risk-medium)',
  Low: 'var(--risk-low)',
};

export default function Gauge({ probability, riskLabel }) {
  const pct = Math.min(Math.max(probability * 100, 0), 100);
  const radius = 68;
  const stroke = 10;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (pct / 100) * circumference;
  const color = RISK_COLORS[riskLabel] || 'var(--text-muted)';

  const gradientId = `gauge-gradient-${riskLabel}`;

  return (
    <div className="gauge-container">
      <svg width="160" height="160" viewBox="0 0 160 160">
        <defs>
          <linearGradient id={gradientId} x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor={color} stopOpacity="0.6" />
            <stop offset="100%" stopColor={color} stopOpacity="1" />
          </linearGradient>
        </defs>
        <circle
          cx="80"
          cy="80"
          r={radius}
          fill="none"
          stroke="var(--border)"
          strokeWidth={stroke}
          transform="rotate(-90 80 80)"
        />
        <circle
          cx="80"
          cy="80"
          r={radius}
          fill="none"
          stroke={`url(#${gradientId})`}
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          transform="rotate(-90 80 80)"
          style={{ transition: 'stroke-dashoffset 0.8s ease' }}
        />
        <text
          x="80"
          y="76"
          textAnchor="middle"
          fill="var(--text-primary)"
          fontSize="26"
          fontWeight="600"
          fontFamily="var(--font-sans)"
        >
          {pct.toFixed(1)}%
        </text>
        <text
          x="80"
          y="96"
          textAnchor="middle"
          fill="var(--text-muted)"
          fontSize="10"
          fontFamily="var(--font-sans)"
        >
          default prob.
        </text>
      </svg>
      <div className="gauge-label" style={{ color }}>
        {riskLabel} Risk
      </div>
    </div>
  );
}
