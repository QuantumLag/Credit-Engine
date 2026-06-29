import Gauge from '../shared/Gauge';

const RISK_COLORS = {
  High: 'var(--risk-high)',
  Medium: 'var(--risk-medium)',
  Low: 'var(--risk-low)',
};

export default function RiskCard({ applicant }) {
  const probs = applicant.class_probabilities;
  const tf = applicant.thin_file_explanation;

  return (
    <div className="card">
      <div className="risk-card-inner">
        <div className="applicant-id-label">{applicant.applicant_id}</div>
        <Gauge
          probability={applicant.predicted_default_probability}
          riskLabel={applicant.predicted_risk_label}
        />
        <div className="prob-pills">
          {['Low', 'Medium', 'High'].map((tier) => (
            <span
              key={tier}
              className="prob-pill"
              style={{
                background: `${RISK_COLORS[tier]}1a`,
                color: RISK_COLORS[tier],
              }}
            >
              {tier}: {(probs[tier] * 100).toFixed(0)}%
            </span>
          ))}
        </div>
        {applicant.is_thin_file && tf && (
          <div className="thin-file-banner">
            ⊙ Thin-file borrower — assessed via alternate data
            <div className="thin-file-stats">
              <span>Alt score: {Math.round(tf.alt_credit_score)}</span>
              <span>Confidence: {(tf.thin_file_confidence * 100).toFixed(0)}%</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
