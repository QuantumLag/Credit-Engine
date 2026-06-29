function hasActionableCounterfactuals(counterfactuals) {
  if (!counterfactuals?.length) return false;
  return counterfactuals.some((cf) => cf.estimated_probability_reduction != null);
}

export default function CounterfactualCard({ counterfactuals, currentProbability }) {
  const actionable = hasActionableCounterfactuals(counterfactuals);

  return (
    <div className="card">
      <div className="card-title">How to improve this score</div>
      <div className="card-subtitle">Actionable changes that could reduce default risk</div>
      {!actionable ? (
        <div className="cf-empty">
          Risk is driven by non-actionable factors.
          Manual underwriter review recommended.
        </div>
      ) : (
        counterfactuals
          .filter((cf) => cf.estimated_probability_reduction != null)
          .map((cf, i) => {
            const newProb = cf.new_predicted_probability ?? currentProbability;
            return (
              <div key={i} className="cf-row">
                {cf.feasibility && (
                  <span className={`cf-feasibility ${cf.feasibility}`}>
                    {cf.feasibility}
                  </span>
                )}
                <div className="cf-row-header">
                  <div>
                    <div className="cf-action">{cf.action}</div>
                    {cf.current_value_human && cf.suggested_value_human && (
                      <div className="cf-values">
                        {cf.current_value_human} → {cf.suggested_value_human}
                      </div>
                    )}
                  </div>
                  <span className="cf-reduction-badge">
                    ↓ {(cf.estimated_probability_reduction * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="cf-progress">
                  <div
                    className="cf-progress-before"
                    style={{ width: `${currentProbability * 100}%` }}
                  />
                  <div
                    className="cf-progress-after"
                    style={{ width: `${newProb * 100}%` }}
                  />
                </div>
              </div>
            );
          })
      )}
    </div>
  );
}
