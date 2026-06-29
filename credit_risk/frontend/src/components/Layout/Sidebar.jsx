import { useEffect, useState } from 'react';

export default function Sidebar({ view, onNavigate }) {
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    fetch('/api/dashboard/overview')
      .then((r) => r.json())
      .then((d) => setMetrics(d.model_performance))
      .catch(() => {});
  }, []);

  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <span>⊙</span>CreditLens
      </div>
      <nav className="sidebar-nav">
        <button
          className={`nav-item ${view === 'dashboard' ? 'active' : ''}`}
          onClick={() => onNavigate('dashboard')}
        >
          Dashboard
        </button>
        <button
          className={`nav-item ${view === 'lookup' ? 'active' : ''}`}
          onClick={() => onNavigate('lookup')}
        >
          Applicant Lookup
        </button>
      </nav>
      <div className="sidebar-status">
        {metrics ? (
          <>
            <div>Model AUC-ROC: {metrics.auc_roc.toFixed(4)}</div>
            <div>R²: {metrics.regression_r2.toFixed(4)}</div>
          </>
        ) : (
          <>
            <div>Model AUC-ROC: —</div>
            <div>R²: —</div>
          </>
        )}
      </div>
    </aside>
  );
}
