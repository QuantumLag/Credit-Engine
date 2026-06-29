import { useState } from 'react';
import Sidebar from './components/Layout/Sidebar';
import TopBar from './components/Layout/TopBar';
import Overview from './components/Dashboard/Overview';
import RiskDistribution from './components/Dashboard/RiskDistribution';
import FeatureImportance from './components/Dashboard/FeatureImportance';
import ThinFilePanel from './components/Dashboard/ThinFilePanel';
import SearchBar from './components/Applicant/SearchBar';
import RiskCard from './components/Applicant/RiskCard';
import ShapWaterfall from './components/Applicant/ShapWaterfall';
import CounterfactualCard from './components/Applicant/CounterfactualCard';
import NarrativePanel from './components/Applicant/NarrativePanel';
import Spinner from './components/shared/Spinner';

const VIEWS = {
  dashboard: 'Dashboard',
  lookup: 'Applicant Lookup',
};

export default function App() {
  const [view, setView] = useState('dashboard');
  const [applicant, setApplicant] = useState(null);
  const [loadingApplicant, setLoadingApplicant] = useState(false);
  const [applicantError, setApplicantError] = useState(null);

  async function loadApplicant(id) {
    setLoadingApplicant(true);
    setApplicantError(null);
    setApplicant(null);
    try {
      const res = await fetch(`/api/applicants/${id}`);
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Failed to load applicant');
      }
      const data = await res.json();
      setApplicant(data);
      setView('lookup');
    } catch (e) {
      setApplicantError(e.message);
    } finally {
      setLoadingApplicant(false);
    }
  }

  return (
    <div className="app-shell">
      <Sidebar view={view} onNavigate={setView} />
      <div className="main-area">
        <TopBar title={VIEWS[view]} />
        <main className="content">
          {view === 'dashboard' && (
            <div className="dashboard-view">
              <Overview />
              <RiskDistribution />
              <FeatureImportance />
              <ThinFilePanel />
            </div>
          )}
          {view === 'lookup' && (
            <div className="lookup-view">
              <SearchBar onSelect={loadApplicant} />
              {loadingApplicant && (
                <div className="center-loader">
                  <Spinner />
                  <span>Loading applicant...</span>
                </div>
              )}
              {applicantError && (
                <div className="error-banner">{applicantError}</div>
              )}
              {applicant && !loadingApplicant && (
                <div className="lookup-grid">
                  <div className="lookup-left">
                    <RiskCard applicant={applicant} />
                    <ShapWaterfall applicant={applicant} />
                  </div>
                  <div className="lookup-right">
                    <NarrativePanel applicantId={applicant.applicant_id} />
                    <CounterfactualCard
                      counterfactuals={applicant.counterfactuals}
                      currentProbability={applicant.predicted_default_probability}
                    />
                  </div>
                </div>
              )}
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
