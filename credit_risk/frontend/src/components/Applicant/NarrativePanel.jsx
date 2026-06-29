import { useState } from 'react';

const REC_CLASS = {
  APPROVE: 'approve',
  'APPROVE WITH CONDITIONS': 'approve-conditions',
  'REFER FOR REVIEW': 'refer',
  DECLINE: 'decline',
};

function parseMemoSections(text) {
  const lines = text.split('\n');
  const elements = [];
  let currentSection = null;
  let buffer = [];

  function flushBuffer() {
    if (buffer.length === 0) return;
    buffer.forEach((line) => {
      const trimmed = line.trim();
      if (!trimmed) return;
      if (trimmed.startsWith('•') || trimmed.startsWith('-') || trimmed.startsWith('*')) {
        elements.push(
          <div key={`${elements.length}-b`} className="memo-bullet">
            {trimmed.replace(/^[•\-*]\s*/, '')}
          </div>
        );
      } else {
        elements.push(<p key={`${elements.length}-p`}>{trimmed}</p>);
      }
    });
    buffer = [];
  }

  lines.forEach((line) => {
    const trimmed = line.trim();
    const sectionMatch = trimmed.match(/^(\d+\.\s*)?([A-Z][A-Z\s\-]+):?\s*$/);
    const knownSections = [
      'CREDIT SUMMARY',
      'KEY RISK FACTORS',
      'MITIGATING STRENGTHS',
      'THIN-FILE ASSESSMENT',
      'RECOMMENDED ACTIONS',
      'ANALYST RECOMMENDATION',
    ];
    const isHeader = knownSections.some((s) => trimmed.toUpperCase().startsWith(s));

    if (isHeader) {
      flushBuffer();
      const headerText = trimmed.replace(/^\d+\.\s*/, '').replace(/:$/, '');
      elements.push(
        <div key={`${elements.length}-h`} className="memo-section-header">
          {headerText}
        </div>
      );
      currentSection = headerText;
    } else if (trimmed.startsWith('•') || trimmed.startsWith('-') || trimmed.startsWith('*')) {
      elements.push(
        <div key={`${elements.length}-b`} className="memo-bullet">
          {trimmed.replace(/^[•\-*]\s*/, '')}
        </div>
      );
    } else if (trimmed) {
      buffer.push(trimmed);
    }
  });
  flushBuffer();
  return elements;
}

export default function NarrativePanel({ applicantId }) {
  const [narrative, setNarrative] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function generate(force = false) {
    setLoading(true);
    setError(null);
    if (force) setNarrative(null);
    try {
      const res = await fetch('/api/explain/narrative', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ applicant_id: applicantId, force }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Failed to generate narrative');
      }
      const data = await res.json();
      setNarrative(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  const recClass = narrative
    ? REC_CLASS[narrative.recommendation] || 'refer'
    : '';

  return (
    <div className="card">
      <div className="narrative-header">
        <div className="card-title" style={{ marginBottom: 0 }}>AI Credit Memo</div>
        <span className="claude-badge">Powered by Claude</span>
      </div>

      {!narrative && !loading && (
        <div style={{ padding: '24px 0' }}>
          <button
            type="button"
            className="narrative-generate-btn"
            onClick={() => generate()}
          >
            Generate Credit Memo
          </button>
          <p className="narrative-hint">Uses Claude AI to generate analyst memo</p>
        </div>
      )}

      {loading && (
        <div className="narrative-loading">
          <div className="loading-dots">
            <span /><span /><span />
          </div>
          <p style={{ color: 'var(--text-secondary)', fontSize: 13 }}>
            Analyzing risk profile...
          </p>
          <div className="skeleton-lines">
            <div className="skeleton-line" />
            <div className="skeleton-line" />
            <div className="skeleton-line" />
            <div className="skeleton-line" />
          </div>
        </div>
      )}

      {error && <div className="error-banner">{error}</div>}

      {narrative && !loading && (
        <>
          <div className={`recommendation-banner ${recClass}`}>
            {narrative.recommendation}
          </div>
          <div className="memo-content">
            {parseMemoSections(narrative.narrative)}
          </div>
          <div className="memo-footer">
            <span>Generated at {new Date(narrative.generated_at).toLocaleString()}</span>
            <button
              type="button"
              className="regenerate-link"
              onClick={() => generate(true)}
            >
              Regenerate
            </button>
          </div>
        </>
      )}
    </div>
  );
}
