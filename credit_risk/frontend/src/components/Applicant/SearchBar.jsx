import { useEffect, useState } from 'react';

export default function SearchBar({ onSelect }) {
  const [query, setQuery] = useState('');
  const [applicants, setApplicants] = useState([]);

  useEffect(() => {
    fetch('/api/applicants/list')
      .then((r) => r.json())
      .then(setApplicants)
      .catch(() => {});
  }, []);

  function handleSubmit(e) {
    e.preventDefault();
    if (!query.trim()) return;
    fetch(`/api/applicants/search?q=${encodeURIComponent(query.trim())}`)
      .then((r) => r.json())
      .then((results) => {
        if (results.length > 0) {
          onSelect(results[0].applicant_id);
        }
      });
  }

  function shortId(id) {
    return id.slice(0, 8);
  }

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <input
          className="search-input"
          type="text"
          placeholder="Enter applicant ID..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
      </form>
      <div className="applicant-pills">
        {applicants.map((a) => (
          <button
            key={a.applicant_id}
            type="button"
            className="applicant-pill"
            onClick={() => onSelect(a.applicant_id)}
            title={a.applicant_id}
          >
            {shortId(a.applicant_id)}
            {a.is_thin_file && <sup className="tf-mark">TF</sup>}
          </button>
        ))}
      </div>
    </div>
  );
}
