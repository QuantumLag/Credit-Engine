import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useMemo, useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, ReferenceLine,
} from "recharts";
import { api, type Applicant } from "@/lib/api";

export const Route = createFileRoute("/applicants")({
  head: () => ({
    meta: [
      { title: "Applicant Lookup — CreditLens" },
      { name: "description", content: "Inspect individual applicant credit risk and explanations." },
    ],
  }),
  component: ApplicantsPage,
});

function ApplicantsPage() {
  const [ids, setIds] = useState<string[]>([]);
  const [query, setQuery] = useState("");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [applicant, setApplicant] = useState<Applicant | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [memo, setMemo] = useState<string | null>(null);
  const [memoLoading, setMemoLoading] = useState(false);

  useEffect(() => {
    api.applicantsList()
      .then((r) => setIds(r.ids))
      .catch((e) => setErr(String(e.message || e)));
  }, []);

  useEffect(() => {
    if (!selectedId) return;
    setLoading(true);
    setMemo(null);
    setApplicant(null);
    api.applicant(selectedId)
      .then(setApplicant)
      .catch((e) => setErr(String(e.message || e)))
      .finally(() => setLoading(false));
  }, [selectedId]);

  const filteredIds = useMemo(() => {
    const q = query.trim().toLowerCase();
    const base = q ? ids.filter((i) => i.toLowerCase().includes(q)) : ids;
    return base.slice(0, 12);
  }, [ids, query]);

  const handleGenerateMemo = async () => {
    if (!selectedId) return;
    setMemoLoading(true);
    try {
      const r = await api.narrative(selectedId);
      setMemo(r.narrative);
    } catch (e: any) {
      setMemo(`Error: ${e.message || e}`);
    } finally {
      setMemoLoading(false);
    }
  };

  return (
    <>
      <div className="page-header">
        <h1 className="page-title">Applicant Lookup</h1>
        <p className="page-subtitle">Inspect risk score, drivers, and improvement levers per applicant</p>
      </div>

      <div className="card" style={{ marginBottom: 18 }}>
        <input
          className="search-bar"
          placeholder="Search applicant ID…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        {err && !ids.length && <div className="error" style={{ marginTop: 12 }}>Unable to reach backend at localhost:8000 — {err}</div>}
        <div className="id-pills">
          {filteredIds.map((id) => (
            <button
              key={id}
              className={`id-pill ${selectedId === id ? "active" : ""}`}
              onClick={() => setSelectedId(id)}
            >
              {id}
            </button>
          ))}
          {!filteredIds.length && ids.length > 0 && <span className="empty-state">No matches</span>}
        </div>
      </div>

      {!selectedId && (
        <div className="card"><div className="empty-state">Select an applicant ID above to view their risk profile.</div></div>
      )}

      {selectedId && loading && <div className="card"><div className="loading">Loading applicant…</div></div>}

      {selectedId && applicant && (
        <div className="grid-3">
          {/* LEFT: gauge + probs */}
          <div className="card">
            <h3 className="card-title">Default Probability</h3>
            <p className="card-subtitle">{applicant.applicant_id}</p>
            <RiskGauge probability={applicant.default_probability} label={applicant.risk_label} />
            <div style={{ display: "flex", justifyContent: "center" }}>
              <span className={`risk-badge ${applicant.risk_label.toLowerCase()}`}>{applicant.risk_label} Risk</span>
            </div>
            <div style={{ marginTop: 20 }}>
              <p className="stat-label" style={{ marginBottom: 6 }}>Class Probabilities</p>
              <div className="prob-row"><span>Low</span><span className="val">{(applicant.class_probabilities.low * 100).toFixed(1)}%</span></div>
              <div className="prob-row"><span>Medium</span><span className="val">{(applicant.class_probabilities.medium * 100).toFixed(1)}%</span></div>
              <div className="prob-row"><span>High</span><span className="val">{(applicant.class_probabilities.high * 100).toFixed(1)}%</span></div>
            </div>
          </div>

          {/* CENTER: SHAP waterfall */}
          <div className="card">
            <h3 className="card-title">Score Drivers</h3>
            <p className="card-subtitle">SHAP contributions — red increases risk, green decreases</p>
            <ShapWaterfall values={applicant.shap_values} />
          </div>

          {/* RIGHT: improve + memo */}
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <div className="card">
              <h3 className="card-title">How to Improve</h3>
              <p className="card-subtitle">Counterfactual suggestions</p>
              {applicant.counterfactuals.length === 0 && <div className="empty-state">No suggestions available.</div>}
              {applicant.counterfactuals.map((c, i) => (
                <div className="improve-item" key={i}>
                  <div className="feature">{c.feature}</div>
                  <div className="change">{String(c.current)} → {String(c.suggested)}</div>
                  <div className="impact">−{(c.impact * 100).toFixed(1)}% risk</div>
                </div>
              ))}
            </div>

            <div className="card">
              <h3 className="card-title">AI Credit Memo</h3>
              <p className="card-subtitle">Analyst-ready narrative</p>
              {!memo && (
                <button className="btn-primary" onClick={handleGenerateMemo} disabled={memoLoading}>
                  {memoLoading ? "Generating…" : "Generate Memo"}
                </button>
              )}
              {memo && <div className="memo-text">{memo}</div>}
            </div>
          </div>
        </div>
      )}
    </>
  );
}

function RiskGauge({ probability, label }: { probability: number; label: string }) {
  const pct = Math.max(0, Math.min(1, probability));
  const size = 180;
  const stroke = 14;
  const r = (size - stroke) / 2;
  const cx = size / 2;
  const cy = size / 2;
  const circ = 2 * Math.PI * r;
  // 3/4 arc
  const arcLen = circ * 0.75;
  const offset = arcLen * (1 - pct);
  const color = label === "High" ? "#ef4444" : label === "Medium" ? "#f59e0b" : "#10b981";

  return (
    <div className="gauge-wrap" style={{ position: "relative", height: size }}>
      <svg width={size} height={size} style={{ transform: "rotate(135deg)" }}>
        <circle cx={cx} cy={cy} r={r} fill="none" stroke="#f1f5f9" strokeWidth={stroke} strokeDasharray={`${arcLen} ${circ}`} strokeLinecap="round" />
        <circle cx={cx} cy={cy} r={r} fill="none" stroke={color} strokeWidth={stroke}
          strokeDasharray={`${arcLen} ${circ}`} strokeDashoffset={offset} strokeLinecap="round"
          style={{ transition: "stroke-dashoffset 600ms ease" }}
        />
      </svg>
      <div style={{ position: "absolute", top: "50%", left: 0, right: 0, transform: "translateY(-50%)", textAlign: "center" }}>
        <div className="gauge-pct" style={{ color }}>{(pct * 100).toFixed(1)}%</div>
        <div className="gauge-label">Default Prob.</div>
      </div>
    </div>
  );
}

function ShapWaterfall({ values }: { values: { feature: string; value: number; feature_value?: string | number }[] }) {
  // Sort by absolute value, top 10
  const data = [...values]
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, 10)
    .reverse()
    .map((v) => ({
      name: v.feature,
      value: v.value,
      fv: v.feature_value,
    }));

  const max = Math.max(...data.map((d) => Math.abs(d.value)), 0.01);

  return (
    <ResponsiveContainer width="100%" height={Math.max(360, data.length * 36)}>
      <BarChart data={data} layout="vertical" margin={{ top: 8, right: 24, left: 16, bottom: 0 }}>
        <XAxis type="number" domain={[-max, max]} stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} />
        <YAxis type="category" dataKey="name" stroke="#475569" fontSize={12} tickLine={false} axisLine={false} width={150} />
        <ReferenceLine x={0} stroke="#cbd5e1" />
        <Tooltip
          cursor={{ fill: "rgba(79,126,248,0.06)" }}
          contentStyle={{ background: "#fff", border: "1px solid #e5e7eb", borderRadius: 6, fontSize: 12 }}
          formatter={((val: number, _name: string, item: { payload?: { fv?: string | number } }) => [
            `${val > 0 ? "+" : ""}${val.toFixed(3)}${item?.payload?.fv !== undefined ? ` (value: ${item.payload.fv})` : ""}`,
            "SHAP",
          ]) as never}
        />
        <Bar dataKey="value" radius={[4, 4, 4, 4]}>
          {data.map((d, i) => (
            <Cell key={i} fill={d.value >= 0 ? "#ef4444" : "#10b981"} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
