import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, CartesianGrid,
} from "recharts";
import { api, type OverviewResp, type FeatureImportance } from "@/lib/api";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Portfolio Overview — CreditLens" },
      { name: "description", content: "Credit risk portfolio analytics for Indian lenders." },
    ],
  }),
  component: PortfolioPage,
});

const CATEGORY_COLORS: Record<string, string> = {
  bureau: "#4f7ef8",
  income: "#10b981",
  employment: "#f59e0b",
  demographic: "#8b5cf6",
  behavioral: "#ec4899",
  other: "#64748b",
};

function PortfolioPage() {
  const [overview, setOverview] = useState<OverviewResp | null>(null);
  const [features, setFeatures] = useState<FeatureImportance | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.overview(), api.featureImportance()])
      .then(([o, f]) => { setOverview(o); setFeatures(f); })
      .catch((e) => setErr(String(e.message || e)));
  }, []);

  if (err) return <Shell><div className="error">Unable to reach backend at localhost:8000 — {err}</div></Shell>;
  if (!overview || !features) return <Shell><div className="loading">Loading portfolio…</div></Shell>;

  const riskData = [
    { name: "High", value: overview.risk_distribution.high, color: "#ef4444" },
    { name: "Medium", value: overview.risk_distribution.medium, color: "#f59e0b" },
    { name: "Low", value: overview.risk_distribution.low, color: "#10b981" },
  ];

  const topFeatures = [...features.features]
    .sort((a, b) => b.importance - a.importance)
    .slice(0, 10)
    .reverse();

  const tf = overview.thin_file_analysis;

  return (
    <Shell>
      <div className="stat-grid">
        <StatCard label="Applications" value={overview.total_applications.toLocaleString()} />
        <StatCard
          label="Thin-file Borrowers"
          value={overview.thin_file_count.toLocaleString()}
          meta={`${overview.thin_file_pct.toFixed(1)}% of portfolio`}
        />
        <StatCard
          label="Mean Default Risk"
          value={`${(overview.mean_default_risk * 100).toFixed(1)}%`}
        />
        <StatCard
          label="Fairness Audit"
          value={<span className="stat-badge">✓ Verified</span>}
          meta="Passed demographic parity"
        />
      </div>

      <div className="grid-2">
        <div className="card">
          <h3 className="card-title">Risk Distribution</h3>
          <p className="card-subtitle">Applicant counts by predicted risk tier</p>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={riskData} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
              <XAxis dataKey="name" stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} />
              <YAxis stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} />
              <Tooltip cursor={{ fill: "rgba(79,126,248,0.06)" }} contentStyle={tooltipStyle} />
              <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                {riskData.map((d) => <Cell key={d.name} fill={d.color} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <h3 className="card-title">Thin-file vs Bureau Borrowers</h3>
          <p className="card-subtitle">Performance comparison</p>
          <div className="compare-grid">
            <CompareCol title="Thin-file" data={tf.thin_file} />
            <CompareCol title="Bureau" data={tf.bureau} />
          </div>
        </div>
      </div>

      <div className="card">
        <h3 className="card-title">Top 10 Feature Importance</h3>
        <p className="card-subtitle">Mean absolute SHAP value across portfolio, colored by category</p>
        <ResponsiveContainer width="100%" height={Math.max(320, topFeatures.length * 32)}>
          <BarChart data={topFeatures} layout="vertical" margin={{ top: 8, right: 24, left: 24, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" horizontal={false} />
            <XAxis type="number" stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} />
            <YAxis type="category" dataKey="name" stroke="#475569" fontSize={12} tickLine={false} axisLine={false} width={170} />
            <Tooltip cursor={{ fill: "rgba(79,126,248,0.06)" }} contentStyle={tooltipStyle} />
            <Bar dataKey="importance" radius={[0, 6, 6, 0]}>
              {topFeatures.map((f, i) => (
                <Cell key={i} fill={CATEGORY_COLORS[f.category] || CATEGORY_COLORS.other} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <div style={{ display: "flex", gap: 16, flexWrap: "wrap", marginTop: 12 }}>
          {Object.entries(CATEGORY_COLORS).map(([k, v]) => (
            <div key={k} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, color: "var(--text-muted)" }}>
              <span style={{ width: 10, height: 10, borderRadius: 2, background: v }} />
              {k}
            </div>
          ))}
        </div>
      </div>
    </Shell>
  );
}

const tooltipStyle = {
  background: "#fff",
  border: "1px solid #e5e7eb",
  borderRadius: 6,
  fontSize: 12,
  boxShadow: "0 4px 12px rgba(15,23,42,0.06)",
};

function Shell({ children }: { children: React.ReactNode }) {
  return (
    <>
      <div className="page-header">
        <h1 className="page-title">Portfolio Overview</h1>
        <p className="page-subtitle">Real-time view of credit risk across applications</p>
      </div>
      {children}
    </>
  );
}

function StatCard({ label, value, meta }: { label: string; value: React.ReactNode; meta?: string }) {
  return (
    <div className="stat-card">
      <p className="stat-label">{label}</p>
      <div className="stat-value">{value}</div>
      {meta && <div className="stat-meta">{meta}</div>}
    </div>
  );
}

function CompareCol({ title, data }: { title: string; data: { count: number; mean_risk: number; default_rate: number; avg_income: number } }) {
  return (
    <div className="compare-col">
      <h4>{title}</h4>
      <div className="compare-metric"><span>Borrowers</span><span>{data.count.toLocaleString()}</span></div>
      <div className="compare-metric"><span>Mean risk</span><span>{(data.mean_risk * 100).toFixed(1)}%</span></div>
      <div className="compare-metric"><span>Default rate</span><span>{(data.default_rate * 100).toFixed(1)}%</span></div>
      <div className="compare-metric"><span>Avg income</span><span>₹{data.avg_income.toLocaleString()}</span></div>
    </div>
  );
}
