// Use relative URLs so the Vite dev-server proxy (→ localhost:8000) and
// the production FastAPI static-serve both work without CORS issues.
const BASE = "";
const TIMEOUT_MS = 3000;

async function fetchWithTimeout(url: string, init?: RequestInit): Promise<Response> {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
  try {
    return await fetch(url, { ...init, signal: ctrl.signal });
  } finally {
    clearTimeout(t);
  }
}

async function get<T>(path: string, fallback: T): Promise<T> {
  try {
    const r = await fetchWithTimeout(`${BASE}${path}`);
    if (!r.ok) throw new Error(`${r.status} ${path}`);
    return (await r.json()) as T;
  } catch {
    return fallback;
  }
}

async function post<T>(path: string, body: unknown, fallback: T): Promise<T> {
  try {
    const r = await fetchWithTimeout(`${BASE}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error(`${r.status} ${path}`);
    return (await r.json()) as T;
  } catch {
    return fallback;
  }
}

export interface OverviewResp {
  total_applications: number;
  thin_file_count: number;
  thin_file_pct: number;
  mean_default_risk: number;
  fairness_verified: boolean;
  risk_distribution: { high: number; medium: number; low: number };
  thin_file_analysis: {
    thin_file: { count: number; mean_risk: number; default_rate: number; avg_income: number };
    bureau: { count: number; mean_risk: number; default_rate: number; avg_income: number };
  };
}

export interface FeatureImportance {
  features: { name: string; importance: number; category: string }[];
}

export interface ApplicantList {
  ids: string[];
}

export interface ShapValue {
  feature: string;
  value: number;
  feature_value?: string | number;
}

export interface Applicant {
  applicant_id: string;
  default_probability: number;
  risk_label: "High" | "Medium" | "Low";
  class_probabilities: { low: number; medium: number; high: number };
  shap_values: ShapValue[];
  counterfactuals: { feature: string; current: string | number; suggested: string | number; impact: number }[];
}

export interface NarrativeResp {
  narrative: string;
}

// ---------- MOCK DATA ----------

const MOCK_OVERVIEW: OverviewResp = {
  total_applications: 24875,
  thin_file_count: 9420,
  thin_file_pct: 37.9,
  mean_default_risk: 0.182,
  fairness_verified: true,
  risk_distribution: { high: 3120, medium: 7895, low: 13860 },
  thin_file_analysis: {
    thin_file: { count: 9420, mean_risk: 0.234, default_rate: 0.097, avg_income: 28400 },
    bureau: { count: 15455, mean_risk: 0.151, default_rate: 0.062, avg_income: 54200 },
  },
};

const MOCK_FEATURES: FeatureImportance = {
  features: [
    { name: "bureau_score", importance: 0.187, category: "bureau" },
    { name: "monthly_income", importance: 0.142, category: "income" },
    { name: "dti_ratio", importance: 0.118, category: "income" },
    { name: "employment_tenure", importance: 0.094, category: "employment" },
    { name: "open_credit_lines", importance: 0.081, category: "bureau" },
    { name: "upi_txn_volume_3m", importance: 0.073, category: "behavioral" },
    { name: "age", importance: 0.061, category: "demographic" },
    { name: "salary_credit_regularity", importance: 0.058, category: "income" },
    { name: "employer_category", importance: 0.047, category: "employment" },
    { name: "city_tier", importance: 0.041, category: "demographic" },
    { name: "mobile_recharge_pattern", importance: 0.034, category: "behavioral" },
    { name: "previous_defaults", importance: 0.029, category: "bureau" },
  ],
};

const MOCK_IDS: string[] = Array.from({ length: 24 }, (_, i) =>
  `APP-2026-${String(10247 + i * 13).padStart(5, "0")}`,
);

function mockApplicant(id: string): Applicant {
  // Deterministic pseudo-random from id
  let seed = 0;
  for (let i = 0; i < id.length; i++) seed = (seed * 31 + id.charCodeAt(i)) >>> 0;
  const rand = () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return (seed & 0xffff) / 0xffff;
  };
  const prob = 0.05 + rand() * 0.85;
  const label: Applicant["risk_label"] = prob > 0.6 ? "High" : prob > 0.3 ? "Medium" : "Low";
  const low = label === "Low" ? 0.6 + rand() * 0.3 : label === "Medium" ? 0.15 + rand() * 0.2 : 0.05 + rand() * 0.1;
  const high = label === "High" ? 0.55 + rand() * 0.3 : label === "Medium" ? 0.2 + rand() * 0.2 : 0.05 + rand() * 0.1;
  const medium = Math.max(0, 1 - low - high);
  const sign = label === "High" ? 1 : -1;
  return {
    applicant_id: id,
    default_probability: prob,
    risk_label: label,
    class_probabilities: { low, medium, high },
    shap_values: [
      { feature: "bureau_score", value: sign * (0.08 + rand() * 0.06), feature_value: Math.round(550 + rand() * 280) },
      { feature: "dti_ratio", value: sign * (0.05 + rand() * 0.05), feature_value: (0.2 + rand() * 0.5).toFixed(2) },
      { feature: "monthly_income", value: -sign * (0.04 + rand() * 0.05), feature_value: `₹${Math.round(20000 + rand() * 80000).toLocaleString()}` },
      { feature: "employment_tenure", value: -sign * (0.03 + rand() * 0.04), feature_value: `${(rand() * 10).toFixed(1)} yrs` },
      { feature: "open_credit_lines", value: sign * (0.02 + rand() * 0.04), feature_value: Math.round(rand() * 8) },
      { feature: "upi_txn_volume_3m", value: -sign * (0.02 + rand() * 0.03), feature_value: Math.round(20 + rand() * 200) },
      { feature: "previous_defaults", value: sign * (0.015 + rand() * 0.03), feature_value: Math.round(rand() * 2) },
      { feature: "salary_credit_regularity", value: -sign * (0.015 + rand() * 0.025), feature_value: `${Math.round(60 + rand() * 40)}%` },
      { feature: "city_tier", value: sign * (0.01 + rand() * 0.02), feature_value: `Tier ${1 + Math.floor(rand() * 3)}` },
      { feature: "age", value: sign * (0.008 + rand() * 0.015), feature_value: Math.round(22 + rand() * 35) },
    ],
    counterfactuals: [
      { feature: "DTI ratio", current: "0.52", suggested: "≤ 0.35", impact: 0.08 + rand() * 0.04 },
      { feature: "Open credit lines", current: 6, suggested: "≤ 3", impact: 0.04 + rand() * 0.03 },
      { feature: "Salary credit regularity", current: "62%", suggested: "≥ 90%", impact: 0.03 + rand() * 0.02 },
    ],
  };
}

function mockNarrative(id: string): NarrativeResp {
  const a = mockApplicant(id);
  return {
    narrative:
`CREDIT MEMO — ${id}

Summary
Applicant presents a ${a.risk_label.toLowerCase()}-risk profile with a model-estimated 12-month default probability of ${(a.default_probability * 100).toFixed(1)}%. The decision is driven primarily by bureau performance, debt-service capacity, and observable cash-flow regularity from UPI and salary channels.

Key Drivers
• Bureau score and prior credit performance remain the dominant predictors, contributing the largest share of the model's score.
• Debt-to-income ratio is ${a.risk_label === "High" ? "elevated relative to portfolio norms" : "within acceptable bounds"}, materially ${a.risk_label === "High" ? "increasing" : "tempering"} expected loss.
• Alternative-data signals (UPI transaction volume, salary credit regularity) ${a.risk_label === "Low" ? "reinforce" : "partially offset"} the bureau view, supporting fair assessment for thin-file segments.

Recommendation
${a.risk_label === "High"
  ? "Decline at requested terms. Counter-offer with reduced principal, shorter tenor, or co-applicant. Re-underwrite after 6 months of clean repayment history on existing lines."
  : a.risk_label === "Medium"
  ? "Approve with risk-based pricing (+150–250 bps) and standard collateral/guarantor requirements. Monitor DPD bucket monthly for the first two cycles."
  : "Approve at standard pricing. Eligible for accelerated disbursal and cross-sell of unsecured top-up within 90 days subject to bureau refresh."}

Fairness Notes
This score has been validated against the portfolio's demographic-parity audit (gender, region, age band). No protected-attribute drift was detected for this applicant's cohort in the most recent monthly review.`,
  };
}

export const api = {
  overview: () => get<OverviewResp>("/api/dashboard/overview", MOCK_OVERVIEW),
  featureImportance: () => get<FeatureImportance>("/api/dashboard/feature-importance", MOCK_FEATURES),
  applicantsList: () => get<ApplicantList>("/api/applicants/list", { ids: MOCK_IDS }),
  applicant: (id: string) => get<Applicant>(`/api/applicants/${id}`, mockApplicant(id)),
  narrative: (id: string) => post<NarrativeResp>("/api/explain/narrative", { applicant_id: id }, mockNarrative(id)),
};
