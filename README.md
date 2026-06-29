# CreditLens — AI-Powered Credit Risk Assessment

An end-to-end credit risk intelligence platform for Indian fintech lenders,
built to assess thin-file borrowers (gig workers, MSMEs) who lack traditional
credit bureau history.

## The Problem

35% of loan applicants in India have no credit bureau score. Traditional models
reject them outright. CreditLens uses cash-flow patterns, UPI transaction
behavior, and GST compliance data to assess creditworthiness — and explains
every decision in plain English.

## Results

| Metric | Value |
|---|---|
| Regression R² | 0.8857 |
| RMSE | 0.0583 |
| Classification Accuracy | 86.3% |
| AUC-ROC | 0.9646 |
| Within-0.1 Accuracy | 90.4% |
| Thin-file Fairness Flag | ✓ Not detected |

## Architecture

```
Data Pipeline → Feature Engineering → ML Model → XAI Layer → Dashboard + LLM Narrative
  (Module 1)       (Module 2)          (Module 3)  (Module 4)     (Module 5 + 6)
```

**Module 1 — Data Pipeline**
Synthetic dataset of 5,000 Indian borrowers. Realistic income distributions,
UPI patterns, GST filings, bureau scores (NaN for 35.6% — thin-file segment).

**Module 2 — Feature Engineering**
Derives 32 model-input features: EMI-to-income ratio, disposable income,
loan stress metrics, UPI engagement score, savings rate. Thin-file borrowers
get an alternate credit score derived from behavioral signals instead of
bureau imputation.

**Module 3 — ML Model**
XGBoost regression (default probability) + classification (Low/Medium/High).
SMOTE applied to handle severe class imbalance (Low: 148 samples).

**Module 4 — XAI / SHAP Layer**
TreeSHAP explanations for every prediction. Global feature importance,
per-applicant waterfall breakdowns, counterfactual engine ("what would
improve this score"), thin-file specific alternate explanations.

**Module 5 — FastAPI Backend + React Dashboard**
REST API serving model predictions, SHAP explanations, and applicant data.
Premium React frontend with risk gauge, SHAP waterfall chart, counterfactual
cards, and portfolio overview with feature importance visualization.

**Module 6 — LLM Narrative Engine**
Claude AI generates plain-English credit memos from SHAP output.
Structured as: Credit Summary → Risk Factors → Mitigating Strengths →
Thin-file Assessment → Recommended Actions → Analyst Recommendation.

## Key Features

- **Thin-file assessment** — 1,778 borrowers assessed without bureau scores
- **Explainability** — every score explained via SHAP values
- **Counterfactuals** — actionable suggestions to improve risk profile
- **Fairness validated** — thin-file borrowers assessed as accurately as bureau-backed
- **LLM credit memos** — Claude generates analyst-grade narratives per applicant

## Tech Stack

| Layer | Technology |
|---|---|
| ML Models | XGBoost, scikit-learn, imbalanced-learn |
| Explainability | SHAP (TreeExplainer) |
| Backend | FastAPI, uvicorn |
| Frontend | React, Vite, Recharts |
| LLM | Anthropic Claude (claude-sonnet-4-6) |
| Data | pandas, numpy, faker |
| Validation | Pydantic, loguru |

## Project Structure

```
credit_risk/
├── pipeline/          # Module 1: data generation + cleaning
├── features/          # Module 2: feature engineering
├── models/            # Module 3: XGBoost training
│   └── saved/         # trained model .pkl files
├── explainability/    # Module 4: SHAP + counterfactuals
│   └── outputs/       # SHAP reports + 10 sample explanations
├── app/               # Module 5+6: FastAPI backend + LLM narrative
└── frontend/          # React dashboard (Lovable)
```

## Setup

### Prerequisites

- Python 3.10+
- Node.js 18+
- Anthropic API key (optional — mock narrative used if absent)

### Install

```bash
# Python dependencies
pip install -r requirements.txt
pip install -r requirements_app.txt

# Frontend dependencies
cd credit_risk/frontend
npm install
```

### Environment

Create `credit_risk/.env`:
```
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### Reproduce the ML pipeline

```bash
python pipeline/run_pipeline.py                  # Module 1
python features/run_features.py                  # Module 2
python models/run_models.py                      # Module 3
python explainability/run_explainability.py      # Module 4
```

### Run the application

Terminal 1 — Backend:
```bash
cd credit_risk
python run.py
```

Terminal 2 — Frontend:
```bash
cd credit_risk/frontend
npm run dev
```

Open **http://localhost:5173**

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/dashboard/overview` | Portfolio stats + model performance |
| GET | `/api/dashboard/feature-importance` | Global SHAP importance |
| GET | `/api/dashboard/thin-file` | Thin-file segment analysis |
| GET | `/api/applicants/list` | All sample applicant IDs |
| GET | `/api/applicants/{id}` | Full SHAP explanation for applicant |
| POST | `/api/explain/narrative` | Generate Claude credit memo |
| GET | `/api/explain/counterfactuals/{id}` | Actionable improvement suggestions |

## Live Demo

Frontend: [lens-credit-insight.lovable.app](https://lens-credit-insight.lovable.app)

> Note: AI narrative generation requires the backend running locally with a valid API key.

## Dataset

Synthetic data generated to reflect Indian fintech market characteristics.
Not based on real borrower data. For research and demonstration purposes only.

## License

MIT
