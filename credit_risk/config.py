from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURE_DATA_DIR = DATA_DIR / "features"

RAW_BORROWERS_PATH = RAW_DATA_DIR / "borrowers_raw.csv"
CLEAN_BORROWERS_PATH = PROCESSED_DATA_DIR / "borrowers_clean.csv"
PIPELINE_REPORT_PATH = PROCESSED_DATA_DIR / "pipeline_report.json"
FEATURE_MATRIX_PATH = FEATURE_DATA_DIR / "feature_matrix.csv"
FEATURE_MATRIX_META_PATH = FEATURE_DATA_DIR / "feature_matrix_meta.json"
FEATURE_REPORT_PATH = FEATURE_DATA_DIR / "feature_report.json"
MODEL_DIR = PROJECT_ROOT / "models"
SAVED_MODEL_DIR = MODEL_DIR / "saved"
REGRESSION_MODEL_PATH = SAVED_MODEL_DIR / "regression_model.pkl"
CLASSIFICATION_MODEL_PATH = SAVED_MODEL_DIR / "classification_model.pkl"
LABEL_ENCODER_PATH = SAVED_MODEL_DIR / "label_encoder.pkl"
MODEL_REPORT_PATH = SAVED_MODEL_DIR / "model_report.json"
TEST_INDICES_PATH = SAVED_MODEL_DIR / "test_indices.npy"
EXPLAINABILITY_DIR = PROJECT_ROOT / "explainability"
EXPLAINABILITY_OUTPUT_DIR = EXPLAINABILITY_DIR / "outputs"
SAMPLE_EXPLANATIONS_DIR = EXPLAINABILITY_OUTPUT_DIR / "sample_explanations"
GLOBAL_SHAP_REPORT_PATH = EXPLAINABILITY_OUTPUT_DIR / "global_shap_report.json"
XAI_SUMMARY_REPORT_PATH = EXPLAINABILITY_OUTPUT_DIR / "xai_summary_report.json"

N_RECORDS = 5000
RANDOM_SEED = 42

CITY_TIERS = ["Tier1", "Tier2", "Tier3"]
EMPLOYMENT_TYPES = ["Salaried", "Self-employed", "Gig", "MSME"]
LOAN_TENURES_MONTHS = [6, 12, 18, 24, 36, 48, 60]
LOAN_PURPOSES = ["Business", "Education", "Medical", "Home", "Personal"]
RISK_LABELS = ["Low", "Medium", "High"]

RATIO_COLUMNS = [
    "emi_to_income_ratio",
    "disposable_income_ratio",
    "loan_to_income_ratio",
    "burden_to_income_ratio",
    "upi_to_income_ratio",
    "upi_engagement_score",
    "savings_rate",
    "expense_to_income_ratio",
    "age_risk_band",
]

SCALE_COLUMNS = [
    "monthly_income_inr",
    "disposable_income_inr",
    "monthly_loan_burden_inr",
    "loan_amount_requested",
    "upi_inflow_avg_inr",
    "alt_credit_score",
]

FEATURE_MATRIX_COLUMNS = [
    "applicant_id",
    "age",
    "existing_loans_count",
    "loan_tenure_months",
    "ever_defaulted",
    "is_thin_file",
    "emi_to_income_ratio",
    "disposable_income_ratio",
    "loan_to_income_ratio",
    "burden_to_income_ratio",
    "upi_to_income_ratio",
    "upi_engagement_score",
    "savings_rate",
    "expense_to_income_ratio",
    "age_risk_band",
    "alt_credit_score",
    "thin_file_confidence",
    "monthly_income_inr_scaled",
    "disposable_income_inr_scaled",
    "monthly_loan_burden_inr_scaled",
    "loan_amount_requested_scaled",
    "upi_inflow_avg_inr_scaled",
    "alt_credit_score_scaled",
    "emp_Salaried",
    "emp_SelfEmployed",
    "emp_Gig",
    "emp_MSME",
    "city_tier_ord",
    "purpose_Business",
    "purpose_Education",
    "purpose_Medical",
    "purpose_Home",
    "purpose_Personal",
    "default_probability",
    "risk_label",
]

MODEL_EXCLUDE_COLUMNS = [
    "applicant_id",
    "default_probability",
    "risk_label",
    "alt_credit_score_scaled",
]

MODEL_RANDOM_STATE = 42
TEST_SIZE = 0.2

FEATURE_GROUPS = {
    "Credit signals": ["alt_credit_score", "ever_defaulted", "existing_loans_count"],
    "Income & capacity": [
        "monthly_income_inr_scaled",
        "disposable_income_ratio",
        "emi_to_income_ratio",
        "savings_rate",
        "disposable_income_inr_scaled",
    ],
    "Loan stress": [
        "loan_to_income_ratio",
        "burden_to_income_ratio",
        "monthly_loan_burden_inr_scaled",
        "loan_amount_requested_scaled",
    ],
    "UPI behavior": [
        "upi_engagement_score",
        "upi_to_income_ratio",
        "upi_inflow_avg_inr_scaled",
    ],
    "Employment profile": [
        "emp_Salaried",
        "emp_SelfEmployed",
        "emp_Gig",
        "emp_MSME",
        "age_risk_band",
        "age",
    ],
    "Thin-file signals": ["thin_file_confidence", "is_thin_file"],
    "Loan structure": [
        "loan_tenure_months",
        "purpose_Business",
        "purpose_Education",
        "purpose_Medical",
        "purpose_Home",
        "purpose_Personal",
    ],
    "Geography": ["city_tier_ord"],
}

SCHEMA_COLUMNS = [
    "applicant_id",
    "age",
    "city_tier",
    "employment_type",
    "monthly_income_inr",
    "income_volatility",
    "upi_inflow_avg_inr",
    "upi_transaction_count",
    "monthly_expense_inr",
    "emi_obligations_inr",
    "credit_bureau_score",
    "existing_loans_count",
    "ever_defaulted",
    "gst_filing_months",
    "avg_monthly_gst_inr",
    "loan_amount_requested",
    "loan_tenure_months",
    "loan_purpose",
    "default_probability",
    "risk_label",
]
