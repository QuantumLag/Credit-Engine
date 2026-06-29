import json
import sys
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import (  # noqa: E402
    CLEAN_BORROWERS_PATH,
    EMPLOYMENT_TYPES,
    LOAN_PURPOSES,
    LOAN_TENURES_MONTHS,
    PIPELINE_REPORT_PATH,
    RAW_BORROWERS_PATH,
    SCHEMA_COLUMNS,
)


class BorrowerRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    applicant_id: str
    age: int = Field(ge=22, le=58)
    city_tier: Literal["Tier1", "Tier2", "Tier3"]
    employment_type: Literal["Salaried", "Self-employed", "Gig", "MSME"]
    monthly_income_inr: float = Field(ge=8000, le=150000)
    income_volatility: float = Field(ge=0.0, le=1.0)
    upi_inflow_avg_inr: float
    upi_transaction_count: int = Field(ge=10, le=300)
    monthly_expense_inr: float
    emi_obligations_inr: float
    credit_bureau_score: Optional[int] = Field(default=None, ge=300, le=900)
    existing_loans_count: int = Field(ge=0, le=4)
    ever_defaulted: bool
    gst_filing_months: Optional[int] = Field(default=None, ge=0, le=24)
    avg_monthly_gst_inr: Optional[float] = None
    loan_amount_requested: float = Field(ge=0)
    loan_tenure_months: Literal[6, 12, 18, 24, 36, 48, 60]
    loan_purpose: Literal["Business", "Education", "Medical", "Home", "Personal"]
    default_probability: float = Field(ge=0.0, le=1.0)
    risk_label: Literal["Low", "Medium", "High"]
    is_thin_file: bool

    @field_validator("credit_bureau_score", "gst_filing_months", "avg_monthly_gst_inr", mode="before")
    @classmethod
    def convert_nan_to_none(cls, value):
        if pd.isna(value):
            return None
        return value


def _validate_columns(df: pd.DataFrame) -> None:
    expected_columns = SCHEMA_COLUMNS + ["is_thin_file"]
    missing_columns = sorted(set(expected_columns) - set(df.columns))
    extra_columns = sorted(set(df.columns) - set(expected_columns))

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    if extra_columns:
        raise ValueError(f"Unexpected columns found: {extra_columns}")

    logger.info("Validated required column set: {} columns present.", len(expected_columns))


def validate_records(df: pd.DataFrame) -> None:
    _validate_columns(df)
    errors = []
    for index, record in enumerate(df.to_dict(orient="records")):
        try:
            BorrowerRecord.model_validate(record)
        except ValidationError as exc:
            errors.append({"row": index, "errors": exc.errors()})
            if len(errors) >= 5:
                break

    if errors:
        raise ValueError(f"Pydantic validation failed for borrower records: {errors}")

    logger.info("Pydantic validation passed for {} cleaned records.", len(df))


def clean_borrower_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting cleaning for {} raw records.", len(df))
    cleaned = df.copy()

    before_income_filter = len(cleaned)
    cleaned = cleaned[cleaned["monthly_income_inr"] >= 8000].copy()
    logger.info(
        "Dropped {} records because monthly_income_inr was below 8000.",
        before_income_filter - len(cleaned),
    )

    loan_cap = cleaned["monthly_income_inr"] * 20
    rows_to_cap = cleaned["loan_amount_requested"] > loan_cap
    cleaned.loc[rows_to_cap, "loan_amount_requested"] = loan_cap[rows_to_cap].round(2)
    logger.info(
        "Capped loan_amount_requested at 20x monthly_income_inr for {} records.",
        int(rows_to_cap.sum()),
    )

    cleaned["is_thin_file"] = cleaned["credit_bureau_score"].isna()
    logger.info(
        "Flagged {} records as thin-file because credit_bureau_score is missing.",
        int(cleaned["is_thin_file"].sum()),
    )

    cleaned = cleaned[SCHEMA_COLUMNS + ["is_thin_file"]]
    cleaned["credit_bureau_score"] = cleaned["credit_bureau_score"].astype("Int64")
    cleaned["gst_filing_months"] = cleaned["gst_filing_months"].astype("Int64")

    validate_records(cleaned)
    return cleaned


def build_pipeline_report(raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> dict:
    return {
        "total_records_generated": int(len(raw_df)),
        "records_after_cleaning": int(len(clean_df)),
        "thin_file_count": int(clean_df["is_thin_file"].sum()),
        "risk_label_distribution": {
            label: int(count)
            for label, count in clean_df["risk_label"].value_counts().sort_index().items()
        },
        "employment_type_distribution": {
            employment_type: int(clean_df["employment_type"].eq(employment_type).sum())
            for employment_type in EMPLOYMENT_TYPES
        },
        "mean_default_probability": round(float(clean_df["default_probability"].mean()), 4),
    }


def save_processed_outputs(
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    clean_output_path: Path = CLEAN_BORROWERS_PATH,
    report_output_path: Path = PIPELINE_REPORT_PATH,
) -> dict:
    clean_output_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(clean_output_path, index=False)
    logger.info("Saved cleaned borrower data to {}.", clean_output_path)

    report = build_pipeline_report(raw_df, clean_df)
    report_output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Saved pipeline report to {}.", report_output_path)
    return report


def main() -> None:
    raw_df = pd.read_csv(RAW_BORROWERS_PATH)
    clean_df = clean_borrower_data(raw_df)
    report = save_processed_outputs(raw_df, clean_df)
    logger.info(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
