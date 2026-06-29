import sys
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker
from loguru import logger

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import (  # noqa: E402
    CITY_TIERS,
    EMPLOYMENT_TYPES,
    LOAN_PURPOSES,
    LOAN_TENURES_MONTHS,
    N_RECORDS,
    RANDOM_SEED,
    RAW_BORROWERS_PATH,
)


def _risk_label(default_probability: float) -> str:
    if default_probability < 0.2:
        return "Low"
    if default_probability <= 0.5:
        return "Medium"
    return "High"


def _default_probability(row: dict, rng: np.random.Generator) -> float:
    base_score = 0.5
    credit_score = row["credit_bureau_score"]

    if pd.isna(credit_score):
        base_score += 0.05
    elif credit_score > 750:
        base_score -= 0.20
    elif credit_score < 550:
        base_score += 0.20

    if row["income_volatility"] > 0.6:
        base_score += 0.15

    if row["emi_obligations_inr"] / row["monthly_income_inr"] > 0.4:
        base_score += 0.20

    if row["ever_defaulted"]:
        base_score += 0.25

    if row["upi_transaction_count"] > 150:
        base_score -= 0.10

    if row["employment_type"] == "Gig":
        base_score += 0.08

    gst_filing_months = row["gst_filing_months"]
    if not pd.isna(gst_filing_months) and gst_filing_months > 18:
        base_score -= 0.10

    noisy_score = base_score + rng.normal(0, 0.03)
    return float(np.clip(noisy_score, 0.05, 0.95))


def generate_borrower_data(n_records: int = N_RECORDS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    fake = Faker("en_IN")
    Faker.seed(seed)
    rng = np.random.default_rng(seed)

    monthly_income = np.clip(rng.lognormal(mean=np.log(35_000), sigma=0.65, size=n_records), 8_000, 150_000)
    employment_types = rng.choice(EMPLOYMENT_TYPES, size=n_records, p=[0.35, 0.25, 0.25, 0.15])
    existing_loans = rng.choice([0, 1, 2, 3, 4], size=n_records, p=[0.38, 0.30, 0.18, 0.10, 0.04])

    records = []
    for idx in range(n_records):
        income = float(monthly_income[idx])
        employment_type = str(employment_types[idx])
        loans_count = int(existing_loans[idx])
        has_existing_loan = loans_count > 0

        credit_score = int(rng.integers(300, 901))
        if rng.random() < 0.35:
            credit_score = np.nan

        if employment_type in {"Self-employed", "MSME"}:
            gst_filing_months = int(rng.integers(0, 25))
            avg_monthly_gst = float(max(0, income * rng.uniform(0.02, 0.18) + rng.normal(0, income * 0.01)))
        else:
            gst_filing_months = np.nan
            avg_monthly_gst = np.nan

        record = {
            "applicant_id": fake.uuid4(),
            "age": int(rng.integers(22, 59)),
            "city_tier": str(rng.choice(CITY_TIERS, p=[0.32, 0.43, 0.25])),
            "employment_type": employment_type,
            "monthly_income_inr": round(income, 2),
            "income_volatility": round(float(np.clip(rng.beta(2.2, 4.2), 0, 1)), 4),
            "upi_inflow_avg_inr": round(float(income * rng.uniform(0.85, 1.15) + rng.normal(0, income * 0.03)), 2),
            "upi_transaction_count": int(rng.integers(10, 301)),
            "monthly_expense_inr": round(float(income * rng.uniform(0.40, 0.80)), 2),
            "emi_obligations_inr": round(float(income * rng.uniform(0.02, 0.40)), 2) if has_existing_loan else 0.0,
            "credit_bureau_score": credit_score,
            "existing_loans_count": loans_count,
            "ever_defaulted": bool(rng.random() < (0.08 + 0.05 * loans_count)),
            "gst_filing_months": gst_filing_months,
            "avg_monthly_gst_inr": round(avg_monthly_gst, 2) if not pd.isna(avg_monthly_gst) else np.nan,
            "loan_amount_requested": round(float(rng.uniform(10_000, 1_500_000)), 2),
            "loan_tenure_months": int(rng.choice(LOAN_TENURES_MONTHS)),
            "loan_purpose": str(rng.choice(LOAN_PURPOSES, p=[0.30, 0.12, 0.16, 0.14, 0.28])),
        }
        record["default_probability"] = round(_default_probability(record, rng), 4)
        record["risk_label"] = _risk_label(record["default_probability"])
        records.append(record)

    df = pd.DataFrame(records)
    logger.info("Generated {} synthetic borrower records.", len(df))
    return df


def save_raw_data(df: pd.DataFrame, output_path: Path = RAW_BORROWERS_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved raw borrower data to {}.", output_path)
    return output_path


def main() -> None:
    save_raw_data(generate_borrower_data())


if __name__ == "__main__":
    main()
