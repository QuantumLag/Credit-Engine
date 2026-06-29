import numpy as np
import pandas as pd
from loguru import logger


def add_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    features = df.copy()
    income = features["monthly_income_inr"]

    features["emi_to_income_ratio"] = np.where(
        features["emi_obligations_inr"] == 0,
        0.0,
        features["emi_obligations_inr"] / income,
    )
    features["disposable_income_inr"] = (
        features["monthly_income_inr"]
        - features["monthly_expense_inr"]
        - features["emi_obligations_inr"]
    )
    features["disposable_income_ratio"] = features["disposable_income_inr"] / income
    features["loan_to_income_ratio"] = features["loan_amount_requested"] / income
    features["monthly_loan_burden_inr"] = (
        features["loan_amount_requested"] / features["loan_tenure_months"]
    )
    features["burden_to_income_ratio"] = features["monthly_loan_burden_inr"] / income
    features["upi_to_income_ratio"] = features["upi_inflow_avg_inr"] / income
    features["upi_engagement_score"] = features["upi_transaction_count"] / 300.0
    features["savings_rate"] = (
        (features["monthly_income_inr"] - features["monthly_expense_inr"]) / income
    ).clip(-1.0, 1.0)
    features["expense_to_income_ratio"] = features["monthly_expense_inr"] / income
    features["age_risk_band"] = np.select(
        [
            features["age"].between(22, 29),
            features["age"].between(30, 40),
            features["age"].between(41, 50),
            features["age"].between(51, 58),
        ],
        [3, 1, 2, 3],
        default=3,
    ).astype(int)

    logger.info("Added financial ratio and behavioral signal columns.")
    return features
