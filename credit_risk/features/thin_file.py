import numpy as np
import pandas as pd
from loguru import logger


def add_thin_file_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df.copy()
    thin_mask = features["is_thin_file"].astype(bool)

    positive_savings_component = np.where(
        features["savings_rate"] > 0,
        features["savings_rate"] * 100,
        0.0,
    )
    gst_component = np.where(
        features["gst_filing_months"].notna(),
        features["gst_filing_months"].fillna(0) / 24 * 80,
        0.0,
    )
    alt_scores = (
        500
        + (features["upi_engagement_score"] * 150)
        + positive_savings_component
        - (features["income_volatility"] * 120)
        + gst_component
        - (features["emi_to_income_ratio"] * 100)
        - np.where(features["ever_defaulted"].astype(bool), 50, 0)
    )

    features["alt_credit_score"] = features["credit_bureau_score"]
    features.loc[thin_mask, "alt_credit_score"] = alt_scores[thin_mask]
    features["alt_credit_score"] = features["alt_credit_score"].clip(300, 900).astype(float)

    confidence = (
        0.3
        + np.where(features["upi_transaction_count"] > 100, 0.2, 0.0)
        + np.where(
            features["gst_filing_months"].notna() & (features["gst_filing_months"] > 12),
            0.2,
            0.0,
        )
        + np.where(features["employment_type"].isin(["MSME", "Self-employed"]), 0.15, 0.0)
        + np.where(features["existing_loans_count"] > 0, 0.15, 0.0)
    )
    features["thin_file_confidence"] = 1.0
    features.loc[thin_mask, "thin_file_confidence"] = confidence[thin_mask]
    features["thin_file_confidence"] = features["thin_file_confidence"].clip(0.0, 1.0).astype(float)

    logger.info(
        "Added alternate credit features for {} thin-file and {} bureau-backed records.",
        int(thin_mask.sum()),
        int((~thin_mask).sum()),
    )
    return features
