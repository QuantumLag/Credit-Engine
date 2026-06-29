import sys
from pathlib import Path

import pandas as pd
from loguru import logger
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import SCALE_COLUMNS  # noqa: E402


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    features = df.copy()
    features["emp_Salaried"] = (features["employment_type"] == "Salaried").astype(int)
    features["emp_SelfEmployed"] = (features["employment_type"] == "Self-employed").astype(int)
    features["emp_Gig"] = (features["employment_type"] == "Gig").astype(int)
    features["emp_MSME"] = (features["employment_type"] == "MSME").astype(int)

    features["city_tier_ord"] = features["city_tier"].map({"Tier1": 3, "Tier2": 2, "Tier3": 1}).astype(int)

    for purpose in ["Business", "Education", "Medical", "Home", "Personal"]:
        features[f"purpose_{purpose}"] = (features["loan_purpose"] == purpose).astype(int)

    features["ever_defaulted"] = features["ever_defaulted"].astype(int)
    features["is_thin_file"] = features["is_thin_file"].astype(int)
    logger.info("Encoded categorical, boolean, and ordinal fields.")
    return features


def scale_numeric_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    features = df.copy()
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(features[SCALE_COLUMNS])

    scaler_params = {}
    for index, column in enumerate(SCALE_COLUMNS):
        scaled_column = f"{column}_scaled"
        features[scaled_column] = scaled_values[:, index]
        scaler_params[column] = {
            "mean": float(scaler.mean_[index]),
            "scale": float(scaler.scale_[index]),
        }

    logger.info("Scaled {} numerical columns with StandardScaler.", len(SCALE_COLUMNS))
    return features, scaler_params
