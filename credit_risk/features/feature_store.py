import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import (  # noqa: E402
    FEATURE_MATRIX_COLUMNS,
    FEATURE_MATRIX_META_PATH,
    FEATURE_MATRIX_PATH,
    FEATURE_REPORT_PATH,
    RATIO_COLUMNS,
)


MODEL_EXCLUDE_COLUMNS = {"applicant_id", "default_probability", "risk_label"}


def _replace_infinite_values(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    features = df.copy()
    for column in columns:
        finite_values = features.loc[np.isfinite(features[column]), column]
        max_value = finite_values.max()
        min_value = finite_values.min()
        pos_inf_count = int(np.isposinf(features[column]).sum())
        neg_inf_count = int(np.isneginf(features[column]).sum())
        features[column] = features[column].replace({np.inf: max_value, -np.inf: min_value})
        if pos_inf_count or neg_inf_count:
            logger.info(
                "Replaced {} +inf and {} -inf values in {}.",
                pos_inf_count,
                neg_inf_count,
                column,
            )
    return features


def assemble_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    features = _replace_infinite_values(df, RATIO_COLUMNS)
    missing_columns = sorted(set(FEATURE_MATRIX_COLUMNS) - set(features.columns))
    if missing_columns:
        raise ValueError(f"Missing required feature matrix columns: {missing_columns}")

    matrix = features[FEATURE_MATRIX_COLUMNS].copy()
    logger.info("Assembled feature matrix with shape {}.", matrix.shape)
    return matrix


def validate_feature_matrix(matrix: pd.DataFrame, expected_rows: int = 5000) -> None:
    if len(matrix) != expected_rows:
        raise ValueError(f"feature_matrix.csv must have exactly {expected_rows} rows; found {len(matrix)}.")

    model_columns = [column for column in matrix.columns if column not in MODEL_EXCLUDE_COLUMNS]
    nan_counts = matrix[model_columns].isna().sum()
    nan_counts = nan_counts[nan_counts > 0]
    if not nan_counts.empty:
        raise ValueError(f"NaN values found in model input columns: {nan_counts.to_dict()}")

    if not matrix["alt_credit_score"].between(300, 900).all():
        raise ValueError("alt_credit_score must be in [300, 900] for all rows.")

    if not matrix["thin_file_confidence"].between(0.0, 1.0).all():
        raise ValueError("thin_file_confidence must be in [0.0, 1.0] for all rows.")

    for column in RATIO_COLUMNS:
        if not np.isfinite(matrix[column]).all():
            raise ValueError(f"Ratio column {column} contains non-finite values.")

    logger.info("Feature matrix validation passed.")


def build_feature_report(
    feature_source: pd.DataFrame,
    matrix: pd.DataFrame,
    scaler_params: dict,
) -> dict:
    thin_segment = feature_source[feature_source["is_thin_file"].astype(bool)]
    non_thin_segment = feature_source[~feature_source["is_thin_file"].astype(bool)]
    model_columns = [column for column in matrix.columns if column not in MODEL_EXCLUDE_COLUMNS]

    numeric_features = matrix[model_columns].select_dtypes(include=[np.number])
    correlations = (
        numeric_features.corrwith(matrix["default_probability"])
        .dropna()
        .abs()
        .sort_values(ascending=False)
        .head(5)
    )
    signed_correlations = numeric_features.corrwith(matrix["default_probability"])
    top_correlated = [
        {"feature": feature, "correlation": round(float(signed_correlations[feature]), 4)}
        for feature in correlations.index
    ]

    label_counts = matrix["risk_label"].value_counts()
    class_note = (
        f"High:{int(label_counts.get('High', 0))} "
        f"Medium:{int(label_counts.get('Medium', 0))} "
        f"Low:{int(label_counts.get('Low', 0))} - Module 3 must use class_weight or SMOTE"
    )

    return {
        "total_features": int(len(matrix.columns)),
        "model_input_features": int(len(model_columns)),
        "thin_file_segment": {
            "count": int(len(thin_segment)),
            "mean_alt_credit_score": round(float(thin_segment["alt_credit_score"].mean()), 4),
            "mean_thin_file_confidence": round(float(thin_segment["thin_file_confidence"].mean()), 4),
        },
        "non_thin_file_segment": {
            "count": int(len(non_thin_segment)),
            "mean_bureau_score": round(float(non_thin_segment["credit_bureau_score"].mean()), 4),
        },
        "top_correlated_with_default": top_correlated,
        "class_imbalance_note": class_note,
        "scaler_params": scaler_params,
    }


def build_feature_meta(matrix: pd.DataFrame, scaler_params: dict) -> dict:
    model_columns = [column for column in matrix.columns if column not in MODEL_EXCLUDE_COLUMNS]
    return {
        "columns": list(matrix.columns),
        "column_types": {column: str(dtype) for column, dtype in matrix.dtypes.items()},
        "model_input_columns": model_columns,
        "target_columns": ["default_probability", "risk_label"],
        "excluded_from_model": ["applicant_id"],
        "scaler_params": scaler_params,
    }


def save_feature_outputs(
    matrix: pd.DataFrame,
    report: dict,
    meta: dict,
    matrix_path: Path = FEATURE_MATRIX_PATH,
    report_path: Path = FEATURE_REPORT_PATH,
    meta_path: Path = FEATURE_MATRIX_META_PATH,
) -> None:
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(matrix_path, index=False)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Saved feature matrix to {}.", matrix_path)
    logger.info("Saved feature report to {}.", report_path)
    logger.info("Saved feature metadata to {}.", meta_path)
