import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import FEATURE_GROUPS, MODEL_EXCLUDE_COLUMNS  # noqa: E402
from explainability.explanation_schema import ApplicantExplanation  # noqa: E402


REGRESSION_MODEL = None
CLASSIFICATION_MODEL = None


def set_prediction_models(regression_model, classification_model) -> None:
    global REGRESSION_MODEL, CLASSIFICATION_MODEL
    REGRESSION_MODEL = regression_model
    CLASSIFICATION_MODEL = classification_model


def get_feature_cols(feature_matrix: pd.DataFrame) -> list[str]:
    return [column for column in feature_matrix.columns if column not in MODEL_EXCLUDE_COLUMNS]


def normalize_classification_shap(shap_values_clf) -> np.ndarray:
    values = np.asarray(shap_values_clf)
    if values.ndim == 3 and values.shape[0] == 3:
        return values
    if values.ndim == 3 and values.shape[2] == 3:
        return np.moveaxis(values, 2, 0)
    raise ValueError(f"Unexpected classification SHAP shape: {values.shape}")


def magnitude(shap_value: float) -> str:
    value = abs(shap_value)
    if value > 0.05:
        return "high"
    if value >= 0.02:
        return "medium"
    return "low"


def inverse_scaled(feature: str, value: float, scaler_params: dict) -> float:
    source_feature = feature.replace("_scaled", "")
    params = scaler_params[source_feature]
    return value * params["scale"] + params["mean"]


def human_readable(feature: str, actual_value: float, scaler_params: dict) -> str:
    if feature == "emi_to_income_ratio":
        return f"EMI burden is {actual_value * 100:.0f}% of income"
    if feature == "disposable_income_ratio":
        return f"Disposable income is {actual_value * 100:.0f}% of monthly income"
    if feature == "loan_to_income_ratio":
        return f"Loan is {actual_value:.1f}x monthly income"
    if feature == "burden_to_income_ratio":
        return f"Monthly loan repayment would be {actual_value * 100:.0f}% of income"
    if feature == "upi_engagement_score":
        return f"UPI activity: {actual_value * 300:.0f} transactions/month"
    if feature == "savings_rate":
        return f"Saves {actual_value * 100:.0f}% of income monthly"
    if feature == "alt_credit_score":
        return f"Credit score equivalent: {actual_value:.0f}"
    if feature == "ever_defaulted":
        return "Has previously defaulted on a loan" if actual_value == 1 else "No prior loan defaults"
    if feature == "emp_Gig" and actual_value == 1:
        return "Employment type: Gig worker"
    if feature == "emp_MSME" and actual_value == 1:
        return "Employment type: MSME owner"
    if feature == "emp_Salaried" and actual_value == 1:
        return "Employment type: Salaried"
    if feature == "emp_SelfEmployed" and actual_value == 1:
        return "Employment type: Self-employed"
    if feature == "income_volatility":
        return f"Income varies by {actual_value * 100:.0f}% month to month"
    if feature == "thin_file_confidence":
        return f"Alternate data confidence: {actual_value * 100:.0f}%"
    if feature == "monthly_income_inr_scaled":
        return f"Monthly income: INR {inverse_scaled(feature, actual_value, scaler_params):.0f}"
    if feature == "age":
        return f"Age: {actual_value:.0f} years"
    if feature == "existing_loans_count":
        return f"Has {actual_value:.0f} existing loan(s)"
    if feature == "loan_tenure_months":
        return f"Requested tenure: {actual_value:.0f} months"
    if feature == "loan_amount_requested_scaled":
        return f"Requested loan amount: INR {inverse_scaled(feature, actual_value, scaler_params):.0f}"
    if feature == "monthly_loan_burden_inr_scaled":
        return f"Estimated monthly loan burden: INR {inverse_scaled(feature, actual_value, scaler_params):.0f}"
    if feature == "disposable_income_inr_scaled":
        return f"Disposable income: INR {inverse_scaled(feature, actual_value, scaler_params):.0f}"
    if feature == "upi_inflow_avg_inr_scaled":
        return f"Average UPI inflow: INR {inverse_scaled(feature, actual_value, scaler_params):.0f}"
    return f"{feature}: {actual_value:.4f}"


def _top_contributions(
    row: pd.Series,
    feature_cols: list[str],
    shap_row: np.ndarray,
    direction: str,
    scaler_params: dict,
) -> list[dict]:
    indexed = list(zip(feature_cols, shap_row))
    if direction == "increases_risk":
        candidates = sorted(indexed, key=lambda item: item[1], reverse=True)
    else:
        candidates = sorted(indexed, key=lambda item: item[1])

    rows = []
    for feature, shap_value in candidates[:5]:
        actual_value = float(row[feature])
        rows.append(
            {
                "feature": feature,
                "shap_value": round(float(shap_value), 6),
                "actual_value": round(actual_value, 6),
                "human_readable": human_readable(feature, actual_value, scaler_params),
                "direction": direction,
                "magnitude": magnitude(float(shap_value)),
            }
        )
    return rows


def _group_contributions(feature_cols: list[str], shap_row: np.ndarray) -> dict[str, float]:
    shap_by_feature = dict(zip(feature_cols, shap_row))
    return {
        group: round(float(sum(shap_by_feature.get(feature, 0.0) for feature in members)), 6)
        for group, members in FEATURE_GROUPS.items()
    }


def explain_applicant(
    applicant_id: str,
    feature_matrix: pd.DataFrame,
    shap_values_reg: np.ndarray,
    shap_values_clf: np.ndarray,
    reg_explainer,
    clf_explainer,
    label_encoder,
    scaler_params: dict,
) -> dict:
    if REGRESSION_MODEL is None or CLASSIFICATION_MODEL is None:
        raise RuntimeError("Prediction models must be set with set_prediction_models before explaining applicants.")

    feature_cols = get_feature_cols(feature_matrix)
    row_matches = feature_matrix.index[feature_matrix["applicant_id"] == applicant_id].tolist()
    if not row_matches:
        raise ValueError(f"Applicant not found: {applicant_id}")
    local_position = row_matches[0]

    row = feature_matrix.iloc[local_position]
    X_row = row[feature_cols].to_frame().T.astype(float)
    predicted_prob = float(np.clip(REGRESSION_MODEL.predict(X_row)[0], 0.0, 1.0))
    class_probabilities_array = CLASSIFICATION_MODEL.predict_proba(X_row)[0]
    predicted_label = str(label_encoder.inverse_transform([int(np.argmax(class_probabilities_array))])[0])
    class_probabilities = {
        str(label): round(float(class_probabilities_array[index]), 6)
        for index, label in enumerate(label_encoder.classes_)
    }

    shap_row = shap_values_reg[local_position]
    alt_score_shap = float(shap_row[feature_cols.index("alt_credit_score")])
    thin_file_explanation = None
    if bool(row["is_thin_file"]):
        thin_file_explanation = {
            "alt_credit_score": round(float(row["alt_credit_score"]), 4),
            "thin_file_confidence": round(float(row["thin_file_confidence"]), 4),
            "alt_score_shap": round(alt_score_shap, 6),
            "note": (
                "Bureau score unavailable. Assessment based on UPI behavior, "
                "cash-flow patterns, and GST compliance."
            ),
        }

    explanation = {
        "applicant_id": applicant_id,
        "is_thin_file": bool(row["is_thin_file"]),
        "predicted_default_probability": round(predicted_prob, 6),
        "predicted_risk_label": predicted_label,
        "class_probabilities": class_probabilities,
        "baseline_default_probability": round(float(np.asarray(reg_explainer.expected_value).reshape(-1)[0]), 6),
        "top_risk_drivers": _top_contributions(row, feature_cols, shap_row, "increases_risk", scaler_params),
        "top_risk_mitigants": _top_contributions(row, feature_cols, shap_row, "decreases_risk", scaler_params),
        "thin_file_explanation": thin_file_explanation,
        "feature_group_contributions": _group_contributions(feature_cols, shap_row),
    }
    return ApplicantExplanation.model_validate(explanation).model_dump()
