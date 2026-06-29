import numpy as np
import pandas as pd

from explainability.shap_local import human_readable


def _risk_label(probability: float) -> str:
    if probability < 0.2:
        return "Low"
    if probability <= 0.5:
        return "Medium"
    return "High"


def _feasibility(change_fraction: float) -> str:
    if change_fraction < 0.1:
        return "easy"
    if change_fraction <= 0.25:
        return "medium"
    return "hard"


def _candidate_value(feature: str, current: float, fraction: float) -> float:
    if feature in {"emi_to_income_ratio", "burden_to_income_ratio", "loan_amount_requested_scaled"}:
        return current * (1 - fraction)
    if feature == "upi_engagement_score":
        return min(1.0, current + (1.0 - current) * fraction)
    if feature == "savings_rate":
        return min(1.0, current + (1.0 - current) * fraction)
    if feature == "loan_tenure_months":
        allowed = np.array([6, 12, 18, 24, 36, 48, 60])
        longer = allowed[allowed > current]
        if len(longer) == 0:
            return current
        step_index = min(len(longer) - 1, int(np.ceil(fraction * 3)) - 1)
        return float(longer[step_index])
    return current


def _action(feature: str) -> str:
    return {
        "emi_to_income_ratio": "Reduce EMI obligations",
        "upi_engagement_score": "Increase UPI transaction activity",
        "savings_rate": "Improve monthly savings rate",
        "loan_amount_requested_scaled": "Request a smaller loan amount",
        "loan_tenure_months": "Choose a longer repayment tenure",
        "burden_to_income_ratio": "Reduce projected monthly loan burden",
    }[feature]


def generate_counterfactuals(
    applicant_id: str,
    applicant_features: pd.Series,
    regression_model,
    current_prob: float,
    scaler_params: dict,
) -> list[dict]:
    feature_cols = list(regression_model.feature_names_in_)
    actionable = [
        "emi_to_income_ratio",
        "upi_engagement_score",
        "savings_rate",
        "loan_amount_requested_scaled",
        "loan_tenure_months",
        "burden_to_income_ratio",
    ]
    rows = []
    for feature in actionable:
        current_value = float(applicant_features[feature])
        for fraction in [0.10, 0.20, 0.30]:
            suggested_value = _candidate_value(feature, current_value, fraction)
            if suggested_value == current_value:
                continue
            modified = applicant_features[feature_cols].copy().astype(float)
            modified[feature] = suggested_value
            new_prob = float(np.clip(regression_model.predict(modified.to_frame().T.astype(float))[0], 0.0, 1.0))
            reduction = current_prob - new_prob
            if reduction > 0.02:
                rows.append(
                    {
                        "action": _action(feature),
                        "feature": feature,
                        "current_value_human": human_readable(feature, current_value, scaler_params),
                        "suggested_value_human": human_readable(feature, suggested_value, scaler_params),
                        "estimated_probability_reduction": round(float(reduction), 6),
                        "new_predicted_probability": round(float(new_prob), 6),
                        "new_risk_label": _risk_label(new_prob),
                        "feasibility": _feasibility(fraction),
                    }
                )

    rows.sort(key=lambda row: row["estimated_probability_reduction"], reverse=True)
    if not rows:
        return [
            {
                "action": "No simple improvements identified",
                "note": "Risk profile driven by non-actionable factors (prior default, employment type)",
            }
        ]
    return rows[:3]
