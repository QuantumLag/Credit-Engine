import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from loguru import logger

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import (  # noqa: E402
    CLASSIFICATION_MODEL_PATH,
    FEATURE_MATRIX_META_PATH,
    FEATURE_MATRIX_PATH,
    GLOBAL_SHAP_REPORT_PATH,
    LABEL_ENCODER_PATH,
    REGRESSION_MODEL_PATH,
    SAMPLE_EXPLANATIONS_DIR,
    TEST_INDICES_PATH,
    XAI_SUMMARY_REPORT_PATH,
)
from explainability.counterfactual import generate_counterfactuals  # noqa: E402
from explainability.explanation_schema import ApplicantExplanation  # noqa: E402
from explainability.shap_global import build_global_shap_report, feature_group_for  # noqa: E402
from explainability.shap_local import (  # noqa: E402
    explain_applicant,
    get_feature_cols,
    normalize_classification_shap,
    set_prediction_models,
)
from explainability.thin_file_xai import build_thin_file_xai  # noqa: E402


def _classification_high_values(shap_values_clf: np.ndarray) -> np.ndarray:
    return normalize_classification_shap(shap_values_clf)[2]


def _select_sample_applicants(test_frame: pd.DataFrame) -> list[str]:
    selected: list[str] = []

    def add_matches(mask, limit: int) -> None:
        for applicant_id in test_frame.loc[mask, "applicant_id"].tolist():
            if applicant_id not in selected:
                selected.append(applicant_id)
            if len([item for item in selected if item in test_frame.loc[mask, "applicant_id"].tolist()]) >= limit:
                break

    add_matches((test_frame["risk_label"] == "High") & (test_frame["is_thin_file"] == 1), 2)
    add_matches((test_frame["risk_label"] == "Medium") & (test_frame["is_thin_file"] == 1), 2)
    add_matches((test_frame["risk_label"] == "High") & (test_frame["is_thin_file"] == 0), 2)
    add_matches((test_frame["risk_label"] == "Medium") & (test_frame["is_thin_file"] == 0), 2)
    add_matches(test_frame["risk_label"] == "Low", 1)
    add_matches(test_frame["emp_Gig"] == 1, 1)

    for applicant_id in test_frame["applicant_id"].tolist():
        if len(selected) >= 10:
            break
        if applicant_id not in selected:
            selected.append(applicant_id)

    return selected[:10]


def _validate_shap_shapes(shap_values_reg: np.ndarray, shap_values_clf: np.ndarray, test_count: int, feature_count: int) -> None:
    if shap_values_reg.shape != (test_count, feature_count):
        raise ValueError(f"Unexpected regression SHAP shape: {shap_values_reg.shape}")
    clf_normalized = normalize_classification_shap(shap_values_clf)
    if clf_normalized.shape != (3, test_count, feature_count):
        raise ValueError(f"Unexpected classification SHAP shape: {clf_normalized.shape}")


def run_explainability_pipeline() -> dict:
    logger.info("Running Module 4 explainability pipeline.")
    regression_model = joblib.load(REGRESSION_MODEL_PATH)
    classification_model = joblib.load(CLASSIFICATION_MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    test_indices = np.load(TEST_INDICES_PATH)
    feature_matrix = pd.read_csv(FEATURE_MATRIX_PATH)
    meta = json.loads(FEATURE_MATRIX_META_PATH.read_text(encoding="utf-8"))
    scaler_params = meta["scaler_params"]
    feature_cols = get_feature_cols(feature_matrix)
    X_test = feature_matrix.iloc[test_indices][feature_cols]
    test_frame = feature_matrix.iloc[test_indices].reset_index(drop=True)

    set_prediction_models(regression_model, classification_model)
    reg_explainer = shap.TreeExplainer(regression_model)
    clf_explainer = shap.TreeExplainer(classification_model)
    shap_values_reg = np.asarray(reg_explainer.shap_values(X_test))
    shap_values_clf = clf_explainer.shap_values(X_test)
    shap_values_clf = normalize_classification_shap(shap_values_clf)
    _validate_shap_shapes(shap_values_reg, shap_values_clf, len(test_indices), len(feature_cols))
    logger.info("Computed SHAP values for {} test rows.", len(test_indices))

    global_report = build_global_shap_report(
        feature_cols,
        shap_values_reg,
        _classification_high_values(shap_values_clf),
        reg_explainer,
        clf_explainer,
    )
    GLOBAL_SHAP_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    GLOBAL_SHAP_REPORT_PATH.write_text(json.dumps(global_report, indent=2), encoding="utf-8")
    logger.info("Saved global SHAP report to {}.", GLOBAL_SHAP_REPORT_PATH)

    thin_file_report = build_thin_file_xai(X_test, shap_values_reg, regression_model, feature_cols)
    SAMPLE_EXPLANATIONS_DIR.mkdir(parents=True, exist_ok=True)
    sample_ids = _select_sample_applicants(test_frame)
    counterfactual_successes = 0

    for applicant_id in sample_ids:
        test_position = test_frame.index[test_frame["applicant_id"] == applicant_id][0]
        local_explanation = explain_applicant(
            applicant_id,
            test_frame,
            shap_values_reg,
            shap_values_clf,
            reg_explainer,
            clf_explainer,
            label_encoder,
            scaler_params,
        )
        applicant_features = test_frame.iloc[test_position][feature_cols]
        counterfactuals = generate_counterfactuals(
            applicant_id,
            applicant_features,
            regression_model,
            local_explanation["predicted_default_probability"],
            scaler_params,
        )
        for counterfactual in counterfactuals:
            if "new_predicted_probability" in counterfactual and counterfactual["new_predicted_probability"] >= local_explanation["predicted_default_probability"]:
                raise ValueError("Counterfactual worsened applicant score.")
        if any("estimated_probability_reduction" in row for row in counterfactuals):
            counterfactual_successes += 1
        merged = {**local_explanation, "counterfactuals": counterfactuals}
        ApplicantExplanation.model_validate(merged)
        output_path = SAMPLE_EXPLANATIONS_DIR / f"{applicant_id}.json"
        output_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")

    sample_files = list(SAMPLE_EXPLANATIONS_DIR.glob("*.json"))
    if len(sample_files) < 10:
        raise ValueError(f"Expected at least 10 sample explanations; found {len(sample_files)}.")

    top_3_global = [
        {
            "feature": row["feature"],
            "mean_abs_shap": row["mean_abs_shap"],
            "group": feature_group_for(row["feature"]),
        }
        for row in global_report["regression_global_importance"][:3]
    ]
    summary = {
        "global_shap": global_report,
        "thin_file_xai": thin_file_report,
        "sample_applicants_generated": len(sample_ids),
        "explanation_schema_version": "1.0",
        "top_3_global_risk_drivers": top_3_global,
        "counterfactual_coverage": round(counterfactual_successes / len(sample_ids), 4),
        "module_4_notes": (
            "Explanations ready for LLM narrative generation in Module 6. "
            "Thin-file borrowers have complete alternate explanations. "
            "Counterfactuals target only actionable features."
        ),
    }
    XAI_SUMMARY_REPORT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Saved XAI summary report to {}.", XAI_SUMMARY_REPORT_PATH)
    logger.info("Module 4 explainability pipeline finished successfully.")
    return summary


def main() -> None:
    summary = run_explainability_pipeline()
    print(json.dumps(summary, indent=2))
    print("Module 4 complete. Explanations ready for dashboard and LLM narrative.")


if __name__ == "__main__":
    main()
