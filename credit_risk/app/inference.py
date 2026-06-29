"""Load models and data once at startup; expose normalised API shapes."""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (  # noqa: E402
    CLASSIFICATION_MODEL_PATH,
    FEATURE_GROUPS,
    FEATURE_MATRIX_PATH,
    FEATURE_REPORT_PATH,
    GLOBAL_SHAP_REPORT_PATH,
    LABEL_ENCODER_PATH,
    MODEL_REPORT_PATH,
    REGRESSION_MODEL_PATH,
    SAMPLE_EXPLANATIONS_DIR,
    XAI_SUMMARY_REPORT_PATH,
)

# Map internal feature-group names → frontend category tokens
_GROUP_TO_CATEGORY: dict[str, str] = {
    "Credit signals": "bureau",
    "Income & capacity": "income",
    "Loan stress": "income",
    "UPI behavior": "behavioral",
    "Employment profile": "employment",
    "Thin-file signals": "behavioral",
    "Loan structure": "other",
    "Geography": "demographic",
}


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file missing: {path}")


def _load_json(path: Path) -> dict:
    _require_file(path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def feature_to_group(feature: str) -> str:
    for group, features in FEATURE_GROUPS.items():
        if feature in features:
            return group
    return "Others"


def feature_to_category(feature: str) -> str:
    group = feature_to_group(feature)
    return _GROUP_TO_CATEGORY.get(group, "other")


class AppState:
    def __init__(self) -> None:
        self.regression_model = None
        self.classification_model = None
        self.label_encoder = None
        self.feature_matrix: pd.DataFrame | None = None
        self.global_shap_report: dict = {}
        self.xai_summary_report: dict = {}
        self.model_report: dict = {}
        self.feature_report: dict = {}
        self.sample_explanations: dict[str, dict] = {}
        self.narrative_cache: dict[str, dict] = {}

    def load_all(self) -> None:
        _require_file(REGRESSION_MODEL_PATH)
        _require_file(CLASSIFICATION_MODEL_PATH)
        _require_file(LABEL_ENCODER_PATH)
        _require_file(FEATURE_MATRIX_PATH)
        _require_file(GLOBAL_SHAP_REPORT_PATH)
        _require_file(XAI_SUMMARY_REPORT_PATH)
        _require_file(MODEL_REPORT_PATH)
        _require_file(FEATURE_REPORT_PATH)
        _require_file(SAMPLE_EXPLANATIONS_DIR)

        self.regression_model = joblib.load(REGRESSION_MODEL_PATH)
        self.classification_model = joblib.load(CLASSIFICATION_MODEL_PATH)
        self.label_encoder = joblib.load(LABEL_ENCODER_PATH)
        logger.info("✓ Models loaded (regression + classification)")

        self.feature_matrix = pd.read_csv(FEATURE_MATRIX_PATH)
        logger.info(f"✓ Feature matrix loaded ({len(self.feature_matrix)} rows)")

        self.global_shap_report = _load_json(GLOBAL_SHAP_REPORT_PATH)
        self.xai_summary_report = _load_json(XAI_SUMMARY_REPORT_PATH)
        self.model_report = _load_json(MODEL_REPORT_PATH)
        self.feature_report = _load_json(FEATURE_REPORT_PATH)
        logger.info("✓ SHAP reports loaded")

        self.sample_explanations = {}
        for json_path in sorted(SAMPLE_EXPLANATIONS_DIR.glob("*.json")):
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            self.sample_explanations[data["applicant_id"]] = data

        logger.info(f"✓ Sample explanations loaded ({len(self.sample_explanations)} applicants)")

    @property
    def sample_ids(self) -> list[str]:
        return list(self.sample_explanations.keys())

    # ------------------------------------------------------------------
    # /api/dashboard/overview
    # Returns shape that matches the Lovable frontend's OverviewResp type
    # ------------------------------------------------------------------
    def get_overview(self) -> dict:
        df = self.feature_matrix
        risk_counts = df["risk_label"].value_counts().to_dict()
        total = len(df)
        thin_file_count = int(df["is_thin_file"].sum())
        non_thin_count = total - thin_file_count

        # Risk distribution — lowercase keys for frontend
        risk_distribution = {
            "high": int(risk_counts.get("High", 0)),
            "medium": int(risk_counts.get("Medium", 0)),
            "low": int(risk_counts.get("Low", 0)),
        }

        clf_metrics = self.model_report["classification_model"]["test_metrics"]
        fairness_flag_reg = self.model_report["regression_model"]["thin_file_metrics"]["fairness_flag"]
        fairness_flag_clf = self.model_report["classification_model"]["thin_file_metrics"]["fairness_flag"]

        # Thin-file vs bureau comparison block
        thin_df = df[df["is_thin_file"] == True]  # noqa: E712
        non_thin_df = df[df["is_thin_file"] == False]  # noqa: E712

        def _segment_stats(seg: pd.DataFrame) -> dict:
            return {
                "count": len(seg),
                "mean_risk": round(float(seg["default_probability"].mean()), 4) if len(seg) else 0.0,
                "default_rate": round(float((seg["risk_label"] == "High").mean()), 4) if len(seg) else 0.0,
                "avg_income": 0.0,  # income not in feature matrix (scaled only)
            }

        return {
            # Frontend field names
            "total_applications": total,
            "thin_file_count": thin_file_count,
            "thin_file_pct": round(thin_file_count / total * 100, 1),
            "mean_default_risk": round(float(df["default_probability"].mean()), 4),
            "fairness_verified": bool(fairness_flag_reg == "PASS" and fairness_flag_clf == "PASS"),
            "risk_distribution": risk_distribution,
            "thin_file_analysis": {
                "thin_file": _segment_stats(thin_df),
                "bureau": _segment_stats(non_thin_df),
            },
            # Additional model metrics (bonus, not required by frontend)
            "model_performance": {
                "classification_accuracy": clf_metrics["accuracy"],
                "auc_roc": clf_metrics["auc_roc_macro"],
            },
        }

    # ------------------------------------------------------------------
    # /api/dashboard/feature-importance
    # Returns shape: { features: [{name, importance, category}] }
    # ------------------------------------------------------------------
    def get_feature_importance(self) -> dict:
        report = self.global_shap_report
        # Use regression global importance as the primary signal (normalised to sum=1)
        raw = report["regression_global_importance"][:15]
        total_shap = sum(item["mean_abs_shap"] for item in raw) or 1.0
        features = [
            {
                "name": item["feature"],
                "importance": round(item["mean_abs_shap"] / total_shap, 4),
                "category": feature_to_category(item["feature"]),
            }
            for item in raw
        ]
        return {"features": features}

    # ------------------------------------------------------------------
    # /api/applicants/list  →  { ids: [...] }
    # ------------------------------------------------------------------
    def get_applicant_list(self) -> dict:
        return {"ids": sorted(self.sample_explanations.keys())}

    # ------------------------------------------------------------------
    # /api/applicants/{id}
    # Maps stored explanation JSON to the Applicant shape the frontend expects
    # ------------------------------------------------------------------
    def get_applicant_detail(self, applicant_id: str) -> dict:
        exp = self.sample_explanations[applicant_id]

        # SHAP values — combine risk drivers + mitigants into a flat list
        shap_values = []
        for driver in exp.get("top_risk_drivers", []):
            shap_values.append({
                "feature": driver["feature"],
                "value": driver["shap_value"],
                "feature_value": driver.get("human_readable", driver.get("actual_value")),
            })
        for mitigant in exp.get("top_risk_mitigants", []):
            shap_values.append({
                "feature": mitigant["feature"],
                "value": mitigant["shap_value"],
                "feature_value": mitigant.get("human_readable", mitigant.get("actual_value")),
            })

        # Class probabilities — lowercase keys
        raw_probs = exp.get("class_probabilities", {})
        class_probabilities = {
            "low": raw_probs.get("Low", 0.0),
            "medium": raw_probs.get("Medium", 0.0),
            "high": raw_probs.get("High", 0.0),
        }

        # Counterfactuals — map to frontend shape
        counterfactuals = []
        for cf in exp.get("counterfactuals", []):
            counterfactuals.append({
                "feature": cf.get("feature", ""),
                "current": cf.get("current_value_human", ""),
                "suggested": cf.get("suggested_value_human", ""),
                "impact": cf.get("estimated_probability_reduction", 0.0),
            })

        return {
            "applicant_id": exp["applicant_id"],
            "default_probability": exp["predicted_default_probability"],
            "risk_label": exp["predicted_risk_label"],
            "is_thin_file": exp.get("is_thin_file", False),
            "class_probabilities": class_probabilities,
            "shap_values": shap_values,
            "counterfactuals": counterfactuals,
            # Extra detail fields for the memo
            "feature_group_contributions": exp.get("feature_group_contributions", {}),
            "top_risk_drivers": exp.get("top_risk_drivers", []),
            "top_risk_mitigants": exp.get("top_risk_mitigants", []),
            "thin_file_explanation": exp.get("thin_file_explanation"),
        }

    # ------------------------------------------------------------------
    # Legacy helpers (still used by thin-file endpoint)
    # ------------------------------------------------------------------
    def get_thin_file(self) -> dict:
        thin_xai = self.xai_summary_report["thin_file_xai"]
        thin_seg = self.feature_report["thin_file_segment"]
        non_thin_seg = self.feature_report["non_thin_file_segment"]
        return {
            **thin_xai,
            "thin_file_count": thin_seg["count"],
            "mean_alt_credit_score": thin_seg["mean_alt_credit_score"],
            "mean_thin_file_confidence": thin_seg["mean_thin_file_confidence"],
            "mean_bureau_score": non_thin_seg["mean_bureau_score"],
        }

    def search_applicants(self, query: str) -> list[dict]:
        query = query.strip().lower()
        results = []
        for aid, exp in self.sample_explanations.items():
            if aid.lower().startswith(query):
                results.append({
                    "applicant_id": aid,
                    "risk_label": exp["predicted_risk_label"],
                    "default_probability": exp["predicted_default_probability"],
                    "is_thin_file": exp["is_thin_file"],
                })
        if results:
            return results

        closest = min(
            self.sample_explanations.values(),
            key=lambda exp: _prefix_distance(exp["applicant_id"].lower(), query),
        )
        return [{
            "applicant_id": closest["applicant_id"],
            "risk_label": closest["predicted_risk_label"],
            "default_probability": closest["predicted_default_probability"],
            "is_thin_file": closest["is_thin_file"],
        }]


def _prefix_distance(candidate: str, query: str) -> int:
    if not query:
        return 0
    common = 0
    for a, b in zip(candidate, query):
        if a == b:
            common += 1
        else:
            break
    return len(query) - common


state = AppState()
