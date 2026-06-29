"""Load models and data once at startup."""

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

    def get_overview(self) -> dict:
        df = self.feature_matrix
        risk_counts = df["risk_label"].value_counts().to_dict()
        risk_distribution = {label: int(risk_counts.get(label, 0)) for label in ["High", "Medium", "Low"]}
        thin_file_count = int(df["is_thin_file"].sum())
        total = len(df)

        reg_metrics = self.model_report["regression_model"]["test_metrics"]
        clf_metrics = self.model_report["classification_model"]["test_metrics"]

        return {
            "total_applicants": total,
            "risk_distribution": risk_distribution,
            "thin_file_count": thin_file_count,
            "thin_file_pct": round(thin_file_count / total * 100, 1),
            "mean_default_probability": round(float(df["default_probability"].mean()), 4),
            "model_performance": {
                "regression_r2": reg_metrics["r2"],
                "classification_accuracy": clf_metrics["accuracy"],
                "auc_roc": clf_metrics["auc_roc_macro"],
            },
            "fairness_flags": {
                "regression": self.model_report["regression_model"]["thin_file_metrics"]["fairness_flag"],
                "classification": self.model_report["classification_model"]["thin_file_metrics"]["fairness_flag"],
            },
        }

    def get_feature_importance(self) -> dict:
        report = self.global_shap_report
        reg_top = report["regression_global_importance"][:15]
        clf_top = report["classification_high_risk_importance"][:15]

        def enrich(features: list) -> list:
            return [
                {**item, "group": feature_to_group(item["feature"])}
                for item in features
            ]

        return {
            "regression": enrich(reg_top),
            "classification": enrich(clf_top),
            "feature_group_importance": report["feature_group_importance"],
        }

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

    def get_applicant_list(self) -> list[dict]:
        return [
            {
                "applicant_id": aid,
                "risk_label": exp["predicted_risk_label"],
                "is_thin_file": exp["is_thin_file"],
            }
            for aid, exp in sorted(self.sample_explanations.items())
        ]


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
