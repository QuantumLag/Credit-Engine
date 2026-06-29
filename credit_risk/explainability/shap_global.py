import sys
from pathlib import Path

import numpy as np
from loguru import logger

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import FEATURE_GROUPS  # noqa: E402


def _expected_value(value, class_index: int | None = None) -> float:
    arr = np.asarray(value)
    if arr.ndim == 0:
        return float(arr)
    if class_index is not None and len(arr) > class_index:
        return float(arr[class_index])
    return float(arr[0])


def _top_importance(feature_cols: list[str], importances: np.ndarray, limit: int = 20) -> list[dict]:
    rows = sorted(zip(feature_cols, importances), key=lambda item: item[1], reverse=True)
    return [
        {"feature": feature, "mean_abs_shap": round(float(value), 6), "rank": index + 1}
        for index, (feature, value) in enumerate(rows[:limit])
    ]


def build_global_shap_report(
    feature_cols: list[str],
    shap_values_reg: np.ndarray,
    shap_values_clf_high: np.ndarray,
    reg_explainer,
    clf_explainer,
) -> dict:
    reg_importance = np.abs(shap_values_reg).mean(axis=0)
    clf_importance = np.abs(shap_values_clf_high).mean(axis=0)
    feature_to_reg = dict(zip(feature_cols, reg_importance))
    feature_to_clf = dict(zip(feature_cols, clf_importance))

    group_rows = []
    for group, members in FEATURE_GROUPS.items():
        reg_total = sum(float(feature_to_reg.get(feature, 0.0)) for feature in members)
        clf_total = sum(float(feature_to_clf.get(feature, 0.0)) for feature in members)
        group_rows.append((group, reg_total, clf_total))

    ranked_groups = sorted(group_rows, key=lambda row: row[1] + row[2], reverse=True)
    feature_group_importance = {
        group: {
            "reg_importance": round(reg_total, 6),
            "clf_importance": round(clf_total, 6),
            "rank": rank,
        }
        for rank, (group, reg_total, clf_total) in enumerate(ranked_groups, start=1)
    }

    report = {
        "regression_global_importance": _top_importance(feature_cols, reg_importance),
        "classification_high_risk_importance": _top_importance(feature_cols, clf_importance),
        "feature_group_importance": feature_group_importance,
        "expected_value_reg": round(_expected_value(reg_explainer.expected_value), 6),
        "expected_value_clf_high": round(_expected_value(clf_explainer.expected_value, 2), 6),
    }
    logger.info("Built global SHAP report for {} features.", len(feature_cols))
    return report


def feature_group_for(feature: str) -> str:
    for group, members in FEATURE_GROUPS.items():
        if feature in members:
            return group
    return "Other"
