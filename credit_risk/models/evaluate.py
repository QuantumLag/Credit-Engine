import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    return {
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "within_0.1_accuracy": round(float(np.mean(np.abs(y_pred - y_true.to_numpy()) < 0.1)), 4),
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, labels: list[str]) -> dict:
    per_class = f1_score(y_true, y_pred, average=None, labels=list(range(len(labels))), zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "per_class_f1": {
            label: round(float(per_class[index]), 4)
            for index, label in enumerate(labels)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(labels)))).tolist(),
        "auc_roc_macro": round(float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")), 4),
    }


def top_features_by_gain(model, feature_names: list[str], limit: int = 15) -> list[dict]:
    booster = model.get_booster()
    gain_scores = booster.get_score(importance_type="gain")
    rows = []
    for raw_name, importance in gain_scores.items():
        if raw_name.startswith("f") and raw_name[1:].isdigit():
            feature = feature_names[int(raw_name[1:])]
        else:
            feature = raw_name
        rows.append({"feature": feature, "importance": float(importance)})

    rows.sort(key=lambda row: row["importance"], reverse=True)
    return [
        {"feature": row["feature"], "importance": round(row["importance"], 4)}
        for row in rows[:limit]
    ]


def thin_file_regression_metrics(y_true: pd.Series, y_pred: np.ndarray, thin_mask: pd.Series) -> dict:
    thin_mask_array = thin_mask.to_numpy().astype(bool)
    rmse_thin = np.sqrt(mean_squared_error(y_true.to_numpy()[thin_mask_array], y_pred[thin_mask_array]))
    rmse_non_thin = np.sqrt(mean_squared_error(y_true.to_numpy()[~thin_mask_array], y_pred[~thin_mask_array]))
    return {
        "rmse_thin": round(float(rmse_thin), 4),
        "rmse_non_thin": round(float(rmse_non_thin), 4),
        "fairness_flag": bool(abs(rmse_thin - rmse_non_thin) > 0.05),
    }


def thin_file_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, thin_mask: pd.Series) -> dict:
    thin_mask_array = thin_mask.to_numpy().astype(bool)
    accuracy_thin = accuracy_score(y_true[thin_mask_array], y_pred[thin_mask_array])
    accuracy_non_thin = accuracy_score(y_true[~thin_mask_array], y_pred[~thin_mask_array])
    return {
        "accuracy_thin": round(float(accuracy_thin), 4),
        "accuracy_non_thin": round(float(accuracy_non_thin), 4),
        "fairness_flag": bool(abs(accuracy_thin - accuracy_non_thin) > 0.08),
    }
