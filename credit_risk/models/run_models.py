import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import (  # noqa: E402
    CLASSIFICATION_MODEL_PATH,
    FEATURE_MATRIX_PATH,
    LABEL_ENCODER_PATH,
    MODEL_EXCLUDE_COLUMNS,
    MODEL_RANDOM_STATE,
    MODEL_REPORT_PATH,
    REGRESSION_MODEL_PATH,
    SAVED_MODEL_DIR,
    TEST_INDICES_PATH,
    TEST_SIZE,
)
from models.train_classification import (  # noqa: E402
    apply_smote,
    train_classification_model,
)
from models.train_regression import train_regression_model  # noqa: E402


def load_and_split_data():
    matrix = pd.read_csv(FEATURE_MATRIX_PATH)
    feature_cols = [column for column in matrix.columns if column not in MODEL_EXCLUDE_COLUMNS]
    X = matrix[feature_cols]
    y_reg = matrix["default_probability"]
    y_clf = matrix["risk_label"]

    split = train_test_split(
        X,
        y_reg,
        y_clf,
        matrix["is_thin_file"],
        matrix.index.to_numpy(),
        test_size=TEST_SIZE,
        random_state=MODEL_RANDOM_STATE,
        stratify=y_clf,
    )
    (
        X_train,
        X_test,
        y_train_reg,
        y_test_reg,
        y_train_clf,
        y_test_clf,
        thin_train,
        thin_test,
        train_indices,
        test_indices,
    ) = split

    if len(X_test) != 1000:
        raise ValueError(f"Expected test size of 1000 rows; found {len(X_test)}.")

    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    np.save(TEST_INDICES_PATH, test_indices)
    logger.info("Saved test indices to {}.", TEST_INDICES_PATH)
    logger.info("Split feature matrix into {} train and {} test rows.", len(X_train), len(X_test))
    return X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf, thin_test


def run_model_pipeline() -> dict:
    logger.info("Running Module 3 model training pipeline.")
    X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf, thin_test = load_and_split_data()

    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(["Low", "Medium", "High"], dtype=object)
    y_train_clf_encoded = label_encoder.transform(y_train_clf)
    y_test_clf_encoded = label_encoder.transform(y_test_clf)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    logger.info("Saved label encoder to {}.", LABEL_ENCODER_PATH)

    X_train_clf_resampled, y_train_clf_resampled, smote_distribution = apply_smote(
        X_train,
        y_train_clf_encoded,
        label_encoder,
    )

    _, regression_report = train_regression_model(
        X_train,
        X_test,
        y_train_reg,
        y_test_reg,
        thin_test,
        REGRESSION_MODEL_PATH,
    )
    _, classification_report = train_classification_model(
        X_train_clf_resampled,
        y_train_clf_resampled,
        X_test,
        y_test_clf_encoded,
        thin_test,
        label_encoder,
        CLASSIFICATION_MODEL_PATH,
    )
    classification_report["train_class_distribution_after_smote"] = smote_distribution

    report = {
        "regression_model": regression_report,
        "classification_model": classification_report,
        "data_split": {
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
            "stratified": True,
            "smote_on_train_only": True,
            "feature_columns": list(X_train.columns),
        },
        "module_3_notes": (
            "alt_credit_score expected to dominate importance. "
            "Check thin_file fairness_flag before proceeding to Module 4."
        ),
    }
    MODEL_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Saved model report to {}.", MODEL_REPORT_PATH)

    for artifact in [REGRESSION_MODEL_PATH, CLASSIFICATION_MODEL_PATH, LABEL_ENCODER_PATH, TEST_INDICES_PATH]:
        if not artifact.exists():
            raise FileNotFoundError(f"Expected artifact was not created: {artifact}")

    logger.info("Module 3 model training pipeline finished successfully.")
    return report


def main() -> None:
    report = run_model_pipeline()
    logger.info(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("Module 3 complete. Models saved. Ready for SHAP explainability.")


if __name__ == "__main__":
    main()
