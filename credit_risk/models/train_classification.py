import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from loguru import logger
from xgboost import XGBClassifier

from models.evaluate import classification_metrics, thin_file_classification_metrics, top_features_by_gain


CLASSIFICATION_HYPERPARAMETERS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "use_label_encoder": False,
    "eval_metric": "mlogloss",
    "random_state": 42,
    "n_jobs": -1,
    "early_stopping_rounds": 30,
}


def apply_smote(X_train, y_train_encoded, label_encoder):
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train_encoded)
    labels, counts = np.unique(y_resampled, return_counts=True)
    distribution = {
        str(label_encoder.inverse_transform([int(label)])[0]): int(count)
        for label, count in zip(labels, counts)
    }
    logger.info("Applied SMOTE to training data only. Class distribution: {}.", distribution)
    return X_resampled, y_resampled, distribution


def train_classification_model(
    X_train_resampled,
    y_train_resampled,
    X_test,
    y_test_encoded,
    thin_mask_test,
    label_encoder,
    model_path,
):
    model = XGBClassifier(**CLASSIFICATION_HYPERPARAMETERS)
    model.fit(X_train_resampled, y_train_resampled, eval_set=[(X_test, y_test_encoded)], verbose=False)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    if np.isnan(predictions).any() or np.isnan(probabilities).any():
        raise ValueError("Classification predictions contain NaN values.")

    labels = list(label_encoder.classes_)
    metrics = classification_metrics(y_test_encoded, predictions, probabilities, labels)
    if metrics["accuracy"] <= 0.70:
        logger.warning("Classification accuracy is {}, below the requested 0.70 threshold.", metrics["accuracy"])

    joblib.dump(model, model_path)
    logger.info("Saved classification model to {}.", model_path)

    return model, {
        "algorithm": "XGBoostClassifier",
        "hyperparameters": CLASSIFICATION_HYPERPARAMETERS,
        "smote_applied": True,
        "test_metrics": metrics,
        "thin_file_metrics": thin_file_classification_metrics(y_test_encoded, predictions, thin_mask_test),
        "top_15_features_by_gain": top_features_by_gain(model, list(X_test.columns)),
    }
