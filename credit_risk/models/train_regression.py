import joblib
import numpy as np
from loguru import logger
from xgboost import XGBRegressor

from models.evaluate import regression_metrics, thin_file_regression_metrics, top_features_by_gain


REGRESSION_HYPERPARAMETERS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "eval_metric": "rmse",
    "early_stopping_rounds": 30,
}


def train_regression_model(X_train, X_test, y_train, y_test, thin_mask_test, model_path):
    model = XGBRegressor(**REGRESSION_HYPERPARAMETERS)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    predictions = np.clip(model.predict(X_test), 0.0, 1.0)

    if np.isnan(predictions).any():
        raise ValueError("Regression predictions contain NaN values.")

    metrics = regression_metrics(y_test, predictions)
    if metrics["r2"] <= 0.60:
        logger.warning("Regression R2 is {}, below the requested 0.60 threshold.", metrics["r2"])

    joblib.dump(model, model_path)
    logger.info("Saved regression model to {}.", model_path)

    return model, {
        "algorithm": "XGBoostRegressor",
        "hyperparameters": REGRESSION_HYPERPARAMETERS,
        "test_metrics": metrics,
        "thin_file_metrics": thin_file_regression_metrics(y_test, predictions, thin_mask_test),
        "top_15_features_by_gain": top_features_by_gain(model, list(X_train.columns)),
    }
