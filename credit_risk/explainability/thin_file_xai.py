import numpy as np
import pandas as pd
from loguru import logger


def build_thin_file_xai(
    X_test: pd.DataFrame,
    shap_values_reg: np.ndarray,
    regression_model,
    feature_cols: list[str],
) -> dict:
    thin_mask = X_test["is_thin_file"] == 1
    non_thin_mask = ~thin_mask
    predictions = regression_model.predict(X_test)

    alt_idx = feature_cols.index("alt_credit_score")
    upi_idx = feature_cols.index("upi_engagement_score")
    confidence_idx = feature_cols.index("thin_file_confidence")
    gig_idx = feature_cols.index("emp_Gig")
    msme_idx = feature_cols.index("emp_MSME")

    thin_alt = float(np.mean(shap_values_reg[thin_mask.to_numpy(), alt_idx]))
    thin_upi = float(np.mean(shap_values_reg[thin_mask.to_numpy(), upi_idx]))
    thin_confidence = float(np.mean(shap_values_reg[thin_mask.to_numpy(), confidence_idx]))
    thin_gig = float(np.mean(shap_values_reg[thin_mask.to_numpy(), gig_idx]))
    thin_msme = float(np.mean(shap_values_reg[thin_mask.to_numpy(), msme_idx]))
    non_thin_alt_proxy = float(np.mean(shap_values_reg[non_thin_mask.to_numpy(), alt_idx]))
    total_signal = np.abs(shap_values_reg[thin_mask.to_numpy()]).sum()
    alt_upi_signal = (
        np.abs(shap_values_reg[thin_mask.to_numpy(), alt_idx]).sum()
        + np.abs(shap_values_reg[thin_mask.to_numpy(), upi_idx]).sum()
    )
    signal_share = 0.0 if total_signal == 0 else alt_upi_signal / total_signal * 100

    report = {
        "thin_file_test_count": int(thin_mask.sum()),
        "mean_predicted_prob_thin": round(float(np.mean(predictions[thin_mask.to_numpy()])), 6),
        "mean_predicted_prob_non_thin": round(float(np.mean(predictions[non_thin_mask.to_numpy()])), 6),
        "mean_shap_alt_credit_score_thin": round(thin_alt, 6),
        "mean_shap_upi_engagement_thin": round(thin_upi, 6),
        "mean_shap_thin_file_confidence": round(thin_confidence, 6),
        "mean_shap_emp_gig_thin": round(thin_gig, 6),
        "mean_shap_emp_msme_thin": round(thin_msme, 6),
        "mean_alt_credit_score_shap_non_thin_proxy": round(non_thin_alt_proxy, 6),
        "interpretation": (
            "For thin-file borrowers, UPI behavior and alternate credit score contribute "
            f"{signal_share:.1f}% of total explainable risk signal, replacing the bureau "
            "score used for standard applicants."
        ),
    }
    logger.info("Built thin-file XAI report for {} test applicants.", int(thin_mask.sum()))
    return report
