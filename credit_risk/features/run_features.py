import json
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import CLEAN_BORROWERS_PATH  # noqa: E402
from features.encode_scale import encode_categoricals, scale_numeric_features  # noqa: E402
from features.feature_store import (  # noqa: E402
    assemble_feature_matrix,
    build_feature_meta,
    build_feature_report,
    save_feature_outputs,
    validate_feature_matrix,
)
from features.ratios import add_financial_ratios  # noqa: E402
from features.thin_file import add_thin_file_features  # noqa: E402


def run_feature_pipeline() -> dict:
    logger.info("Running Module 2 feature engineering pipeline.")
    clean_df = pd.read_csv(CLEAN_BORROWERS_PATH)
    logger.info("Loaded cleaned borrower data from {} with shape {}.", CLEAN_BORROWERS_PATH, clean_df.shape)

    features = add_financial_ratios(clean_df)
    features = add_thin_file_features(features)
    features = encode_categoricals(features)
    features, scaler_params = scale_numeric_features(features)

    matrix = assemble_feature_matrix(features)
    validate_feature_matrix(matrix)

    report = build_feature_report(features, matrix, scaler_params)
    meta = build_feature_meta(matrix, scaler_params)
    save_feature_outputs(matrix, report, meta)
    logger.info("Module 2 feature engineering pipeline finished successfully.")
    return report


def main() -> None:
    report = run_feature_pipeline()
    logger.info(json.dumps(report, indent=2))
    print("Module 2 complete. Feature matrix ready for model training.")


if __name__ == "__main__":
    main()
