import json
import sys
from pathlib import Path

from loguru import logger

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline.clean_data import clean_borrower_data, save_processed_outputs  # noqa: E402
from pipeline.generate_data import generate_borrower_data, save_raw_data  # noqa: E402


def run_pipeline() -> dict:
    logger.info("Running Module 1 data pipeline.")
    raw_df = generate_borrower_data()
    save_raw_data(raw_df)

    clean_df = clean_borrower_data(raw_df)
    report = save_processed_outputs(raw_df, clean_df)
    logger.info("Module 1 data pipeline finished successfully.")
    return report


def main() -> None:
    report = run_pipeline()
    print(json.dumps(report, indent=2))
    print("Module 1 complete. Ready for feature engineering.")


if __name__ == "__main__":
    main()
