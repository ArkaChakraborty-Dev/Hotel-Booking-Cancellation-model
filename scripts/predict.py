"""CLI script to run batch or single inference using a trained model bundle."""
import argparse
import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.models.predict import BookingCancellationPredictor
from src.utils.logger import setup_logger

logger = setup_logger("predict_cli")


def main():
    parser = argparse.ArgumentParser(description="Run Cancellation Prediction on New Booking Data")
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to input CSV containing booking records"
    )
    parser.add_argument(
        "--model-bundle",
        type=str,
        default=str(project_root / "models" / "hotel_cancellation_model.joblib"),
        help="Path to trained model bundle .joblib"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=str(project_root / "data" / "predictions.csv"),
        help="Path to save predictions CSV"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional custom decision threshold override"
    )

    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.is_file():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    logger.info(f"Reading input data from: {input_path}")
    df_input = pd.read_csv(input_path)

    predictor = BookingCancellationPredictor(model_bundle_path=args.model_bundle)
    results_df = predictor.predict_detailed(df_input, threshold=args.threshold)

    # Combine input with predictions
    output_df = pd.concat([df_input.reset_index(drop=True), results_df], axis=1)

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(out_path, index=False)
    logger.info(f"Predictions saved to: {out_path}")
    logger.info(f"Cancellation summary:\n{output_df['risk_level'].value_counts()}")


if __name__ == "__main__":
    main()
