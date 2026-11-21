"""CLI script to trigger model training and optimization."""
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import load_config
from src.models.train import train_and_optimize
from src.utils.logger import setup_logger

logger = setup_logger("train_cli")


def main():
    parser = argparse.ArgumentParser(description="Train Hotel Booking Cancellation Model")
    parser.add_argument(
        "--config",
        type=str,
        default=str(project_root / "config" / "config.yaml"),
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Optional override path to dataset (.csv or .csv.zip)"
    )

    args = parser.parse_args()

    logger.info("Initializing training workflow...")
    config = load_config(args.config)
    bundle = train_and_optimize(config=config, data_path=args.data_path)

    logger.info("================ Training Completed Successfully ================")
    logger.info(f"Optimal Decision Threshold: {bundle['threshold']:.3f}")
    logger.info(f"Validation Metrics: {bundle['metrics_val']}")
    logger.info(f"Test Metrics: {bundle['metrics_test']}")
    logger.info("==================================================================")


if __name__ == "__main__":
    main()
