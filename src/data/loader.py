"""Data loader module for hotel booking cancellation dataset."""
import os
from pathlib import Path
from typing import Union
import pandas as pd
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def load_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """Loads hotel booking data from a CSV or compressed CSV (.zip).
    
    Args:
        file_path: Path to the CSV or ZIP file.
        
    Returns:
        pd.DataFrame: Loaded dataset.
        
    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If file format is unsupported.
    """
    path = Path(file_path)
    if not path.is_file():
        # Check relative to project root
        project_root = Path(__file__).resolve().parent.parent.parent
        resolved_path = project_root / file_path
        if resolved_path.is_file():
            path = resolved_path
        else:
            raise FileNotFoundError(f"Dataset not found at '{file_path}' or '{resolved_path}'")

    logger.info(f"Loading dataset from: {path}")
    
    if path.suffix == ".zip" or str(path).endswith(".csv.zip"):
        df = pd.read_csv(path, compression="zip")
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Expected .csv or .csv.zip")

    logger.info(f"Dataset successfully loaded. Shape: {df.shape}")
    return df
