"""Data cleaning and quality sanitization module."""
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

DEFAULT_LEAKAGE_COLUMNS = [
    "reservation_status",
    "reservation_status_date",
    "assigned_room_type",
    "booking_changes"
]


def clean_raw_data(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    leakage_columns: Optional[List[str]] = None,
    cleaning_config: Optional[Dict] = None
) -> pd.DataFrame:
    """Applies comprehensive data cleaning and anti-leakage guards to the raw dataset.
    
    Cleaning steps include:
    1. Exact duplicate row removal.
    2. Outlier and anomaly sanitization for Average Daily Rate (ADR < 0 and extreme outliers).
    3. Proper categorical conversion for agent and company identifiers.
    4. Safe handling of missing values (country, children, agent, company).
    5. Removal of post-booking data leakage columns.
    
    Args:
        df: Input raw DataFrame.
        drop_duplicates: Whether to drop exact duplicate rows.
        leakage_columns: List of columns to drop to prevent target leakage.
        cleaning_config: Optional dictionary with custom cleaning thresholds.
        
    Returns:
        pd.DataFrame: Cleaned and leak-free dataset.
    """
    data = df.copy()
    initial_shape = data.shape
    logger.info(f"Starting data cleaning. Initial shape: {initial_shape}")

    # 1. Exact duplicate removal
    if drop_duplicates:
        dups_count = data.duplicated().sum()
        if dups_count > 0:
            data = data.drop_duplicates().reset_index(drop=True)
            logger.info(f"Removed {dups_count} duplicate rows. New shape: {data.shape}")

    # 2. Outlier / Invalid value sanitization for ADR
    min_adr = cleaning_config.get("min_adr", 0.0) if cleaning_config else 0.0
    max_adr = cleaning_config.get("max_adr", 5000.0) if cleaning_config else 5000.0
    
    if "adr" in data.columns:
        negative_adr_mask = data["adr"] < min_adr
        if negative_adr_mask.sum() > 0:
            logger.info(f"Replacing {negative_adr_mask.sum()} negative ADR values with NaN for imputation.")
            data.loc[negative_adr_mask, "adr"] = np.nan

        extreme_adr_mask = data["adr"] > max_adr
        if extreme_adr_mask.sum() > 0:
            logger.info(f"Capping {extreme_adr_mask.sum()} extreme ADR values at {max_adr}.")
            data.loc[extreme_adr_mask, "adr"] = max_adr

    # 3. Categorical Identifier formatting (agent and company)
    for col in ["agent", "company"]:
        if col in data.columns:
            # Pandas casts numeric columns with nulls to float. Convert to string IDs.
            data[col] = (
                data[col]
                .astype("Int64")
                .astype("string")
                .fillna("Unknown")
                .astype(object)
            )

    # 4. Handle other known missing values
    if "children" in data.columns:
        fill_children = cleaning_config.get("fill_missing_children", 0.0) if cleaning_config else 0.0
        data["children"] = data["children"].fillna(fill_children)

    if "country" in data.columns:
        fill_country = cleaning_config.get("fill_missing_country", "Unknown") if cleaning_config else "Unknown"
        data["country"] = data["country"].fillna(fill_country)

    # 5. Flag zero-guest bookings
    if all(col in data.columns for col in ["adults", "children", "babies"]):
        data["zero_guest_booking"] = (
            (data["adults"] == 0) &
            (data["children"] == 0) &
            (data["babies"] == 0)
        ).astype(int)

    # 6. Target leakage prevention
    cols_to_drop = leakage_columns if leakage_columns is not None else DEFAULT_LEAKAGE_COLUMNS
    existing_leakage = [col for col in cols_to_drop if col in data.columns]
    if existing_leakage:
        logger.info(f"Dropping target leakage / post-booking columns: {existing_leakage}")
        data = data.drop(columns=existing_leakage)

    logger.info(f"Data cleaning complete. Final shape: {data.shape}")
    return data
