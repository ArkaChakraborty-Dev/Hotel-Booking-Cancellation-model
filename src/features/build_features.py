"""Feature engineering pipeline for hotel booking cancellation modeling."""
import pandas as pd
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates domain-specific features from raw booking attributes.
    
    Derived features include:
    - total_stay_nights: Sum of weekend and week nights.
    - total_guests: Sum of adults, children, and babies.
    - is_alone: Indicator for solo travelers.
    - has_children_or_babies: Indicator for family bookings with minors.
    - is_agent_booking: Indicator if booked via a third-party agency.
    - is_company_booking: Indicator if booked or paid via corporate entity.
    - has_previous_cancellations: Indicator for repeat cancelers.
    - has_special_requests: Indicator if guest placed special requests.
    - has_parking_space: Indicator if guest requested vehicle parking.
    
    Args:
        df: Input DataFrame (cleaned).
        
    Returns:
        pd.DataFrame: DataFrame enriched with engineered features.
    """
    data = df.copy()
    initial_cols = len(data.columns)

    # Total stay duration
    if "stays_in_weekend_nights" in data.columns and "stays_in_week_nights" in data.columns:
        data["total_stay_nights"] = (
            data["stays_in_weekend_nights"].fillna(0) +
            data["stays_in_week_nights"].fillna(0)
        )

    # Total guest counts & family indicators
    guest_cols = [c for c in ["adults", "children", "babies"] if c in data.columns]
    if guest_cols:
        data["total_guests"] = data[guest_cols].fillna(0).sum(axis=1)
        data["is_alone"] = (data["total_guests"] == 1).astype(int)

    if "children" in data.columns and "babies" in data.columns:
        data["has_children_or_babies"] = (
            (data["children"] > 0) | (data["babies"] > 0)
        ).astype(int)

    # Booking channel indicators
    if "agent" in data.columns:
        data["is_agent_booking"] = (data["agent"] != "Unknown").astype(int)

    if "company" in data.columns:
        data["is_company_booking"] = (data["company"] != "Unknown").astype(int)

    # Behavioral indicators
    if "previous_cancellations" in data.columns:
        data["has_previous_cancellations"] = (data["previous_cancellations"] > 0).astype(int)

    if "total_of_special_requests" in data.columns:
        data["has_special_requests"] = (data["total_of_special_requests"] > 0).astype(int)

    if "required_car_parking_spaces" in data.columns:
        data["has_parking_space"] = (data["required_car_parking_spaces"] > 0).astype(int)

    logger.info(f"Feature engineering completed: added {len(data.columns) - initial_cols} new features.")
    return data
