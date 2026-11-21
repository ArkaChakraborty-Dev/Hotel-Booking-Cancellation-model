"""Unit tests for feature engineering module."""
import pandas as pd
import pytest

from src.features.build_features import engineer_features


@pytest.fixture
def sample_cleaned_data():
    return pd.DataFrame({
        "stays_in_weekend_nights": [1, 0, 2],
        "stays_in_week_nights": [2, 5, 0],
        "adults": [1, 2, 2],
        "children": [0.0, 1.0, 0.0],
        "babies": [0.0, 0.0, 1.0],
        "agent": ["9", "Unknown", "14"],
        "company": ["Unknown", "45", "Unknown"],
        "previous_cancellations": [0, 2, 0],
        "total_of_special_requests": [1, 0, 3],
        "required_car_parking_spaces": [0, 1, 0]
    })


def test_engineer_features_stay_and_guests(sample_cleaned_data):
    featured = engineer_features(sample_cleaned_data)

    # total_stay_nights: [3, 5, 2]
    assert list(featured["total_stay_nights"]) == [3, 5, 2]

    # total_guests: [1, 3, 3]
    assert list(featured["total_guests"]) == [1, 3, 3]

    # is_alone: [1, 0, 0]
    assert list(featured["is_alone"]) == [1, 0, 0]

    # has_children_or_babies: [0, 1, 1]
    assert list(featured["has_children_or_babies"]) == [0, 1, 1]


def test_engineer_features_channels_and_behavior(sample_cleaned_data):
    featured = engineer_features(sample_cleaned_data)

    # is_agent_booking: [1, 0, 1]
    assert list(featured["is_agent_booking"]) == [1, 0, 1]

    # is_company_booking: [0, 1, 0]
    assert list(featured["is_company_booking"]) == [0, 1, 0]

    # has_previous_cancellations: [0, 1, 0]
    assert list(featured["has_previous_cancellations"]) == [0, 1, 0]

    # has_special_requests: [1, 0, 1]
    assert list(featured["has_special_requests"]) == [1, 0, 1]

    # has_parking_space: [0, 1, 0]
    assert list(featured["has_parking_space"]) == [0, 1, 0]
