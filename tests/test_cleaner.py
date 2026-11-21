"""Unit tests for data cleaning and anti-leakage module."""
import numpy as np
import pandas as pd
import pytest

from src.data.cleaner import clean_raw_data


@pytest.fixture
def sample_raw_data():
    return pd.DataFrame({
        "hotel": ["City Hotel", "City Hotel", "Resort Hotel", "City Hotel"],
        "is_canceled": [0, 0, 1, 0],
        "lead_time": [10, 10, 50, 5],
        "adults": [2, 2, 0, 1],
        "children": [0.0, 0.0, 0.0, np.nan],
        "babies": [0, 0, 0, 0],
        "adr": [100.0, 100.0, -5.0, 6000.0],
        "agent": [9.0, 9.0, np.nan, 14.0],
        "company": [np.nan, np.nan, 45.0, np.nan],
        "country": ["PRT", "PRT", np.nan, "ESP"],
        "reservation_status": ["Check-Out", "Check-Out", "Canceled", "Check-Out"],
        "reservation_status_date": ["2015-07-01", "2015-07-01", "2015-07-02", "2015-07-03"],
        "assigned_room_type": ["A", "A", "C", "D"],
        "booking_changes": [0, 0, 1, 2]
    })


def test_clean_raw_data_deduplication(sample_raw_data):
    # Rows 0 and 1 are exact duplicates
    cleaned = clean_raw_data(sample_raw_data, drop_duplicates=True)
    assert len(cleaned) == 3


def test_clean_raw_data_leakage_removal(sample_raw_data):
    cleaned = clean_raw_data(sample_raw_data)
    for leakage_col in ["reservation_status", "reservation_status_date", "assigned_room_type", "booking_changes"]:
        assert leakage_col not in cleaned.columns


def test_clean_raw_data_adr_sanitization(sample_raw_data):
    cleaned = clean_raw_data(sample_raw_data, drop_duplicates=False)
    # Row 2 had ADR = -5.0, should now be NaN for imputation
    assert np.isnan(cleaned.loc[2, "adr"])
    # Row 3 had ADR = 6000.0, should be capped at 5000.0
    assert cleaned.loc[3, "adr"] <= 5000.0


def test_clean_raw_data_agent_company_types(sample_raw_data):
    cleaned = clean_raw_data(sample_raw_data, drop_duplicates=False)
    # Missing agent in row 2 should be 'Unknown'
    assert cleaned.loc[2, "agent"] == "Unknown"
    # Agent 9.0 should be string '9'
    assert str(cleaned.loc[0, "agent"]) == "9"
    # Country NaN should be 'Unknown'
    assert cleaned.loc[2, "country"] == "Unknown"
    # Children NaN should be 0.0
    assert cleaned.loc[3, "children"] == 0.0


def test_clean_raw_data_zero_guests_flag(sample_raw_data):
    cleaned = clean_raw_data(sample_raw_data, drop_duplicates=False)
    assert "zero_guest_booking" in cleaned.columns
    # Row 2 had adults=0, children=0, babies=0 -> flag = 1
    assert cleaned.loc[2, "zero_guest_booking"] == 1
    assert cleaned.loc[0, "zero_guest_booking"] == 0
