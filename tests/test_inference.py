"""Unit tests for inference and prediction engine."""
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from src.models.pipeline import build_pipeline
from src.models.predict import BookingCancellationPredictor


@pytest.fixture
def mock_trained_bundle(tmp_path):
    import joblib

    # Create small mock dataset
    X = pd.DataFrame({
        "lead_time": [10, 100],
        "adr": [80.0, 150.0],
        "total_stay_nights": [2, 4],
        "total_guests": [2, 1],
        "hotel": ["City Hotel", "Resort Hotel"],
        "market_segment": ["Direct", "Online TA"]
    })
    y = np.array([0, 1])

    pipeline, _, _ = build_pipeline(X, model_type="logistic_regression")
    pipeline.fit(X, y)

    bundle = {
        "model": pipeline,
        "threshold": 0.45,
        "feature_columns": X.columns.tolist(),
        "target_column": "is_canceled",
        "metrics_test": {"f1": 0.85}
    }

    bundle_path = tmp_path / "mock_model.joblib"
    joblib.dump(bundle, bundle_path)
    return str(bundle_path)


def test_predictor_dict_and_detailed_output(mock_trained_bundle):
    predictor = BookingCancellationPredictor(model_bundle_path=mock_trained_bundle)

    raw_booking = {
        "hotel": "City Hotel",
        "lead_time": 25,
        "stays_in_weekend_nights": 1,
        "stays_in_week_nights": 2,
        "adults": 2,
        "children": 0,
        "babies": 0,
        "adr": 110.0,
        "market_segment": "Direct",
        "agent": 9.0,
        "company": None
    }

    detailed_df = predictor.predict_detailed(raw_booking)

    assert "is_canceled_pred" in detailed_df.columns
    assert "cancellation_probability" in detailed_df.columns
    assert "risk_level" in detailed_df.columns
    assert "decision_threshold" in detailed_df.columns
    assert detailed_df["decision_threshold"].iloc[0] == 0.45
    assert detailed_df["is_canceled_pred"].iloc[0] in [0, 1]
