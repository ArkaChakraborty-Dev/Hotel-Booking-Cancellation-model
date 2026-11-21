"""Unit tests for Scikit-Learn pipeline creation and execution."""
import numpy as np
import pandas as pd
import pytest

from src.models.pipeline import build_pipeline


@pytest.fixture
def mock_dataset():
    np.random.seed(42)
    n = 60
    return pd.DataFrame({
        "lead_time": np.random.randint(1, 300, n),
        "adr": np.random.uniform(50.0, 250.0, n),
        "total_stay_nights": np.random.randint(1, 10, n),
        "total_guests": np.random.randint(1, 4, n),
        "hotel": np.random.choice(["City Hotel", "Resort Hotel"], n),
        "market_segment": np.random.choice(["Direct", "Online TA", "Corporate"], n),
        "deposit_type": np.random.choice(["No Deposit", "Non Refund"], n),
        "customer_type": np.random.choice(["Transient", "Contract"], n)
    }), pd.Series(np.random.choice([0, 1], n, p=[0.6, 0.4]))


def test_build_and_fit_pipeline(mock_dataset):
    X, y = mock_dataset
    pipeline, num_cols, cat_cols = build_pipeline(
        X_train=X,
        model_type="random_forest",
        model_params={"n_estimators": 10, "max_depth": 5, "random_state": 42}
    )

    assert len(num_cols) == 4
    assert len(cat_cols) == 4

    # Fit pipeline
    pipeline.fit(X, y)

    # Predict
    preds = pipeline.predict(X)
    assert len(preds) == len(y)
    assert set(np.unique(preds)).issubset({0, 1})

    # Predict proba
    probas = pipeline.predict_proba(X)
    assert probas.shape == (len(y), 2)
    assert np.all((probas >= 0.0) & (probas <= 1.0))
