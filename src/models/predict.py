"""Production inference module for predicting hotel booking cancellations."""
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
import pandas as pd

from src.data.cleaner import clean_raw_data
from src.features.build_features import engineer_features
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class BookingCancellationPredictor:
    """Production predictor wrapper loading serialized model bundle and executing predictions."""

    def __init__(self, model_bundle_path: Optional[Union[str, Path]] = None):
        """Initializes the predictor by loading the serialized model bundle.
        
        Args:
            model_bundle_path: Path to .joblib file containing model bundle.
        """
        if model_bundle_path is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            model_bundle_path = project_root / "models" / "hotel_cancellation_model.joblib"

        path = Path(model_bundle_path)
        if not path.is_file():
            raise FileNotFoundError(f"Trained model bundle not found at: {path}")

        logger.info(f"Loading model bundle from: {path}")
        bundle = joblib.load(path)

        self.model = bundle["model"]
        self.default_threshold = float(bundle.get("threshold", 0.5))
        self.expected_features = bundle.get("feature_columns", [])
        self.metrics_test = bundle.get("metrics_test", {})
        logger.info(
            f"Model loaded successfully. Optimal threshold: {self.default_threshold:.3f}"
        )

    def _prepare_input(self, raw_input: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]) -> pd.DataFrame:
        """Sanitizes and transforms input raw booking dictionaries or DataFrame.
        
        Args:
            raw_input: Raw booking records.
            
        Returns:
            pd.DataFrame: Transformed features aligned with training schema.
        """
        if isinstance(raw_input, dict):
            df = pd.DataFrame([raw_input])
        elif isinstance(raw_input, list):
            df = pd.DataFrame(raw_input)
        elif isinstance(raw_input, pd.DataFrame):
            df = raw_input.copy()
        else:
            raise TypeError(f"Unsupported input type: {type(raw_input)}. Expected DataFrame or dict/list.")

        # Clean and engineer features
        cleaned = clean_raw_data(df, drop_duplicates=False)
        featured = engineer_features(cleaned)

        # Drop target column if present
        if "is_canceled" in featured.columns:
            featured = featured.drop(columns=["is_canceled"])

        # Align with expected training features
        if self.expected_features:
            for col in self.expected_features:
                if col not in featured.columns:
                    featured[col] = np.nan
            featured = featured[self.expected_features]

        return featured

    def predict_proba(
        self,
        data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]
    ) -> np.ndarray:
        """Returns cancellation probabilities (probability of class 1).
        
        Args:
            data: Booking records.
            
        Returns:
            np.ndarray: Array of cancellation probabilities between 0.0 and 1.0.
        """
        X = self._prepare_input(data)
        probabilities = self.model.predict_proba(X)[:, 1]
        return probabilities

    def predict(
        self,
        data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """Predicts binary cancellation labels using the calibrated decision threshold.
        
        Args:
            data: Booking records.
            threshold: Optional custom decision threshold. If None, uses calibrated threshold.
            
        Returns:
            np.ndarray: 0 (Not Cancelled) or 1 (Cancelled).
        """
        t = threshold if threshold is not None else self.default_threshold
        proba = self.predict_proba(data)
        return (proba >= t).astype(int)

    def predict_detailed(
        self,
        data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
        threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """Returns a formatted DataFrame with predictions, probabilities, and risk levels.
        
        Args:
            data: Booking records.
            threshold: Optional custom decision threshold.
            
        Returns:
            pd.DataFrame: Contains columns ['is_canceled_pred', 'cancellation_probability',
                          'risk_level', 'decision_threshold'].
        """
        t = threshold if threshold is not None else self.default_threshold
        proba = self.predict_proba(data)
        preds = (proba >= t).astype(int)

        # Map risk levels
        risk_levels = []
        for p in proba:
            if p >= 0.7:
                risk_levels.append("High Risk")
            elif p >= 0.4:
                risk_levels.append("Medium Risk")
            else:
                risk_levels.append("Low Risk")

        return pd.DataFrame({
            "is_canceled_pred": preds,
            "cancellation_probability": np.round(proba, 4),
            "risk_level": risk_levels,
            "decision_threshold": round(t, 4)
        })
