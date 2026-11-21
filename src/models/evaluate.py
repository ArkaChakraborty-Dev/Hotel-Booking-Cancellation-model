"""Evaluation module computing comprehensive classification metrics and threshold tuning."""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def evaluate_predictions(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    y_proba: Optional[Union[np.ndarray, pd.Series]] = None
) -> Dict[str, float]:
    """Computes comprehensive classification metrics.
    
    Args:
        y_true: Ground truth target labels.
        y_pred: Predicted binary labels.
        y_proba: Optional predicted probabilities for positive class (1).
        
    Returns:
        Dict containing accuracy, precision, recall, f1, and roc_auc / pr_auc if available.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0))
    }

    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
            metrics["pr_auc"] = float(average_precision_score(y_true, y_proba))
        except ValueError as e:
            logger.warning(f"Could not compute ROC-AUC or PR-AUC: {e}")

    return metrics


def find_optimal_threshold(
    y_true: Union[np.ndarray, pd.Series],
    val_probabilities: np.ndarray,
    thresholds: Optional[List[float]] = None,
    target_metric: str = "f1"
) -> Tuple[float, pd.DataFrame]:
    """Sweeps over probability thresholds on a validation set to find the optimal decision boundary.
    
    Args:
        y_true: Ground truth validation labels.
        val_probabilities: Validation probabilities for positive class (1).
        thresholds: List of candidate thresholds to test. Defaults to [0.20, 0.25, ..., 0.80].
        target_metric: Metric to maximize ('f1', 'recall', 'precision').
        
    Returns:
        Tuple of (best_threshold, threshold_dataframe).
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.20, 0.85, 0.05)]

    results = []
    for threshold in thresholds:
        preds = (val_probabilities >= threshold).astype(int)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        acc = accuracy_score(y_true, preds)

        results.append({
            "threshold": threshold,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "accuracy": round(acc, 4)
        })

    threshold_df = pd.DataFrame(results)
    
    if target_metric not in threshold_df.columns:
        logger.warning(f"Target metric '{target_metric}' not found. Defaulting to 'f1'.")
        target_metric = "f1"

    best_idx = threshold_df[target_metric].idxmax()
    best_threshold = float(threshold_df.loc[best_idx, "threshold"])
    logger.info(
        f"Optimal threshold found: {best_threshold} with {target_metric.upper()} = "
        f"{threshold_df.loc[best_idx, target_metric]}"
    )

    return best_threshold, threshold_df


def get_feature_importances(
    fitted_pipeline: Any,
    top_n: int = 20
) -> Optional[pd.DataFrame]:
    """Extracts feature importances from a fitted Random Forest pipeline.
    
    Args:
        fitted_pipeline: Scikit-learn Pipeline with 'preprocessor' and 'model' steps.
        top_n: Number of top features to return.
        
    Returns:
        Optional[pd.DataFrame]: Feature importance DataFrame, or None if model has no feature_importances_.
    """
    try:
        model = fitted_pipeline.named_steps["model"]
        preprocessor = fitted_pipeline.named_steps["preprocessor"]
        
        if not hasattr(model, "feature_importances_"):
            return None

        feature_names = preprocessor.get_feature_names_out()
        importances = model.feature_importances_

        df_importance = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)

        return df_importance.head(top_n)
    except Exception as e:
        logger.warning(f"Could not extract feature importances: {e}")
        return None
