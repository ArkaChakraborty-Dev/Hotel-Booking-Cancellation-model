"""Training orchestrator for hotel booking cancellation prediction."""
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.cleaner import clean_raw_data
from src.data.loader import load_data
from src.features.build_features import engineer_features
from src.models.evaluate import (
    evaluate_predictions,
    find_optimal_threshold,
    get_feature_importances
)
from src.models.pipeline import build_pipeline
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def prepare_datasets(
    raw_df: pd.DataFrame,
    target_column: str = "is_canceled",
    drop_duplicates: bool = True,
    leakage_columns: Optional[list] = None,
    cleaning_config: Optional[dict] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """Cleans raw data, applies anti-leakage guards, engineers features, and separates X and y.
    
    Args:
        raw_df: Raw input DataFrame.
        target_column: Name of target column ('is_canceled').
        drop_duplicates: Whether to drop duplicate records.
        leakage_columns: List of columns to drop to prevent data leakage.
        cleaning_config: Optional cleaning parameters.
        
    Returns:
        Tuple of (Features DataFrame X, Target Series y).
    """
    logger.info("Step 1: Cleaning raw data...")
    cleaned_df = clean_raw_data(
        raw_df,
        drop_duplicates=drop_duplicates,
        leakage_columns=leakage_columns,
        cleaning_config=cleaning_config
    )

    logger.info("Step 2: Engineering features...")
    featured_df = engineer_features(cleaned_df)

    if target_column not in featured_df.columns:
        raise KeyError(f"Target column '{target_column}' not found in dataset!")

    X = featured_df.drop(columns=[target_column])
    y = featured_df[target_column].astype(int)

    logger.info(f"Prepared features shape: {X.shape}, target distribution:\n{y.value_counts(normalize=True)}")
    return X, y


def train_and_optimize(
    config: Dict[str, Any],
    data_path: Optional[str] = None
) -> Dict[str, Any]:
    """Executes full ML training workflow with zero-leakage threshold tuning.
    
    Workflow:
    1. Load raw data.
    2. Clean & guard against leakage.
    3. Engineer domain features.
    4. Stratified 3-way split: Dev (Train + Validation) and Test.
    5. Fit pipeline on Train set only.
    6. Tune decision threshold on strictly held-out Validation set.
    7. Evaluate final metrics on held-out Test set.
    8. Refit pipeline on full Dev set with optimal threshold and save bundle.
    
    Args:
        config: Configuration dictionary.
        data_path: Optional override for dataset path.
        
    Returns:
        Dict containing trained pipeline, threshold, evaluation metrics, and test results.
    """
    data_file = data_path or config.get("data", {}).get("raw_path", "data/hotel_bookings.csv.zip")
    target_col = config.get("data", {}).get("target_column", "is_canceled")
    drop_dups = config.get("data", {}).get("drop_duplicates", True)
    leakage_cols = config.get("data", {}).get("leakage_columns", None)
    cleaning_conf = config.get("data", {}).get("cleaning", {})

    # Load & Prepare
    raw_df = load_data(data_file)
    X, y = prepare_datasets(
        raw_df=raw_df,
        target_column=target_col,
        drop_duplicates=drop_dups,
        leakage_columns=leakage_cols,
        cleaning_config=cleaning_conf
    )

    # Splitting config
    split_cfg = config.get("splitting", {})
    test_size = split_cfg.get("test_size", 0.2)
    val_size = split_cfg.get("val_size", 0.2)
    random_state = split_cfg.get("random_state", 42)

    logger.info("Step 3: Creating stratified Train / Validation / Test partitions...")
    # Split into Dev (train + val) and final Test
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Split Dev into train_fit and validation
    X_train_fit, X_val, y_train_fit, y_val = train_test_split(
        X_dev, y_dev,
        test_size=val_size,
        random_state=random_state,
        stratify=y_dev
    )

    logger.info(
        f"Partition sizes: Train={X_train_fit.shape[0]}, "
        f"Validation={X_val.shape[0]}, Test={X_test.shape[0]}"
    )

    # Pipeline construction
    model_type = "random_forest"
    model_params = config.get("models", {}).get("random_forest", {})
    prep_cfg = config.get("preprocessing", {})
    scaler_type = prep_cfg.get("scaler_type", "standard")
    min_freq = prep_cfg.get("ohe_min_frequency", 0.01)

    logger.info("Step 4: Building model pipeline...")
    pipeline, num_cols, cat_cols = build_pipeline(
        X_train=X_train_fit,
        model_type=model_type,
        model_params=model_params,
        scaler_type=scaler_type,
        min_frequency=min_freq
    )

    logger.info("Step 5: Fitting pipeline on training partition...")
    pipeline.fit(X_train_fit, y_train_fit)

    # Threshold optimization on Validation partition
    logger.info("Step 6: Threshold optimization on validation set...")
    val_probabilities = pipeline.predict_proba(X_val)[:, 1]
    
    thresh_cfg = config.get("threshold_tuning", {})
    target_metric = thresh_cfg.get("metric", "f1")

    best_threshold, threshold_df = find_optimal_threshold(
        y_true=y_val,
        val_probabilities=val_probabilities,
        target_metric=target_metric
    )

    val_preds = (val_probabilities >= best_threshold).astype(int)
    val_metrics = evaluate_predictions(y_val, val_preds, val_probabilities)
    logger.info(f"Validation metrics at threshold {best_threshold}: {val_metrics}")

    # Evaluate on held-out Test set
    logger.info("Step 7: Evaluating on held-out Test set...")
    test_probabilities = pipeline.predict_proba(X_test)[:, 1]
    test_preds = (test_probabilities >= best_threshold).astype(int)
    test_metrics = evaluate_predictions(y_test, test_preds, test_probabilities)
    logger.info(f"Final Test metrics at threshold {best_threshold}: {test_metrics}")

    # Step 8: Refit on full development set (X_dev) for production deployment
    logger.info("Step 8: Refitting pipeline on full development set (train + val)...")
    final_pipeline, _, _ = build_pipeline(
        X_train=X_dev,
        model_type=model_type,
        model_params=model_params,
        scaler_type=scaler_type,
        min_frequency=min_freq
    )
    final_pipeline.fit(X_dev, y_dev)

    # Feature importances
    top_features = get_feature_importances(final_pipeline, top_n=15)
    if top_features is not None:
        logger.info(f"Top 10 Feature Importances:\n{top_features.head(10).to_string()}")

    # Save artifact
    art_cfg = config.get("artifacts", {})
    model_dir = Path(art_cfg.get("model_dir", "models"))
    model_file = art_cfg.get("model_filename", "hotel_cancellation_model.joblib")
    
    # Ensure directory exists relative to project root
    project_root = Path(__file__).resolve().parent.parent.parent
    save_path = project_root / model_dir / model_file
    save_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model": final_pipeline,
        "threshold": float(best_threshold),
        "feature_columns": X.columns.tolist(),
        "target_column": target_col,
        "metrics_val": val_metrics,
        "metrics_test": test_metrics,
        "threshold_tuning_table": threshold_df,
        "top_features": top_features
    }

    joblib.dump(bundle, save_path)
    logger.info(f"Model bundle successfully saved to: {save_path}")

    return bundle
