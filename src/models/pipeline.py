"""Pipeline construction module for feature transformations and model architectures."""
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def build_preprocessor(
    X: pd.DataFrame,
    scaler_type: str = "standard",
    numeric_imputer_strategy: str = "median",
    categorical_fill_value: str = "Unknown",
    min_frequency: float = 0.01
) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """Builds a robust Scikit-Learn ColumnTransformer for numeric and categorical attributes.
    
    Args:
        X: Training features dataframe for column type inference.
        scaler_type: 'standard' for StandardScaler or 'robust' for RobustScaler.
        numeric_imputer_strategy: Strategy for numerical imputation ('median', 'mean').
        categorical_fill_value: Value used to impute missing categorical features.
        min_frequency: Minimum category frequency for infrequent grouping in OneHotEncoder.
        
    Returns:
        Tuple containing (preprocessor ColumnTransformer, numeric_columns, categorical_columns).
    """
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()

    logger.info(
        f"Feature separation: {len(numeric_features)} numerical features, "
        f"{len(categorical_features)} categorical features."
    )

    scaler = RobustScaler() if scaler_type.lower() == "robust" else StandardScaler()

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy=numeric_imputer_strategy)),
        ("scaler", scaler)
    ])

    try:
        ohe = OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=min_frequency,
            sparse_output=False
        )
    except (TypeError, ValueError):
        # Fallback for earlier scikit-learn compatibility
        ohe = OneHotEncoder(handle_unknown="ignore")

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=categorical_fill_value)),
        ("encoder", ohe)
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )

    return preprocessor, numeric_features, categorical_features


def build_classifier(
    model_type: str = "random_forest",
    hyperparameters: Optional[Dict[str, Any]] = None
) -> Any:
    """Instantiates a classifier based on model_type and hyperparameter dictionary.
    
    Args:
        model_type: 'random_forest' or 'logistic_regression'.
        hyperparameters: Optional dictionary of keyword arguments passed to the estimator.
        
    Returns:
        Scikit-learn estimator instance.
    """
    params = hyperparameters.copy() if hyperparameters else {}

    if model_type.lower() == "random_forest":
        default_params = {
            "n_estimators": 200,
            "max_depth": 20,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1
        }
        default_params.update(params)
        return RandomForestClassifier(**default_params)

    elif model_type.lower() == "logistic_regression":
        default_params = {
            "max_iter": 1000,
            "class_weight": "balanced",
            "random_state": 42,
            "C": 1.0
        }
        default_params.update(params)
        return LogisticRegression(**default_params)

    else:
        raise ValueError(
            f"Unsupported model_type: '{model_type}'. Choose 'random_forest' or 'logistic_regression'."
        )


def build_pipeline(
    X_train: pd.DataFrame,
    model_type: str = "random_forest",
    model_params: Optional[Dict[str, Any]] = None,
    scaler_type: str = "standard",
    min_frequency: float = 0.01
) -> Tuple[Pipeline, List[str], List[str]]:
    """Builds a complete end-to-end Pipeline with preprocessor and classifier.
    
    Args:
        X_train: Training features dataframe.
        model_type: Type of classifier ('random_forest', 'logistic_regression').
        model_params: Optional estimator hyperparameters.
        scaler_type: Scaler type ('standard', 'robust').
        min_frequency: Threshold for infrequent category grouping.
        
    Returns:
        Tuple of (Pipeline, numeric_features, categorical_features).
    """
    preprocessor, num_cols, cat_cols = build_preprocessor(
        X_train,
        scaler_type=scaler_type,
        min_frequency=min_frequency
    )
    model = build_classifier(model_type=model_type, hyperparameters=model_params)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return pipeline, num_cols, cat_cols
