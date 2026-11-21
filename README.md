# Hotel Booking Cancellation Prediction 🏨

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-grade Machine Learning pipeline that predicts hotel booking cancellations prior to guest check-in. Designed with strict anti-leakage guards, domain feature engineering, automated data sanitization, and calibrated decision threshold optimization.

---

## 📌 Business Value

In the hospitality industry, cancellations create severe revenue volatility, unexpected room vacancies, and operational inefficiency. By predicting cancellations accurately at booking creation:
1. **Revenue Management**: Hotels can dynamically calibrate overbooking strategies without triggering walk penalties.
2. **Targeted Retention**: High-risk bookings can trigger targeted re-engagement campaigns or non-refundable discount incentives.
3. **Staffing & Resource Planning**: Food & beverage and housekeeping allocations can be accurately forecast.

---

## 🔍 Data Discoveries & Anti-Leakage Safeguards

During exploratory data diagnostics, several critical vulnerabilities were addressed:

1. **Exact Duplicate Rows (~31,994 rows)**:
   - A large proportion of the dataset consisted of identical duplicate bookings. Retaining duplicates causes cross-contamination between train and test splits, inflating accuracy artificially. Exact duplicates are systematically removed prior to partitioning.
2. **Target Leakage Prevention**:
   - `reservation_status` (Check-Out, Canceled, No-Show) and `reservation_status_date` are 100% target leakage and strictly excluded.
   - `assigned_room_type` is determined at or near check-in (not booking time) and is excluded.
   - `booking_changes` occur between booking creation and check-in and are excluded for booking-time inference.
3. **Identifier Normalization (`agent` & `company`)**:
   - Travel agent and corporate company IDs were originally stored as floats due to null values. They have been converted to high-cardinality categorical identifiers with `'Unknown'` imputation and frequency thresholding (`min_frequency=0.01`) in the encoder to prevent overfitting.
4. **ADR Sanitization**:
   - Negative Average Daily Rates (`adr < 0`) were sanitized, and extreme outliers (> $5,000) capped to protect numerical scalers.
5. **Zero-Leakage Threshold Optimization**:
   - Fixed validation leakage present in exploratory workflows by strictly isolating the validation partition (`X_val`) from training (`X_train_fit`) before tuning decision thresholds.

---

## 🏗️ Repository Architecture

```text
hotel-booking-cancelation-model/
├── .github/
│   └── workflows/
│       └── ci.yml                     # Automated CI workflow for linting and unit tests
├── .gitignore                         # Comprehensive Git exclusion rules
├── README.md                          # Project documentation
├── requirements.txt                   # Production and development dependencies
├── pyproject.toml                     # Modern build and tool configuration
├── config/
│   └── config.yaml                    # Centralized hyperparameter and pipeline config
├── data/
│   ├── hotel_bookings.csv.zip         # Raw compressed dataset
│   ├── README.md                      # Data schema and dictionary
│   ├── raw/                           # Raw data storage (.gitkeep)
│   └── processed/                     # Processed splits (.gitkeep)
├── notebooks/
│   ├── hotel_cancellation_eda_and_modeling.ipynb  # Cleaned, narrative-driven EDA notebook
│   └── hotel_cancellation_model.ipynb # Original legacy exploratory notebook
├── src/
│   ├── __init__.py
│   ├── config.py                      # YAML configuration loader
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                  # Resilient CSV/ZIP dataset reader
│   │   └── cleaner.py                 # Anti-leakage, deduplication, and anomaly cleaner
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py          # Domain feature engineering
│   ├── models/
│   │   ├── __init__.py
│   │   ├── pipeline.py                # Scikit-Learn ColumnTransformer and Pipeline
│   │   ├── train.py                   # Zero-leakage train & threshold tuning orchestrator
│   │   ├── evaluate.py                # Evaluation metrics, PR-AUC, and feature importances
│   │   └── predict.py                 # Production inference class
│   └── utils/
│       ├── __init__.py
│       └── logger.py                  # Standardized logger
├── tests/
│   ├── __init__.py
│   ├── test_cleaner.py                # Unit tests for data cleaner
│   ├── test_features.py               # Unit tests for feature pipeline
│   ├── test_pipeline.py               # Unit tests for ColumnTransformer
│   └── test_inference.py              # Unit tests for prediction engine
└── scripts/
    ├── train.py                       # CLI entrypoint to train and serialize model
    └── predict.py                     # CLI entrypoint for batch predictions
```

---

## 🚀 Quickstart Guide

### 1. Installation

Clone your repository and set up a Python virtual environment:

```bash
git clone <your-repo-url>
cd hotel-booking-cancelation-model

python -m venv .venv
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 2. Run Tests

Verify code integrity with pytest:

```bash
pytest tests/ -v
```

### 3. Train the Model

Train the complete pipeline, perform zero-leakage threshold calibration, and export the production `.joblib` bundle:

```bash
python scripts/train.py
```

*Custom configurations can be supplied via `--config config/config.yaml` or `--data-path path/to/dataset.csv.zip`.*

### 4. Run Batch Inference

Generate predictions and risk-level scores on new booking records:

```bash
python scripts/predict.py --input-file path/to/new_bookings.csv --output-file data/predictions.csv
```

---

## 💻 Python API Usage

Use the predictor directly in production applications or API endpoints:

```python
from src.models.predict import BookingCancellationPredictor

# Initialize predictor (loads serialized bundle & optimal threshold)
predictor = BookingCancellationPredictor()

# Sample raw booking record
new_booking = {
    "hotel": "City Hotel",
    "lead_time": 180,
    "stays_in_weekend_nights": 1,
    "stays_in_week_nights": 2,
    "adults": 2,
    "children": 0,
    "babies": 0,
    "meal": "BB",
    "country": "PRT",
    "market_segment": "Online TA",
    "distribution_channel": "TA/TO",
    "is_repeated_guest": 0,
    "previous_cancellations": 1,
    "previous_bookings_not_canceled": 0,
    "reserved_room_type": "A",
    "deposit_type": "No Deposit",
    "agent": 9.0,
    "company": None,
    "days_in_waiting_list": 0,
    "customer_type": "Transient",
    "adr": 115.0,
    "required_car_parking_spaces": 0,
    "total_of_special_requests": 0
}

# Detailed prediction with calibrated threshold and risk assessment
result = predictor.predict_detailed(new_booking)
print(result)
# Output:
#    is_canceled_pred  cancellation_probability  risk_level  decision_threshold
# 0                 1                    0.7420   High Risk               0.450
```

---

## 📊 Modeling & Evaluation Highlights

- **Baseline vs. Primary**: Random Forest significantly outperforms Logistic Regression by capturing complex non-linear interactions between `lead_time`, `country`, `deposit_type`, and `total_of_special_requests`.
- **Decision Threshold Tuning**: Swept on a strictly isolated validation split ($0.20 \le \tau \le 0.80$) to find the balance between precision and recall, optimizing the business F1 score.
- **Top Predictive Features**:
  1. `lead_time` (Days elapsed between booking and check-in)
  2. `deposit_type` (Non-refundable vs. No Deposit)
  3. `country` (Country of origin)
  4. `total_of_special_requests` (Guests with requests cancel significantly less often)
  5. `total_stay_nights` & `adr` (Rate and length of stay)

---

## 📤 Git Push Ready Commands

To initialize and push this clean codebase to your Git remote:

```bash
# Check status
git status

# Add all structured files (.gitignore ensures raw zip, caches, and venv are excluded)
git add .

# Create initial commit
git commit -m "feat: restructure ML pipeline, add anti-leakage guards, unit tests, and CI workflow"

# Connect to your GitHub repository
git remote add origin https://github.com/<your-username>/hotel-booking-cancellation-model.git
git branch -M main
git push -u origin main
```

---

## 📄 License
This project is open-source under the [MIT License](LICENSE).
