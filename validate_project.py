
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Locate the project folder.
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"

# Check that the required project files exist.
required_files = [
    "app.py",
    "train_models.py",
    "requirements.txt",
    "README.md",
    "Dry_Bean_Dataset.xlsx",
    "test_data.csv",
    "model_metrics.csv",
]

print("\n=== 1. Project Files ===")

for filename in required_files:
    path = ROOT / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {filename}")
    print(f"PASS: {filename}")

# Check that all five trained models exist and can be loaded.
model_files = [
    "logistic_regression.joblib",
    "decision_tree.joblib",
    "k-nearest_neighbors.joblib",
    "gaussian_naive_bayes.joblib",
    "support_vector_machine.joblib",
]

print("\n=== 2. Saved Models ===")

for filename in model_files:
    path = MODEL_DIR / filename

    if not path.is_file():
        raise FileNotFoundError(f"Missing model: model/{filename}")

    joblib.load(path)
    print(f"PASS: model/{filename}")

# Load and validate the feature list.
feature_path = MODEL_DIR / "feature_names.joblib"

if not feature_path.is_file():
    raise FileNotFoundError("Missing model/feature_names.joblib")

features = joblib.load(feature_path)

if len(features) < 12:
    raise ValueError("The dataset must have at least 12 features.")

print(f"\nPASS: {len(features)} input features recorded.")

# Validate the model comparison table.
print("\n=== 3. Model Metrics ===")

metrics = pd.read_csv(ROOT / "model_metrics.csv")

expected_models = {
    "Logistic Regression",
    "Decision Tree",
    "K-Nearest Neighbors",
    "Gaussian Naive Bayes",
    "Support Vector Machine",
}

required_columns = {
    "Model", "Accuracy", "AUC", "Precision",
    "Recall", "F1 Score", "MCC",
}

if len(metrics) != 5:
    raise ValueError(f"Expected 5 models; found {len(metrics)}.")

if not required_columns.issubset(metrics.columns):
    raise ValueError("One or more required metric columns are missing.")

if set(metrics["Model"]) != expected_models:
    raise ValueError("The model names do not match the required five models.")

if metrics["Model"].duplicated().any():
    raise ValueError("Duplicate model names found.")

for column in required_columns - {"Model"}:
    metrics[column] = pd.to_numeric(metrics[column], errors="coerce")

for column in ["Accuracy", "Precision", "Recall", "F1 Score"]:
    values = metrics[column].to_numpy(dtype=float)

    if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
        raise ValueError(f"Invalid values found in {column}.")

for column, lower in [("AUC", 0), ("MCC", -1)]:
    values = metrics[column].dropna().to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite values found in {column}.")

    if not ((values >= lower) & (values <= 1)).all():
        raise ValueError(f"Values outside the valid range in {column}.")

if metrics[["AUC", "MCC"]].isna().any().any():
    print("WARNING: AUC or MCC contains a missing value. Investigate it.")

print(metrics.round(4).to_string(index=False))

# Validate the held-out test CSV.
print("\n=== 4. Test Dataset ===")

test_data = pd.read_csv(ROOT / "test_data.csv")

if test_data.empty:
    raise ValueError("test_data.csv is empty.")

if "Class" not in test_data.columns:
    raise ValueError("Target column 'Class' is missing.")

missing_features = [
    feature for feature in features if feature not in test_data.columns
]

if missing_features:
    raise ValueError(f"Missing test features: {missing_features}")

if test_data[features + ["Class"]].isna().any().any():
    raise ValueError("Missing values found in the test dataset.")

test_data[features].apply(pd.to_numeric, errors="raise")

print(f"PASS: {len(test_data)} test rows.")
print(f"PASS: {len(features)} expected features and Class target.")

print("\nAll project validation checks passed!")
