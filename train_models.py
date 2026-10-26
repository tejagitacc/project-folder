
"""
ML Assignment 2: Train and evaluate five classification models.

The training data is used only for fitting the models.
The held-out test data is exported for evaluation in Streamlit.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    classification_report,
)

# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Dry_Bean_Dataset.xlsx"
MODEL_DIR = BASE_DIR / "model"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Load and inspect the dataset
# --------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Place Dry_Bean_Dataset.xlsx in the project folder."
    )

df = pd.read_excel(DATA_PATH)

# Remove accidental whitespace from column names.
df.columns = df.columns.astype(str).str.strip()

# The UCI Dry Bean dataset uses 'Class' as its target.
target_column = "Class"

if target_column not in df.columns:
    raise ValueError(
        f"Target column '{target_column}' not found. "
        f"Available columns: {df.columns.tolist()}"
    )

# Remove rows with missing values, if any.
df = df.dropna().reset_index(drop=True)

# Separate input features and target labels.
X = df.drop(columns=[target_column])
y = df[target_column].astype(str).str.strip()

# Validate that all input features are numeric.
X = X.apply(pd.to_numeric, errors="raise")

if X.shape[1] < 12:
    raise ValueError(
        f"At least 12 features are required; found {X.shape[1]}."
    )

if len(df) < 500:
    raise ValueError(
        f"At least 500 instances are required; found {len(df)}."
    )

if y.nunique() < 2:
    raise ValueError("The target must contain at least two classes.")

print("=" * 65)
print("DRY BEAN DATASET")
print("=" * 65)
print(f"Dataset shape: {df.shape}")
print(f"Number of instances: {len(df)}")
print(f"Number of features: {X.shape[1]}")
print(f"Number of target classes: {y.nunique()}")
print("\nClass distribution:")
print(y.value_counts())

# --------------------------------------------------
# 3. Split into training and test datasets
# --------------------------------------------------

# Stratification preserves class proportions in both sets.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print(f"\nTraining instances: {len(X_train)}")
print(f"Test instances: {len(X_test)}")

# Export the held-out test rows, including their true labels.
# Streamlit will use these labels to calculate test metrics.
test_df = X_test.copy()
test_df[target_column] = y_test
test_df.to_csv(BASE_DIR / "test_data.csv", index=False)

# Save the input feature names for future validation.
joblib.dump(
    X.columns.tolist(),
    MODEL_DIR / "feature_names.joblib",
)

# --------------------------------------------------
# 4. Define the five classification models
# --------------------------------------------------

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=3000,
            random_state=42,
        )),
    ]),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
    ),

    "K-Nearest Neighbors": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(
            n_neighbors=5,
        )),
    ]),

    "Gaussian Naive Bayes": GaussianNB(),

    "Support Vector Machine": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            probability=True,
            random_state=42,
        )),
    ]),
}

# --------------------------------------------------
# 5. Train, evaluate, and save each model
# --------------------------------------------------

results = []

for model_name, model in models.items():

    print("\n" + "=" * 65)
    print(f"Training: {model_name}")
    print("=" * 65)

    # Fit using training data only.
    model.fit(X_train, y_train)

    # Generate predictions on the held-out test set.
    y_pred = model.predict(X_test)

    # Calculate class probabilities for multiclass AUC.
    y_prob = model.predict_proba(X_test)

    # Use the estimator's class order for AUC calculation.
    fitted_classifier = (
        model.named_steps["classifier"]
        if isinstance(model, Pipeline)
        else model
    )
    class_labels = fitted_classifier.classes_

    try:
        auc = roc_auc_score(
            y_test,
            y_prob,
            labels=class_labels,
            multi_class="ovr",
            average="macro",
        )
    except ValueError as exc:
        print(f"AUC could not be calculated: {exc}")
        auc = np.nan

    # Calculate all required evaluation metrics.
    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": auc,
        "Precision": precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }

    results.append(metrics)

    # Save the complete pipeline or model.
    # Pipelines include the scaler when scaling is required.
    filename = model_name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(model, MODEL_DIR / filename)

    print(pd.Series(metrics).to_string())
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0,
        )
    )
    print(f"Saved model: model/{filename}")

# --------------------------------------------------
# 6. Save the model comparison table
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    BASE_DIR / "model_metrics.csv",
    index=False,
)

print("\n" + "=" * 85)
print("FINAL MODEL COMPARISON")
print("=" * 85)
print(results_df.round(4).to_string(index=False))

# Rank by weighted F1 score as a practical overall comparison.
# Consider AUC, MCC, and other metrics as well.
winner = results_df.loc[results_df["F1 Score"].idxmax()]

print("\nBest model by weighted F1 score:")
print(winner["Model"])
print(f"Weighted F1 score: {winner['F1 Score']:.4f}")

print("\nTraining and evaluation completed successfully.")
print("Generated: test_data.csv")
print("Generated: model_metrics.csv")
print("Saved all five models in: model/")
