
"""
ML Assignment 2
Interactive evaluation dashboard for five saved classifiers.

The app loads trained models and evaluates uploaded test data.
It does not fit or retrain any model.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

# --------------------------------------------------
# 1. Configure the Streamlit page
# --------------------------------------------------

st.set_page_config(
    page_title="Dry Bean Classifier Lab",
    page_icon="🫘",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
TARGET_COLUMN = "Class"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "k-nearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

# --------------------------------------------------
# 2. Load saved artifacts without retraining
# --------------------------------------------------

@st.cache_resource
def load_artifacts():
    """Load trained models and the expected feature names."""

    models = {}

    for display_name, filename in MODEL_FILES.items():
        model_path = MODEL_DIR / filename

        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing model file: {model_path}"
            )

        models[display_name] = joblib.load(model_path)

    feature_path = MODEL_DIR / "feature_names.joblib"

    if not feature_path.exists():
        raise FileNotFoundError(
            f"Missing feature list: {feature_path}"
        )

    feature_names = joblib.load(feature_path)

    return models, feature_names


# --------------------------------------------------
# 3. Define evaluation functions
# --------------------------------------------------

def get_classifier(model):
    """Return the final estimator from a pipeline, if applicable."""

    if hasattr(model, "named_steps"):
        return model.named_steps["classifier"]

    return model


def evaluate_model(model, X_test, y_test):
    """Calculate the required metrics without fitting the model."""

    y_pred = model.predict(X_test)

    classifier = get_classifier(model)
    class_labels = classifier.classes_

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": np.nan,
        "Precision": precision_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "Recall": recall_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "F1 Score": f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }

    # Calculate macro one-vs-rest AUC from class probabilities.
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)

            metrics["AUC"] = roc_auc_score(
                y_test,
                y_prob,
                labels=class_labels,
                multi_class="ovr",
                average="macro",
            )
        except ValueError:
            # AUC may be undefined if the test data lacks a class.
            metrics["AUC"] = np.nan

    return y_pred, metrics, class_labels


# --------------------------------------------------
# 4. Build the application interface
# --------------------------------------------------

st.title("🫘 Dry Bean Classification Dashboard")

st.markdown(
    """
    Evaluate five machine learning classifiers on labelled test data.
    Upload a CSV containing the 16 input features and the `Class` target.

    **Evaluation policy:** models are loaded from disk and are never
    retrained on uploaded data.
    """
)

with st.sidebar:
    st.header("Model settings")

    selected_model_name = st.selectbox(
        "Select a classifier",
        list(MODEL_FILES.keys()),
    )

    st.caption(
        "All models use the same uploaded test dataset for comparison."
    )

# Load saved models.
try:
    models, feature_names = load_artifacts()
except (FileNotFoundError, OSError, ValueError) as exc:
    st.error(f"Could not load the saved models: {exc}")
    st.info(
        "Run train_models.py successfully and check that the model "
        "folder contains all five .joblib files and feature_names.joblib."
    )
    st.stop()

# --------------------------------------------------
# 5. Upload the labelled test CSV
# --------------------------------------------------

st.subheader("1. Upload test data")

uploaded_file = st.file_uploader(
    "Choose a labelled test CSV file",
    type=["csv"],
    help=(
        "The CSV must include every expected feature and the "
        "Class target column."
    ),
)

# Use the bundled test dataset by default so results appear immediately
# when the app opens. If the evaluator uploads another CSV, evaluate that
# file instead. Neither path retrains the saved models.
if uploaded_file is not None:
    data_source_label = f"Uploaded file: {uploaded_file.name}"
    try:
        test_df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Unable to read the uploaded CSV: {exc}")
        st.stop()
else:
    default_test_path = BASE_DIR / "test_data.csv"
    data_source_label = "Default test dataset: test_data.csv"

    if not default_test_path.exists():
        st.error(
            "The default test_data.csv file was not found. "
            "Please add it to the project root or upload a labelled test CSV."
        )
        st.stop()

    try:
        test_df = pd.read_csv(default_test_path)
    except Exception as exc:
        st.error(f"Unable to read the default test_data.csv: {exc}")
        st.stop()

st.caption(data_source_label)

test_df.columns = test_df.columns.astype(str).str.strip()

if test_df.empty:
    st.error("The uploaded CSV contains no rows.")
    st.stop()

if test_df.columns.duplicated().any():
    st.error("The CSV contains duplicate column names.")
    st.stop()

if TARGET_COLUMN not in test_df.columns:
    st.error(
        f"The CSV must contain the target column '{TARGET_COLUMN}'."
    )
    st.stop()

missing_features = [
    column for column in feature_names
    if column not in test_df.columns
]

if missing_features:
    st.error(
        "The CSV is missing required input features: "
        + ", ".join(missing_features)
    )
    st.stop()

# Select features in exactly the same order used during training.
try:
    X_test = test_df[feature_names].apply(
        pd.to_numeric, errors="raise"
    )
except (ValueError, TypeError) as exc:
    st.error(f"Input features must be numeric: {exc}")
    st.stop()

if not np.isfinite(X_test.to_numpy(dtype=float)).all():
    st.error(
        "The input features contain missing or infinite values. "
        "Please correct the CSV and upload it again."
    )
    st.stop()

y_test = test_df[TARGET_COLUMN].astype(str).str.strip()

if y_test.eq("").any():
    st.error("The target column contains blank labels.")
    st.stop()

# Ensure that labels match the classes known to the trained models.
known_classes = set(
    str(label) for label in get_classifier(
        models[selected_model_name]
    ).classes_
)

unknown_labels = sorted(set(y_test) - known_classes)

if unknown_labels:
    st.error(
        "The target column contains labels not recognized by the "
        f"selected trained model: {unknown_labels}"
    )
    st.stop()

# Show basic dataset information.
st.success("Test data loaded and validated successfully.")

col1, col2, col3 = st.columns(3)
col1.metric("Test instances", f"{len(test_df):,}")
col2.metric("Input features", len(feature_names))
col3.metric("Target classes", y_test.nunique())

with st.expander("Preview uploaded data"):
    st.dataframe(test_df.head(10), use_container_width=True)

# --------------------------------------------------
# 6. Evaluate all five saved models
# --------------------------------------------------

st.subheader("2. Model performance comparison")

all_metrics = {}
predictions = {}
model_classes = {}

for model_name, model in models.items():
    try:
        y_pred, metrics, class_labels = evaluate_model(
            model, X_test, y_test
        )

        all_metrics[model_name] = metrics
        predictions[model_name] = y_pred
        model_classes[model_name] = class_labels

    except (ValueError, TypeError, AttributeError) as exc:
        st.error(f"Evaluation failed for {model_name}: {exc}")
        st.stop()

comparison_df = pd.DataFrame.from_dict(
    all_metrics, orient="index"
)
comparison_df.index.name = "Model"

# Show all six metrics for all five classifiers.
st.dataframe(
    comparison_df.style.format("{:.4f}", na_rep="N/A"),
    use_container_width=True,
)

# Visual comparison of the five classifiers.
st.markdown("#### Metric comparison chart")

metric_to_plot = st.selectbox(
    "Choose a metric to compare",
    ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"],
)

chart_data = comparison_df[metric_to_plot].dropna()

if chart_data.empty:
    st.warning(f"No valid values are available for {metric_to_plot}.")
else:
    st.bar_chart(chart_data)

# --------------------------------------------------
# 7. Show detailed results for the selected model
# --------------------------------------------------

st.subheader("3. Selected model evaluation")
st.write(f"**Selected model:** {selected_model_name}")

selected_metrics = all_metrics[selected_model_name]

metric_names = [
    "Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"
]

metric_columns = st.columns(3)

for index, metric_name in enumerate(metric_names):
    value = selected_metrics[metric_name]

    display_value = (
        f"{value:.4f}" if np.isfinite(value) else "N/A"
    )

    metric_columns[index % 3].metric(
        metric_name, display_value
    )

st.caption(
    "Precision, Recall and F1 use weighted averaging. "
    "AUC uses macro one-vs-rest averaging. MCC is the "
    "multiclass Matthews correlation coefficient."
)

# --------------------------------------------------
# 8. Confusion matrix and classification report
# --------------------------------------------------

y_pred = predictions[selected_model_name]
class_labels = model_classes[selected_model_name]

st.subheader("4. Confusion matrix")

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=class_labels,
)

fig, ax = plt.subplots(figsize=(10, 7))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_labels,
    yticklabels=class_labels,
    ax=ax,
)

ax.set_xlabel("Predicted class")
ax.set_ylabel("Actual class")
ax.set_title(f"Confusion Matrix — {selected_model_name}")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
fig.tight_layout()

st.pyplot(fig)
plt.close(fig)

st.subheader("5. Classification report")

report = classification_report(
    y_test,
    y_pred,
    labels=class_labels,
    zero_division=0,
    output_dict=True,
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df.style.format("{:.4f}", na_rep="N/A"),
    use_container_width=True,
)

# --------------------------------------------------
# 9. Download the comparison results
# --------------------------------------------------

st.subheader("6. Export results")

csv_results = (
    comparison_df
    .reset_index()
    .to_csv(index=False)
    .encode("utf-8")
)

st.download_button(
    label="Download model comparison as CSV",
    data=csv_results,
    file_name="uploaded_test_model_comparison.csv",
    mime="text/csv",
)

st.caption(
    "Results are calculated from the uploaded labelled test data. "
    "The saved models are not retrained."
)
