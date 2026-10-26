## ML Assignment-2
## BITS ID: 2026AJ05485
## Name: Tejasree Pakala
## Dry Bean Classification Dashboard

## 1. Problem Statement

This project trains and compares five machine-learning classifiers to predict the class of a dry bean from its physical measurements. A Streamlit dashboard lets users upload labelled test data, evaluate the saved classifiers, inspect a confusion matrix and classification report, and compare model performance.

## 2. Dataset Description

- **Dataset:** Dry Bean Dataset
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/602/dry+bean)
- **Original instances:** 13,611
- **Input features:** 16 numerical features
- **Target column:** `Class`
- **Task:** Multiclass classification
- **Target classes:** 7

The dataset is split into training and testing subsets using an 80:20 split, `random_state=42`, and stratification. Models are fitted on the training subset. The held-out test subset is exported to `test_data.csv`. The Streamlit application loads saved models and does not retrain them when a test CSV is uploaded.

## 3. GitHub and Live Application

- **GitHub repository:** https://github.com/tejagitacc/project-folder
- **Deployed Streamlit application:** https://project-folder-2026aj05485.streamlit.app/


## 4. Classification Models

The project implements the following five classifiers:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

Scaling is included in the saved pipelines for Logistic Regression, KNN, and SVM. The Decision Tree and Gaussian Naive Bayes models are saved without a scaling step.

## 5. Evaluation Metrics

All five models were evaluated on the same held-out test split. The metrics are defined as follows:

- **Accuracy:** Fraction of test predictions that are correct.
- **AUC:** Macro-averaged one-vs-rest multiclass ROC-AUC, calculated from predicted class probabilities.
- **Precision:** Weighted-average precision across classes.
- **Recall:** Weighted-average recall across classes.
- **F1 Score:** Weighted-average F1 score across classes.
- **MCC:** Matthews correlation coefficient.

Precision, Recall, and F1 use weighted averaging. AUC uses macro one-vs-rest averaging. These averaging methods should be kept consistent when comparing models.

### 5.1 Model Comparison Results

The following are the measured results from the training script's held-out test split.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.921410 | 0.994776 | 0.922178 | 0.921410 | 0.921590 | 0.905045 |
| Decision Tree | 0.891664 | 0.944808 | 0.891247 | 0.891664 | 0.891322 | 0.869059 |
| K-Nearest Neighbors | 0.916636 | 0.983317 | 0.917409 | 0.916636 | 0.916805 | 0.899204 |
| Gaussian Naive Bayes | 0.763863 | 0.967193 | 0.765371 | 0.763863 | 0.761474 | 0.715406 |
| Support Vector Machine | 0.922145 | 0.994927 | 0.922426 | 0.922145 | 0.922177 | 0.905870 |

### 5.2 Performance Observations

- **Logistic Regression:** Achieved an accuracy of 0.921410 and weighted F1 score of 0.921590. Its macro one-vs-rest AUC was 0.994776, indicating strong class-ranking performance on this test split.
- **Decision Tree:** Achieved an accuracy of 0.891664 and weighted F1 score of 0.891322. Among the five models, it performed below Logistic Regression, KNN, and SVM on accuracy and weighted F1.
- **K-Nearest Neighbors:** Achieved an accuracy of 0.916636 and weighted F1 score of 0.916805. Its results were close to Logistic Regression and SVM, although its AUC of 0.983317 was lower than theirs.
- **Gaussian Naive Bayes:** Achieved an accuracy of 0.763863 and weighted F1 score of 0.761474, the lowest values among these five models. Its AUC was 0.967193, showing that its AUC ranking result was stronger than its final hard-label accuracy might suggest.
- **Support Vector Machine:** Achieved the highest accuracy (0.922145), AUC (0.994927), weighted precision (0.922426), weighted F1 score (0.922177), and MCC (0.905870) among the five models. Its accuracy and F1 advantage over Logistic Regression is small.

### 5.3 Best-Performing Model

**Best model on this test split: Support Vector Machine (SVM).**

SVM had the highest measured Accuracy, AUC, Precision, F1 Score, and MCC among the five models. Its Recall was 0.922145, matching its Accuracy because weighted recall equals accuracy in this single-label multiclass setting. Logistic Regression was a very close second, so the observed difference between these two models is small.

This conclusion applies to the current held-out split and selected model settings. It does not establish that SVM will always outperform the other classifiers on every dataset or split.

## 6. Streamlit Application Features

The Streamlit dashboard includes:

1. **CSV upload:** Accepts a labelled test CSV with the expected 16 features and the `Class` target column.
2. **Model selection:** Lets the user select a classifier in the sidebar.
3. **Six evaluation metrics:** Displays Accuracy, AUC, Precision, Recall, F1 Score, and MCC.
4. **Model comparison:** Evaluates all five saved models on the same uploaded rows and displays a comparison table.
5. **Metric chart:** Displays a bar chart for a selected metric.
6. **Confusion matrix:** Visualizes actual and predicted class counts for the selected model.
7. **Classification report:** Shows class-wise precision, recall, F1 score, and support.
8. **CSV export:** Allows the comparison table to be downloaded.

The app validates the uploaded file for required columns, numeric and finite feature values, non-empty rows, and target labels known to the selected model. Models are loaded from the `model/` directory; uploading data does not trigger model fitting or retraining.

## 7. Project Structure

```text
PROJECT-FOLDER/
├── app.py
├── train_models.py
├── validate_project.py
├── requirements.txt
├── README.md
├── Dry_Bean_Dataset.xlsx
├── test_data.csv
├── model_metrics.csv
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── k-nearest_neighbors.joblib
    ├── gaussian_naive_bayes.joblib
    ├── support_vector_machine.joblib
    └── feature_names.joblib
```

## 8. Installation and Execution

### Requirements

Use a compatible Python version and pip.

### Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Train the models and regenerate artifacts

Run this command when reproducing the training process:

```bash
python train_models.py
```

This script trains the five classifiers, saves the models, creates `test_data.csv`, and writes `model_metrics.csv`.

### Validate project artifacts

```bash
python validate_project.py
```

### Start the Streamlit app

```bash
python -m streamlit run app.py
```

Open the local URL printed in the terminal, upload `test_data.csv`, and inspect the model comparison and detailed evaluation sections.

## 9. Conclusion

Five classifiers were trained and evaluated for multiclass Dry Bean classification. On the reported held-out test split, SVM achieved the strongest overall results, with accuracy of 0.922145, macro one-vs-rest AUC of 0.994927, weighted F1 score of 0.922177, and MCC of 0.905870. Logistic Regression performed nearly as well, while KNN also produced competitive results. Decision Tree had lower performance than those three models, and Gaussian Naive Bayes had the lowest accuracy and weighted F1 score in this comparison.

The Streamlit dashboard makes the saved models accessible for evaluation on other labelled test CSV files. Results on a newly uploaded dataset may differ from the values reported above because they are recalculated for the uploaded rows.
