"""
Week 4: Machine Learning Model Development and Evaluation
Dataset: Wisconsin Diagnostic Breast Cancer dataset bundled with scikit-learn.
Positive class: malignant (1); negative class: benign (0).
Run from this project folder:
    pip install -r requirements.txt
    python train_evaluate.py
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    ConfusionMatrixDisplay, roc_curve, roc_auc_score
)

OUT = Path("visualizations")
OUT.mkdir(exist_ok=True)

data = load_breast_cancer(as_frame=True)
X = data.data
y = (data.target == 0).astype(int)  # malignant=1, benign=0

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

models = {
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000, random_state=42))
    ]),
    "Decision Tree": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=42))
    ])
}

metrics = []
probabilities = {}
predictions = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    positive_col = list(model.classes_).index(1)
    prob = model.predict_proba(X_test)[:, positive_col]
    predictions[name] = pred
    probabilities[name] = prob
    metrics.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1-score": f1_score(y_test, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, prob)
    })

metrics_df = pd.DataFrame(metrics)
print(metrics_df.to_string(index=False))
metrics_df.to_csv("model_metrics.csv", index=False)

# Cross-validation is performed on the training set only.
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
    print(f"{name} 5-fold training ROC-AUC: {scores.mean():.3f} ± {scores.std():.3f}")

# Model comparison chart
ax = metrics_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]].plot(
    kind="bar", figsize=(10, 6)
)
ax.set_title("Test-Set Performance Comparison")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUT / "01_model_comparison.png", dpi=180)
plt.close()

# Confusion matrix for logistic regression
plt.figure(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, predictions["Logistic Regression"], labels=[0, 1],
    display_labels=["Benign", "Malignant"], colorbar=False
)
plt.title("Logistic Regression — Confusion Matrix")
plt.tight_layout()
plt.savefig(OUT / "02_logistic_confusion_matrix.png", dpi=180)
plt.close()

# ROC curves
plt.figure(figsize=(7, 6))
for name in models:
    fpr, tpr, _ = roc_curve(y_test, probabilities[name])
    auc = roc_auc_score(y_test, probabilities[name])
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "03_roc_curves.png", dpi=180)
plt.close()

print("\nNote: This is an educational dataset and model, not a medical diagnostic tool.")
