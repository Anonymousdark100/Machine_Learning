# Week 4 — Machine Learning Model Development and Evaluation

## Objective
Train, compare and evaluate two classification models using a public dataset.

## Dataset
Wisconsin Diagnostic Breast Cancer dataset bundled with scikit-learn. It has 569 records and 30 numeric features. The positive class in this project is **malignant (1)** and the negative class is **benign (0)**.

## Models
1. Logistic Regression — standardized features and a reproducible pipeline.
2. Decision Tree — limited depth to reduce overfitting.

## Methodology
- Stratified 75/25 train/test split with `random_state=42`.
- Median imputation pipeline (reproducible even if future data has missing values).
- StandardScaler for Logistic Regression.
- Five-fold cross-validation on the training split using ROC-AUC.
- Final comparison on the held-out test split.
- Metrics: accuracy, precision, recall, F1-score, ROC-AUC.
- Visualizations: metric comparison, confusion matrix, ROC curves.

## Run
```bash
pip install -r requirements.txt
python train_evaluate.py
```

## Important limitation
This is an educational model trained on a small, historical dataset. It is not validated for clinical use and must not be used to diagnose or make medical decisions.
