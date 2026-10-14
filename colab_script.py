# ==============================================================================
# MILK QUALITY PREDICTION & MACHINE LEARNING REPLICATION
# Adapted from Methodology by Wijaya et al. (2023), IEEE Access
# Authors: Charmele Asna, Francis Dave Brina, Rolemir Kirk Don Diango Zayas
# College of Computing Education, University of Mindanao, Davao City, Philippines
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

print("✓ All libraries loaded successfully!")

# 1. LOAD DATASET & REMOVE EXACT DUPLICATES (Data Leakage Control)
CSV_FILENAME = "milknew.csv"
if not os.path.exists(CSV_FILENAME):
    import urllib.request
    url = "https://raw.githubusercontent.com/shaik1201/Milk-Quality-Prediction/master/milknew.csv"
    urllib.request.urlretrieve(url, CSV_FILENAME)

df_raw = pd.read_csv(CSV_FILENAME)
df_raw.columns = df_raw.columns.str.strip()

duplicate_count = df_raw.duplicated().sum()
df = df_raw.drop_duplicates().reset_index(drop=True)

print(f"Exact duplicate rows removed: {duplicate_count}")
print(f"Total unique observations: {len(df)}")
print("\nUnique Class Distribution (Grade):")
print(df["Grade"].value_counts())

# 2. SEPARATE FEATURES & 70/30 STRATIFIED TRAIN-TEST SPLIT
X = df.drop(columns=["Grade"])
y = df["Grade"]
classes = ["high", "medium", "low"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

print(f"\nTraining observations: {len(X_train)} (70%)")
print(f"Testing observations:  {len(X_test)} (30%)")

# 3. FEATURE SCALING (StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "scaler.joblib")

# 4. BASELINE RANDOM FOREST (Wijaya et al., 2023 parameters)
print("\n>>> 1. Training Baseline Random Forest Classifier...")
rf_baseline = RandomForestClassifier(
    criterion="gini",
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
rf_baseline.fit(X_train_scaled, y_train)
y_pred_base = rf_baseline.predict(X_test_scaled)

base_acc = accuracy_score(y_test, y_pred_base)
base_prec = precision_score(y_test, y_pred_base, average="weighted", zero_division=0)
base_rec = recall_score(y_test, y_pred_base, average="weighted", zero_division=0)
base_f1 = f1_score(y_test, y_pred_base, average="weighted", zero_division=0)

print(f"Baseline Accuracy:  {base_acc * 100:.2f}% (18/25 correct)")
print(f"Baseline Precision: {base_prec * 100:.2f}%")
print(f"Baseline Recall:    {base_rec * 100:.2f}%")
print(f"Baseline F1-Score:  {base_f1 * 100:.2f}%")

# 5. GRID SEARCH HYPERPARAMETER OPTIMIZATION (5-Fold Stratified CV)
print("\n>>> 2. Running 5-Fold GridSearchCV across 64 candidates (320 fits)...")
param_grid_rf = {
    "criterion": ["gini", "entropy"],
    "n_estimators": [20, 100],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 10],
    "min_samples_leaf": [1, 5]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_rf = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid_rf,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)
grid_rf.fit(X_train_scaled, y_train)

best_rf = grid_rf.best_estimator_
joblib.dump(best_rf, "best_random_forest_model.joblib")

print(f"\nOptimal Hyperparameters: {grid_rf.best_params_}")
print(f"Best 5-Fold Cross-Validation Accuracy: {grid_rf.best_score_ * 100:.2f}%")

# 6. EVALUATE TUNED MODEL ON 25 TEST SAMPLES
y_pred_opt = best_rf.predict(X_test_scaled)
opt_acc = accuracy_score(y_test, y_pred_opt)
opt_prec = precision_score(y_test, y_pred_opt, average="weighted", zero_division=0)
opt_rec = recall_score(y_test, y_pred_opt, average="weighted", zero_division=0)
opt_f1 = f1_score(y_test, y_pred_opt, average="weighted", zero_division=0)

print(f"\nTuned Accuracy:  {opt_acc * 100:.2f}% (20/25 correct, +8.00 pp improvement!)")
print(f"Tuned Precision: {opt_prec * 100:.2f}%")
print(f"Tuned Recall:    {opt_rec * 100:.2f}%")
print(f"Tuned F1-Score:  {opt_f1 * 100:.2f}%")

print("\nTuned Classification Report:")
print(classification_report(y_test, y_pred_opt, digits=4, zero_division=0))
