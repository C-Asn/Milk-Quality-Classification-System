import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

def run_experiment():
    print("=================================================================")
    print("      MILK QUALITY CLASSIFICATION & BENCHMARKING EXPERIMENT      ")
    print("  Replication of Methodology from Wijaya et al. (2023) IEEE Access ")
    print("=================================================================\n")

    # 1. Load Dataset
    data_path = 'milknew.csv'
    if not os.path.exists(data_path):
        import urllib.request
        url = 'https://raw.githubusercontent.com/shaik1201/Milk-Quality-Prediction/master/milknew.csv'
        print(f"Downloading dataset from {url}...")
        urllib.request.urlretrieve(url, data_path)

    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip() # Strip whitespace, e.g. 'Fat ' -> 'Fat'

    print(f"Dataset Loaded Successfully! Shape: {df.shape}")
    print("Features:", list(df.columns[:-1]))
    print("Target Distribution:\n", df['Grade'].value_counts())
    print("-" * 65)

    X = df.drop('Grade', axis=1)
    y = df['Grade']
    classes = ['high', 'medium', 'low']

    # 2. Train-Test Split (70% Train, 30% Test Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set:  {X_test.shape[0]} samples\n")

    # 3. Feature Scaling (StandardScaler as in Wijaya et al., Eq. 1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, 'scaler.joblib')
    print("StandardScaler fitted and saved as 'scaler.joblib'.\n")

    # 4. Model 1: Baseline Random Forest (Default Scikit-Learn configuration)
    print(">>> 1. Training Baseline Random Forest Classifier...")
    rf_baseline = RandomForestClassifier(random_state=42)
    rf_baseline.fit(X_train_scaled, y_train)
    y_pred_base = rf_baseline.predict(X_test_scaled)

    acc_base = accuracy_score(y_test, y_pred_base)
    prec_base = precision_score(y_test, y_pred_base, average='weighted')
    rec_base = recall_score(y_test, y_pred_base, average='weighted')
    f1_base = f1_score(y_test, y_pred_base, average='weighted')

    print(f"Baseline RF Results -> Accuracy: {acc_base:.4f} | Precision: {prec_base:.4f} | Recall: {rec_base:.4f} | F1: {f1_base:.4f}")
    print("\nBaseline Classification Report:\n", classification_report(y_test, y_pred_base, digits=4))

    # 5. Model 2: Hyperparameter Optimized Random Forest (Grid Search with 5-fold CV)
    print(">>> 2. Hyperparameter Optimization for Random Forest via GridSearchCV...")
    param_grid_rf = {
        'n_estimators': [50, 100, 150, 200],
        'criterion': ['gini', 'entropy'],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_rf = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid_rf,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    grid_rf.fit(X_train_scaled, y_train)

    best_rf = grid_rf.best_estimator_
    joblib.dump(best_rf, 'best_random_forest_model.joblib')

    y_pred_opt = best_rf.predict(X_test_scaled)
    acc_opt = accuracy_score(y_test, y_pred_opt)
    prec_opt = precision_score(y_test, y_pred_opt, average='weighted')
    rec_opt = recall_score(y_test, y_pred_opt, average='weighted')
    f1_opt = f1_score(y_test, y_pred_opt, average='weighted')

    print(f"\nBest Hyperparameters Found: {grid_rf.best_params_}")
    print(f"Best CV Accuracy Score:     {grid_rf.best_score_:.4f}")
    print(f"Optimized RF Results -> Accuracy: {acc_opt:.4f} | Precision: {prec_opt:.4f} | Recall: {rec_opt:.4f} | F1: {f1_opt:.4f}")
    print("\nOptimized Classification Report:\n", classification_report(y_test, y_pred_opt, digits=4))

    # 6. Benchmark Comparison with All 7 Baseline Paper Algorithms
    print("-" * 65)
    print(">>> 3. Benchmarking All Algorithms from Baseline Paper (Wijaya et al., 2023)...")
    models_to_test = {
        'Gradient Tree Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.3, random_state=42),
        'Decision Tree': DecisionTreeClassifier(criterion='gini', max_depth=15, min_samples_split=10, min_samples_leaf=5, random_state=42),
        'Support Vector Machine': SVC(C=10, gamma=0.01, kernel='rbf', probability=True, random_state=42),
        'Neural Network (MLP)': MLPClassifier(hidden_layer_sizes=(30, 15, 10), max_iter=300, activation='relu', solver='adam', random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=200, learning_rate=0.3, random_state=42),
        'Baseline Random Forest': rf_baseline,
        'Optimized Random Forest': best_rf,
        'k-Nearest Neighbors (k-NN)': KNeighborsClassifier(n_neighbors=2, weights='distance', metric='manhattan')
    }

    comparison_results = []
    for name, model in models_to_test.items():
        if name not in ['Baseline Random Forest', 'Optimized Random Forest']:
            model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average='weighted', zero_division=0)
        rec = recall_score(y_test, preds, average='weighted', zero_division=0)
        f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
        comparison_results.append({
            'Algorithm': name,
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1-Score': round(f1, 4)
        })

    comp_df = pd.DataFrame(comparison_results)
    print("\nALGORITHM COMPARISON REPORT ON MILK QUALITY DATASET:")
    print(comp_df.to_string(index=False))

    # 7. Generate Visualizations (Plots saved for Paper / UI)
    print("\n>>> 4. Generating Visualizations for Results & Discussion...")
    os.makedirs('static/images', exist_ok=True)

    # 7a. Confusion Matrix Comparison (Baseline vs Optimized RF)
    cm_base = confusion_matrix(y_test, y_pred_base, labels=classes)
    cm_opt = confusion_matrix(y_test, y_pred_opt, labels=classes)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=axes[0])
    axes[0].set_title(f'Baseline Random Forest\n(Accuracy: {acc_base*100:.2f}%)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Predicted Quality Class')
    axes[0].set_ylabel('Actual Quality Class')

    sns.heatmap(cm_opt, annot=True, fmt='d', cmap='Greens', xticklabels=classes, yticklabels=classes, ax=axes[1])
    axes[1].set_title(f'Optimized Random Forest (HPO)\n(Accuracy: {acc_opt*100:.2f}%)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Predicted Quality Class')
    axes[1].set_ylabel('Actual Quality Class')
    plt.tight_layout()
    plt.savefig('static/images/confusion_matrices_comparison.png', dpi=300)
    plt.close()
    print("Saved 'static/images/confusion_matrices_comparison.png'")

    # 7b. Feature Importances Plot
    feat_imp = pd.Series(best_rf.feature_importances_, index=X.columns).sort_values(ascending=True)
    plt.figure(figsize=(9, 5))
    feat_imp.plot(kind='barh', color='#2b5c8f', edgecolor='black')
    plt.title('Feature Importances - Optimized Random Forest Model', fontsize=13, fontweight='bold')
    plt.xlabel('Relative Importance (Gini Impurity Reduction)')
    plt.ylabel('Milk Quality Attributes / Sensors')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('static/images/feature_importance.png', dpi=300)
    plt.close()
    print("Saved 'static/images/feature_importance.png'")

    # 7c. Model Comparison Bar Chart
    plt.figure(figsize=(10, 5))
    bar_positions = np.arange(len(comp_df))
    plt.bar(bar_positions, comp_df['Accuracy'], color='#1f77b4', edgecolor='black', alpha=0.85)
    plt.xticks(bar_positions, comp_df['Algorithm'], rotation=35, ha='right', fontsize=9)
    plt.ylim(0.7, 1.02)
    plt.ylabel('Test Accuracy Score')
    plt.title('Comparison of Classification Algorithms on Milk Quality Dataset', fontsize=13, fontweight='bold')
    for i, v in enumerate(comp_df['Accuracy']):
        plt.text(i, v + 0.005, f"{v*100:.1f}%", ha='center', fontweight='bold', fontsize=8.5)
    plt.tight_layout()
    plt.savefig('static/images/algorithm_comparison.png', dpi=300)
    plt.close()
    print("Saved 'static/images/algorithm_comparison.png'")

    # Save summary dictionary for easy reference in report
    summary = {
        'dataset_size': len(df),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'baseline_rf': {
            'accuracy': acc_base,
            'precision': prec_base,
            'recall': rec_base,
            'f1_score': f1_base
        },
        'optimized_rf': {
            'best_params': grid_rf.best_params_,
            'cv_accuracy': grid_rf.best_score_,
            'accuracy': acc_opt,
            'precision': prec_opt,
            'recall': rec_opt,
            'f1_score': f1_opt
        },
        'feature_importances': feat_imp.to_dict(),
        'comparison': comparison_results
    }

    with open('experiment_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
    print("\nExperiment finished! Summary saved to 'experiment_summary.json'.")

if __name__ == '__main__':
    run_experiment()
