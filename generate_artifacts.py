import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

def main():
    os.makedirs('static/images', exist_ok=True)
    
    # 1. Load data and remove duplicates
    df_raw = pd.read_csv('milknew.csv')
    df_raw.columns = df_raw.columns.str.strip()
    df = df_raw.drop_duplicates().reset_index(drop=True)
    
    classes = ['high', 'medium', 'low']
    
    # Class Distribution Plot
    plt.figure(figsize=(7, 4.5))
    ax = sns.countplot(data=df, x='Grade', order=classes, palette=['#2ca02c', '#ff7f0e', '#d62728'])
    plt.title('Target Class Distribution After Deduplication (N=83)', fontsize=12, fontweight='bold')
    plt.xlabel('Quality Grade')
    plt.ylabel('Number of Unique Samples')
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontweight='bold')
    plt.tight_layout()
    plt.savefig('static/images/class_distribution.png', dpi=300)
    plt.close()
    
    X = df.drop(columns=['Grade'])
    y = df['Grade']
    
    # Stratified 70/30 split (58 train, 25 test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, 'scaler.joblib')
    
    # Baseline Model (from Wijaya et al., 2023)
    rf_base = RandomForestClassifier(
        criterion='gini',
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    rf_base.fit(X_train_scaled, y_train)
    y_pred_base = rf_base.predict(X_test_scaled)
    
    # Tuned Model (Best parameters from Colab)
    best_rf = RandomForestClassifier(
        criterion='gini',
        n_estimators=20,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    best_rf.fit(X_train_scaled, y_train)
    y_pred_opt = best_rf.predict(X_test_scaled)
    joblib.dump(best_rf, 'best_random_forest_model.joblib')
    
    # Metrics
    base_acc = accuracy_score(y_test, y_pred_base)
    base_prec = precision_score(y_test, y_pred_base, average='weighted', zero_division=0)
    base_rec = recall_score(y_test, y_pred_base, average='weighted', zero_division=0)
    base_f1 = f1_score(y_test, y_pred_base, average='weighted', zero_division=0)
    
    opt_acc = accuracy_score(y_test, y_pred_opt)
    opt_prec = precision_score(y_test, y_pred_opt, average='weighted', zero_division=0)
    opt_rec = recall_score(y_test, y_pred_opt, average='weighted', zero_division=0)
    opt_f1 = f1_score(y_test, y_pred_opt, average='weighted', zero_division=0)
    
    # Plot 1: Confusion Matrices
    cm_base = confusion_matrix(y_test, y_pred_base, labels=classes)
    cm_opt = confusion_matrix(y_test, y_pred_opt, labels=classes)
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=axes[0])
    axes[0].set_title(f'Baseline Random Forest\n(Test Accuracy: {base_acc*100:.2f}%)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Predicted Quality Grade')
    axes[0].set_ylabel('Actual Quality Grade')
    
    sns.heatmap(cm_opt, annot=True, fmt='d', cmap='Greens', xticklabels=classes, yticklabels=classes, ax=axes[1])
    axes[1].set_title(f'Tuned Random Forest (Best HPO)\n(Test Accuracy: {opt_acc*100:.2f}%)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Predicted Quality Grade')
    axes[1].set_ylabel('Actual Quality Grade')
    plt.tight_layout()
    plt.savefig('static/images/confusion_matrices_comparison.png', dpi=300)
    plt.close()
    
    # Plot 2: Performance Comparison Bar Chart
    comp_df = pd.DataFrame({
        'Baseline RF': [base_acc * 100, base_prec * 100, base_rec * 100, base_f1 * 100],
        'Tuned RF': [opt_acc * 100, opt_prec * 100, opt_rec * 100, opt_f1 * 100]
    }, index=['Accuracy', 'Precision', 'Recall', 'F1-Score'])
    
    ax = comp_df.plot(kind='bar', figsize=(9, 5.5), color=['#1f77b4', '#ff7f0e'], edgecolor='black')
    plt.ylim(0, 100)
    plt.ylabel('Score (%)')
    plt.title('Baseline vs. Tuned Random Forest Performance (Test Set: N=25)', fontsize=13, fontweight='bold')
    plt.xticks(rotation=0)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f%%', padding=3, fontweight='bold', fontsize=9)
    plt.legend(title='Model')
    plt.tight_layout()
    plt.savefig('static/images/baseline_vs_tuned_comparison.png', dpi=300)
    plt.close()
    
    # Plot 3: Feature Importances
    feat_imp = pd.Series(best_rf.feature_importances_, index=X.columns).sort_values(ascending=True)
    plt.figure(figsize=(9, 5))
    feat_imp.plot(kind='barh', color='#2b5c8f', edgecolor='black', alpha=0.9)
    plt.title('Feature Importances - Tuned Random Forest Model', fontsize=13, fontweight='bold')
    plt.xlabel('Importance Score')
    plt.ylabel('Milk Quality Attribute')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    for i, v in enumerate(feat_imp):
        plt.text(v + 0.005, i, f"{v*100:.2f}%", va='center', fontweight='bold', fontsize=9)
    plt.tight_layout()
    plt.savefig('static/images/feature_importance.png', dpi=300)
    plt.close()
    
    print("All artifacts successfully created and saved in static/images!")

if __name__ == '__main__':
    main()
