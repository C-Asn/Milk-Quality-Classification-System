import json

def create_colab_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# **Replication Study: Milk Quality Classification Using Random Forest & Hyperparameter Optimization**\n",
                    "### **Authors:** Charmele N. Asna, Francis Dave O. Brina, Rolemir Kirk Don D. Zayas\n",
                    "### **Institution:** College of Computing Education, University of Mindanao, Davao City, Philippines\n",
                    "**Baseline Paper:** Wijaya et al. (2023), *Seafood Quality Detection Using Electronic Nose and Machine Learning Algorithms With Hyperparameter Optimization*, IEEE Access.\n",
                    "**Dataset:** Kaggle Milk Quality Dataset (`cpluzshrijayan/milkquality` / `satishbanne/milk-quality-prediction`)\n",
                    "\n",
                    "---\n",
                    "### **Overview of the Notebook Pipeline**\n",
                    "1. **Environment Setup & Library Importation**\n",
                    "2. **Dataset Acquisition & Exploratory Data Analysis (EDA)**\n",
                    "3. **Data Preprocessing & Feature Scaling (StandardScaler)**\n",
                    "4. **Stratified Train-Test Splitting (70% Train, 30% Test)**\n",
                    "5. **Baseline Random Forest Model Training**\n",
                    "6. **Hyperparameter Optimization via 5-Fold GridSearchCV**\n",
                    "7. **Comparative Model Evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix)**\n",
                    "8. **Feature Importance Analysis**\n",
                    "9. **Cross-Algorithm Benchmark (Replicating all 7 models from Wijaya et al., 2023)**\n",
                    "10. **Saving Artifacts for Deployment & VS Code UI**"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## **Step 1: Install & Import Dependencies**\n",
                    "We import the foundational libraries for data manipulation (`pandas`, `numpy`), data visualization (`matplotlib`, `seaborn`), machine learning and evaluation (`scikit-learn`), and model persistence (`joblib`)."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Step 1: Import required libraries\n",
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "import joblib\n",
                    "import urllib.request\n",
                    "import os\n",
                    "\n",
                    "# Scikit-Learn modules for preprocessing and model evaluation\n",
                    "from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold\n",
                    "from sklearn.preprocessing import StandardScaler\n",
                    "from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix\n",
                    "\n",
                    "# Machine learning classifiers\n",
                    "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier\n",
                    "from sklearn.tree import DecisionTreeClassifier\n",
                    "from sklearn.svm import SVC\n",
                    "from sklearn.neural_network import MLPClassifier\n",
                    "from sklearn.neighbors import KNeighborsClassifier\n",
                    "\n",
                    "print(\"✓ Libraries successfully imported!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## **Step 2: Load and Inspect the Dataset**\n",
                    "The dataset contains 1,059 observations and 8 columns (7 predictor features and 1 target variable `Grade`). We download it directly or load it locally, then strip any accidental whitespace in column names (e.g. `Fat `)."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Step 2: Download/Load the Milk Quality dataset\n",
                    "dataset_url = 'https://raw.githubusercontent.com/shaik1201/Milk-Quality-Prediction/master/milknew.csv'\n",
                    "csv_filename = 'milknew.csv'\n",
                    "\n",
                    "if not os.path.exists(csv_filename):\n",
                    "    print(f\"Downloading dataset from {dataset_url}...\")\n",
                    "    urllib.request.urlretrieve(dataset_url, csv_filename)\n",
                    "    print(\"Download complete!\")\n",
                    "\n",
                    "df = pd.read_csv(csv_filename)\n",
                    "df.columns = df.columns.str.strip()  # Clean whitespace from column names\n",
                    "\n",
                    "print(\"Dataset Shape:\", df.shape)\n",
                    "display(df.head())\n",
                    "\n",
                    "print(\"\\n--- Dataset Information ---\")\n",
                    "df.info()\n",
                    "\n",
                    "print(\"\\n--- Target Distribution (Grade) ---\")\n",
                    "print(df['Grade'].value_counts())"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## **Step 3: Exploratory Data Analysis (EDA)**\n",
                    "We visualize the distribution of classes and statistical relationships among the attributes (pH, Temperature, Taste, Odor, Fat, Turbidity, Colour)."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Step 3: EDA Visualizations\n",
                    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
                    "\n",
                    "# Class distribution bar plot\n",
                    "sns.countplot(data=df, x='Grade', order=['high', 'medium', 'low'], palette=['#2ca02c', '#ff7f0e', '#d62728'], ax=axes[0])\n",
                    "axes[0].set_title('Target Class Distribution (Grade)', fontsize=13, fontweight='bold')\n",
                    "axes[0].set_xlabel('Quality Grade')\n",
                    "axes[0].set_ylabel('Number of Samples')\n",
                    "for p in axes[0].patches:\n",
                    "    axes[0].annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),\n",
                    "                     ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontweight='bold')\n",
                    "\n",
                    "# pH distribution across quality grades\n",
                    "sns.boxplot(data=df, x='Grade', y='pH', order=['high', 'medium', 'low'], palette=['#2ca02c', '#ff7f0e', '#d62728'], ax=axes[1])\n",
                    "axes[1].set_title('pH Distribution across Quality Grades', fontsize=13, fontweight='bold')\n",
                    "axes[1].set_xlabel('Quality Grade')\n",
                    "axes[1].set_ylabel('pH Value')\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## **Step 4: Data Preprocessing, Scaling & Train-Test Split**\n",
                    "In adherence to the baseline paper (Wijaya et al., 2023):\n",
                    "1. Predictor features ($X$) and target label ($y$) are separated.\n",
                    "2. The dataset is split into **70% Training** (741 samples) and **30% Testing** (318 samples) using stratified sampling.\n",
                    "3. Numerical features are normalized via **StandardScaler**: $Z = \\frac{X - \\mu}{\\sigma}$."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Step 4: Separation, 70/30 Split and Standard Scaling\n",
                    "X = df.drop('Grade', axis=1)\n",
                    "y = df['Grade']\n",
                    "classes = ['high', 'medium', 'low']\n",
                    "\n",
                    "X_train, X_test, y_train, y_test = train_test_split(\n",
                    "    X, y, test_size=0.30, random_state=42, stratify=y\n",
                    ")\n",
                    "\n",
                    "print(f\"Training samples: {X_train.shape[0]} ({X_train.shape[0]/len(df)*100:.1f}%)\")\n",
                    "print(f\"Testing samples:  {X_test.shape[0]} ({X_test.shape[0]/len(df)*100:.1f}%)\")\n",
                    "\n",
                    "scaler = StandardScaler()\n",
                    "X_train_scaled = scaler.fit_transform(X_train)\n",
                    "X_test_scaled = scaler.transform(X_test)\n",
                    "\n",
                    "# Save scaler for VS Code deployment\n",
                    "joblib.dump(scaler, 'scaler.joblib')\n",
                    "print(\"✓ Features scaled using StandardScaler and scaler saved to 'scaler.joblib'.\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## **Step 5: Train Baseline Random Forest Model**\n",
                    "We train an unoptimized, default Random Forest classifier to serve as our empirical benchmark prior to hyperparameter optimization."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Step 5: Baseline Random Forest Classifier\n",
                    "rf_baseline = RandomForestClassifier(random_state=42)\n",
                    "rf_baseline.fit(X_train_scaled, y_train)\n",
                    "\n",
                    "y_pred_base = rf_baseline.predict(X_test_scaled)\n",
                    "\n",
                    "acc_base = accuracy_score(y_test, y_pred_base)\n",
                    "prec_base = precision_score(y_test, y_pred_base, average='weighted')\n",
                    "rec_base = recall_score(y_test, y_pred_base, average='weighted')\n",
                    "f1_base = f1_score(y_test, y_pred_base, average='weighted')\n",
                    "\n",
                    "print(\"=== BASELINE RANDOM FOREST PERFORMANCE ===\")\n",
                    "print(f\"Accuracy:  {acc_base * 100:.2f}%\")\n",
                    "print(f\"Precision: {prec_base:.4f}\")\n",
                    "print(f\"Recall:    {rec_base:.4f}\")\n",
                    "print(f\"F1-Score:  {f1_base:.4f}\")\n",
                    "print(\"\\nDetailed Classification Report:\\n\", classification_report(y_test, y_pred_base, digits=4))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## **Step 6: Hyperparameter Optimization via 5-Fold GridSearchCV**\n",
                    "Following Wijaya et al. (2023), Table 2, we optimize the core architectural hyperparameters of the Random Forest:\n",
                    "- `criterion`: `['gini', 'entropy']`\n",
                    "- `n_estimators`: `[50, 100, 150, 200]`\n",
                    "- `max_depth`: `[5, 10, 15, None]`\n",
                    "- `min_samples_split`: `[2, 5, 10]`\n",
                    "- `min_samples_leaf`: `[1, 2, 5]`\n",
                    "\n",
                    "This creates 288 parameter configurations evaluated with 5-fold stratified cross-validation (1,440 total fits) solely on the training set."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Step 6: Grid Search Hyperparameter Optimization\n",
                    "param_grid_rf = {\n",
                    "    'n_estimators': [50, 100, 150, 200],\n",
                    "    'criterion': ['gini', 'entropy'],\n",
                    "    'max_depth': [5, 10, 15, None],\n",
                    "    'min_samples_split': [2, 5, 10],\n",
                    "    'min_samples_leaf': [1, 2, 5]\n",
                    "}\n",
                    "\n",
                    "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n",
                    "\n",
                    "grid_rf = GridSearchCV(\n",
                    "    estimator=RandomForestClassifier(random_state=42),\n",
                    "    param_grid=param_grid_rf,\n",
                    "    cv=cv,\n",
                    "    scoring='accuracy',\n",
                    "    n_jobs=-1,\n",
                    "    verbose=1\n",
                    ")\n",
                    "\n",
                    "print(\"Starting Grid Search...\")\n",
                    "grid_rf.fit(X_train_scaled, y_train)\n",
                    "\n",
                    "best_rf = grid_rf.best_estimator_\n",
                    "joblib.dump(best_rf, 'best_random_forest_model.joblib')\n",
                    "\n",
                    "print(\"\\n=== GRID SEARCH OUTCOMES ===\")\n",
                    "print(\"Best Hyperparameters:\", grid_rf.best_params_)\n",
                    "print(f\"Best 5-Fold Cross-Validation Accuracy: {grid_rf.best_score_ * 100:.2f}%\")\n",
                    "print(\"✓ Optimized model saved to 'best_random_forest_model.joblib'.\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## **Step 7: Evaluate the Optimized Model on Unseen Test Data**\n",
                    "We test the tuned model on the held-out 30% test set to measure true generalization performance."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Step 7: Evaluate Optimized Random Forest on 30% Test Set\n",
                    "y_pred_opt = best_rf.predict(X_test_scaled)\n",
                    "\n",
                    "acc_opt = accuracy_score(y_test, y_pred_opt)\n",
                    "prec_opt = precision_score(y_test, y_pred_opt, average='weighted')\n",
                    "rec_opt = recall_score(y_test, y_pred_opt, average='weighted')\n",
                    "f1_opt = f1_score(y_test, y_pred_opt, average='weighted')\n",
                    "\n",
                    "print(\"=== OPTIMIZED RANDOM FOREST PERFORMANCE ===\")\n",
                    "print(f\"Test Accuracy:  {acc_opt * 100:.2f}%\")\n",
                    "print(f\"Test Precision: {prec_opt:.4f}\")\n",
                    "print(f\"Test Recall:    {rec_opt:.4f}\")\n",
                    "print(f\"Test F1-Score:  {f1_opt:.4f}\")\n",
                    "print(\"\\nDetailed Classification Report:\\n\", classification_report(y_test, y_pred_opt, digits=4))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## **Step 8: Confusion Matrix Comparison (Baseline vs Optimized)**\n",
                    "We generate side-by-side confusion matrices to examine true positives, true negatives, and false misclassifications across the three grades."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Step 8: Plot Confusion Matrices\n",
                    "cm_base = confusion_matrix(y_test, y_pred_base, labels=classes)\n",
                    "cm_opt = confusion_matrix(y_test, y_pred_opt, labels=classes)\n",
                    "\n",
                    "fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))\n",
                    "sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=axes[0])\n",
                    "axes[0].set_title(f'Baseline Random Forest\\n(Test Accuracy: {acc_base*100:.2f}%)', fontsize=12, fontweight='bold')\n",
                    "axes[0].set_xlabel('Predicted Quality Grade')\n",
                    "axes[0].set_ylabel('Actual Quality Grade')\n",
                    "\n",
                    "sns.heatmap(cm_opt, annot=True, fmt='d', cmap='Greens', xticklabels=classes, yticklabels=classes, ax=axes[1])\n",
                    "axes[1].set_title(f'Optimized Random Forest (HPO)\\n(Test Accuracy: {acc_opt*100:.2f}%)', fontsize=12, fontweight='bold')\n",
                    "axes[1].set_xlabel('Predicted Quality Grade')\n",
                    "axes[1].set_ylabel('Actual Quality Grade')\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.savefig('confusion_matrices_comparison.png', dpi=300)\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## **Step 9: Feature Importance Analysis**\n",
                    "Gini importance (Mean Decrease in Impurity) reveals which physicochemical attributes most heavily influence milk grade determination."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Step 9: Compute & Plot Feature Importances\n",
                    "feat_importances = pd.Series(best_rf.feature_importances_, index=X.columns).sort_values(ascending=True)\n",
                    "\n",
                    "plt.figure(figsize=(9, 5))\n",
                    "feat_importances.plot(kind='barh', color='#2b5c8f', edgecolor='black', alpha=0.9)\n",
                    "plt.title('Relative Feature Importances (Optimized Random Forest)', fontsize=13, fontweight='bold')\n",
                    "plt.xlabel('Importance Score (Mean Decrease in Impurity)')\n",
                    "plt.ylabel('Milk Attributes')\n",
                    "plt.grid(axis='x', linestyle='--', alpha=0.6)\n",
                    "\n",
                    "for i, v in enumerate(feat_importances):\n",
                    "    plt.text(v + 0.005, i, f\"{v*100:.1f}%\", va='center', fontweight='bold', fontsize=9.5)\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.savefig('feature_importance.png', dpi=300)\n",
                    "plt.show()\n",
                    "\n",
                    "print(\"Feature Ranking:\")\n",
                    "for rank, (feature, val) in enumerate(feat_importances.sort_values(ascending=False).items(), 1):\n",
                    "    print(f\"{rank}. {feature:12s}: {val*100:.2f}%\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## **Step 10: Multi-Algorithm Benchmark Comparison**\n",
                    "In the baseline paper (Wijaya et al., 2023, Table 5), seven distinct classification algorithms were compared. We replicate all seven algorithms on this dataset to provide thorough, substantive comparative context for the Results and Discussion section."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Step 10: Benchmark All 7 Algorithms from Wijaya et al. (2023)\n",
                    "algorithms = {\n",
                    "    'Gradient Tree Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.3, random_state=42),\n",
                    "    'Decision Tree': DecisionTreeClassifier(criterion='gini', max_depth=15, min_samples_split=10, min_samples_leaf=5, random_state=42),\n",
                    "    'Support Vector Machine': SVC(C=10, gamma=0.01, kernel='rbf', random_state=42),\n",
                    "    'Neural Network (MLP)': MLPClassifier(hidden_layer_sizes=(30, 15, 10), max_iter=300, activation='relu', solver='adam', random_state=42),\n",
                    "    'AdaBoost': AdaBoostClassifier(n_estimators=200, learning_rate=0.3, random_state=42),\n",
                    "    'Baseline Random Forest': rf_baseline,\n",
                    "    'Optimized Random Forest': best_rf,\n",
                    "    'k-Nearest Neighbors (k-NN)': KNeighborsClassifier(n_neighbors=2, weights='distance', metric='manhattan')\n",
                    "}\n",
                    "\n",
                    "benchmark_records = []\n",
                    "for name, clf in algorithms.items():\n",
                    "    if name not in ['Baseline Random Forest', 'Optimized Random Forest']:\n",
                    "        clf.fit(X_train_scaled, y_train)\n",
                    "    preds = clf.predict(X_test_scaled)\n",
                    "    acc = accuracy_score(y_test, preds)\n",
                    "    prec = precision_score(y_test, preds, average='weighted', zero_division=0)\n",
                    "    rec = recall_score(y_test, preds, average='weighted', zero_division=0)\n",
                    "    f1 = f1_score(y_test, preds, average='weighted', zero_division=0)\n",
                    "    benchmark_records.append({\n",
                    "        'Algorithm': name,\n",
                    "        'Accuracy': acc,\n",
                    "        'Precision': prec,\n",
                    "        'Recall': rec,\n",
                    "        'F1-Score': f1\n",
                    "    })\n",
                    "\n",
                    "benchmark_df = pd.DataFrame(benchmark_records)\n",
                    "print(\"=== COMPREHENSIVE BENCHMARK REPORT ===\")\n",
                    "display(benchmark_df.style.format({'Accuracy': '{:.4f}', 'Precision': '{:.4f}', 'Recall': '{:.4f}', 'F1-Score': '{:.4f}'}))\n",
                    "\n",
                    "# Plot Comparison Chart\n",
                    "plt.figure(figsize=(10, 5))\n",
                    "bars = plt.bar(benchmark_df['Algorithm'], benchmark_df['Accuracy'], color='#1f77b4', edgecolor='black', alpha=0.85)\n",
                    "plt.xticks(rotation=35, ha='right')\n",
                    "plt.ylim(0.70, 1.02)\n",
                    "plt.ylabel('Test Accuracy')\n",
                    "plt.title('Algorithm Performance Comparison on Milk Quality Dataset', fontsize=13, fontweight='bold')\n",
                    "for bar in bars:\n",
                    "    yval = bar.get_height()\n",
                    "    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f\"{yval*100:.1f}%\", ha='center', va='bottom', fontweight='bold', fontsize=8.5)\n",
                    "plt.tight_layout()\n",
                    "plt.savefig('algorithm_comparison.png', dpi=300)\n",
                    "plt.show()"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open(r'c:\Users\Charmele\Downloads\Projects\Milk_Quality_ML_Replication.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
    print("Notebook 'Milk_Quality_ML_Replication.ipynb' created successfully!")

if __name__ == '__main__':
    create_colab_notebook()
