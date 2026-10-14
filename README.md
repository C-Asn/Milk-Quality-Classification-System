# Milk Quality Classification System
**A Random Forest Machine Learning Model with Hyperparameter Optimization**

**Authors:**
- **Charmele N. Asna** (`c.asna.558339@umindanao.edu.ph`)
- **Francis Dave O. Brina** (`f.brina.556973@umindanao.edu.ph`)
- **Rolemir Kirk Don Diango Zayas** (`r.zayas.551640@umindanao.edu.ph`)  
*College of Computing Education, University of Mindanao, Davao City, Philippines*

---

## 📌 Project Overview
This project replicates and adapts the machine learning and hyperparameter optimization methodology from:
> **Wijaya et al. (2023)**, *Seafood Quality Detection Using Electronic Nose and Machine Learning Algorithms With Hyperparameter Optimization*, **IEEE Access**, 11, 62484–62495.

We adapt this methodology to classify bovine milk freshness into three distinct grades:
* **HIGH (Premium Quality):** Safe for direct drinking, retail packaging, and commercial distribution.
* **MEDIUM (Moderate Quality):** Acceptable for pasteurization or secondary byproduct processing (cheese, butter).
* **LOW (Spoiled / Unsafe):** Rejected batch due to bacterial acidification (lactic acid fermentation) or severe heat abuse.

---

## 📂 Project Structure

```text
Projects/
│
├── app.py                               # Fully Functional Interactive Web UI (Streamlit)
├── Milk_Quality_ML_Replication.ipynb    # Google Colab / Jupyter Replication Notebook
├── colab_script.py                      # Standalone Python replication script
├── run_ml_experiment.py                 # Core experimental pipeline & benchmark
├── verify_test_cases.py                 # Script verifying 10 diverse test cases
│
├── milknew.csv                          # Kaggle Milk Quality Dataset (1,059 rows)
├── best_random_forest_model.joblib      # Production Random Forest Model (Tuned)
├── scaler.joblib                        # Fitted StandardScaler object
├── experiment_summary.json              # Full experimental evaluation metrics
│
├── requirements.txt                     # Dependencies list
├── README.md                            # Complete documentation & usage manual
│
└── static/
    └── images/                          # High-resolution figures for documentation
        ├── confusion_matrices_comparison.png
        ├── feature_importance.png
        ├── baseline_vs_tuned_comparison.png
        └── class_distribution.png
```

---

## ⚙️ Installation & Setup Instructions

### 1. Prerequisites
* Python 3.10, 3.11, 3.12, 3.13, or 3.14 installed.
* Visual Studio Code (or any modern terminal).

### 2. Clone / Open Project
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd Projects
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Web Application
```bash
python -m streamlit run app.py
```
The browser will automatically open at `http://localhost:8501`.

---

## 🧪 10 Verified Test Cases for Live Demonstration

| Case | Scenario / Description | pH | Temp (°C) | Taste | Odor | Fat | Turbidity | Colour | Expected Grade | Model Output |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | Fresh Milk (Optimal Standards) | 6.6 | 38 | 1 | 1 | 1 | 1 | 255 | **HIGH** | **HIGH (100%)** |
| **2** | Fresh Farm Milk (Clean Scent) | 6.6 | 35 | 1 | 0 | 1 | 0 | 254 | **HIGH** | **HIGH (95%)** |
| **3** | Chilled Farm Milk | 6.8 | 36 | 0 | 1 | 1 | 0 | 253 | **HIGH** | **HIGH (88%)** |
| **4** | Low-Fat Watery Consistency | 6.6 | 37 | 0 | 0 | 0 | 0 | 255 | **MEDIUM** | **MEDIUM (98%)** |
| **5** | Warm Mild-Acidity Intake | 6.7 | 45 | 1 | 1 | 1 | 0 | 245 | **MEDIUM** | **MEDIUM (66%)** |
| **6** | Low-Colour Processing Milk | 6.8 | 34 | 0 | 0 | 0 | 1 | 240 | **MEDIUM** | **MEDIUM (99%)** |
| **7** | Unscented Standard Intake Milk | 6.5 | 36 | 0 | 0 | 1 | 0 | 255 | **MEDIUM** | **MEDIUM (89%)** |
| **8** | Severe Acidic Spoilage (pH 3.0) | 3.0 | 40 | 1 | 1 | 1 | 1 | 255 | **LOW** | **LOW (78%)** |
| **9** | Thermal Breakdown & Curdling | 4.5 | 60 | 0 | 1 | 1 | 1 | 250 | **LOW** | **LOW (95%)** |
| **10**| High Temperature Thermal Abuse | 6.8 | 70 | 1 | 1 | 1 | 1 | 255 | **LOW** | **LOW (73%)** |

*Note: In `app.py`, you can test these immediately using the **Quick Preset Selector** dropdown or adjust sliders manually.*

---

## 📊 Summary of Experimental Results

* **Data Leakage Control:** 976 duplicate rows were removed, yielding 83 unique observations (58 train / 25 test).
* **Baseline RF (Wijaya et al. parameters):** 72.00% Test Accuracy | 76.00% Precision | 72.00% Recall | 72.28% F1-Score.
* **Tuned RF (5-Fold GridSearchCV):** **80.00% Test Accuracy (+8.00 pp)** | **84.80% Precision** | **80.00% Recall** | **80.38% F1-Score**.
* **Primary Drivers:** Temperature (**29.19%**) and pH (**25.97%**) account for over **55%** of the model's predictive importance.

---

## 📜 Citation
If referencing this study or dataset:
1. **Wijaya, D. R. et al. (2023).** Seafood quality detection using electronic nose and machine learning algorithms with hyperparameter optimization. *IEEE Access*, 11, 62484–62495.
2. **cpluzshrijayan (2022).** Milk Quality Prediction [Data set]. *Kaggle*. https://www.kaggle.com/datasets/cpluzshrijayan/milkquality
