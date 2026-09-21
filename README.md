# Credit Card Fraud Detection: Machine Learning on Imbalanced Data

An end-to-end, production-grade machine learning system for detecting credit card fraud on highly imbalanced transaction data (0.172% fraud prevalence).

For detailed mathematical formulations, bug diagnoses from prior work, and architectural comparisons, see [REPORT_OF_IMPROVEMENTS.md](REPORT_OF_IMPROVEMENTS.md).

---

## 📊 Benchmark Results Summary

All models were evaluated on a held-out stratified test set of **56,962 transactions** containing **98 actual fraud cases**.

| Model | PR-AUC (AP) | ROC-AUC | Optimal Threshold | Optimal F1-Score | Optimal Recall | Optimal Precision | Estimated Cost ($) | Train Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SMOTE + Random Forest** | **0.8840** | **0.9840** | 0.5700 | **0.8770** | 83.67% | 92.13% | $1,955 | 23.63s |
| **XGBoost (Cost-Weighted)** | **0.8706** | 0.9750 | 0.5148 | 0.8691 | **84.69%** | 89.25% | **$1,850** | **0.79s** |
| **SMOTE + LightGBM** | **0.8688** | 0.9809 | 0.8353 | **0.8852** | 82.65% | **95.29%** | $2,060 | 1.97s |
| **Random Forest (Balanced)**| 0.8565 | 0.9508 | 0.4695 | 0.8541 | 80.61% | 90.80% | $2,320 | 14.80s |
| **CatBoost (Balanced)** | 0.8189 | 0.9775 | 0.9744 | 0.8283 | 83.67% | 82.00% | $2,010 | 3.85s |
| **Logistic Regression (Balanced)**| 0.7128 | 0.9738 | 1.0000 | 0.8247 | 81.63% | 83.33% | $2,240 | 1.09s |

*Cost calculation: \$120 per False Negative (missed fraud) + \$5 per False Positive (investigation overhead).*

---

## 🛠️ Key Architectural Improvements

1. **Leak-Free Resampling**: SMOTE and sampling strategies are strictly encapsulated within cross-validation and training pipelines (`imblearn.pipeline.Pipeline`). The test split remains pristine and uncorrupted.
2. **Outlier-Robust Feature Scaling**: `RobustScaler` on `Amount` dampens extreme transaction values up to \$25,691, while cyclical $\sin/\cos$ transformations extract time-of-day dynamics from `Time`.
3. **Decision Threshold Optimization**: Replaces the default 0.50 threshold by finding the optimal decision threshold that maximizes $F_1$-score on the Precision-Recall curve.
4. **State-of-the-Art Tree Boosters**: Incorporates XGBoost, LightGBM, and CatBoost alongside Balanced Random Forests.

---

## 📁 Repository Structure

```
Credit-Card-Fraud-Detection/
├── dataset/
│   ├── creditcard_part1.csv           # Part 1 (142,403 rows, 72MB)
│   └── creditcard_part2.csv           # Part 2 (142,404 rows, 72MB)
├── src/
│   ├── __init__.py                    # Compatibility patches & warnings filter
│   ├── config.py                      # Global parameters, paths, seeds, costs
│   ├── data.py                        # Ingestion, validation, stratified splitting
│   ├── features.py                    # RobustScaler, log transforms, cyclical Time
│   ├── resampling.py                  # Leak-free imblearn pipelines
│   ├── models.py                      # Model portfolio & hyperparameters
│   └── evaluation.py                  # PR-AUC, ROC-AUC, F1 threshold tuning, plotting
├── scripts/
│   ├── __init__.py
│   └── train_and_evaluate.py          # End-to-end benchmark execution script
├── outputs/
│   ├── figures/                       # PR curves, ROC curves, confusion matrices
│   └── tables/                        # benchmark_results.csv & benchmark_results.md
├── notebooks/
│   ├── fraud_detection_benchmark.ipynb # Clean, interactive modern benchmark notebook
│   └── credit-card-fraud-detection-kaggle.ipynb # Original historical notebook
├── REPORT_OF_IMPROVEMENTS.md          # Comprehensive diagnostic report & analysis
└── README.md                          # Project documentation
```

---

## 🚀 Quickstart

### 1. Requirements
Python 3.10+ with standard scientific libraries:
```bash
pip install numpy pandas scikit-learn imbalanced-learn xgboost lightgbm catboost matplotlib seaborn
```

### 2. Run Full Model Training & Evaluation
```bash
python3 scripts/train_and_evaluate.py
```
This script will:
- Automatically load and merge `creditcard_part1.csv` and `creditcard_part2.csv` (split to stay under GitHub's 100MB limit).
- Create a stratified 80/20 train/test split.
- Train all 6 candidate models with leak-free preprocessing.
- Evaluate each model on PR-AUC, ROC-AUC, and optimal threshold $F_1$.
- Save all comparison tables to `outputs/tables/` and figures to `outputs/figures/`.

---

## 📈 Visualizations

- **Precision-Recall Curves**: See [outputs/figures/precision_recall_curves.png](outputs/figures/precision_recall_curves.png)
- **ROC Curves**: See [outputs/figures/roc_curves.png](outputs/figures/roc_curves.png)
- **Confusion Matrices (Optimal Threshold)**: See [outputs/figures/confusion_matrices_optimal.png](outputs/figures/confusion_matrices_optimal.png)
- **Feature Importances**: See `outputs/figures/feature_importance_*.png`
