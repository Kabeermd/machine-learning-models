# Machine Learning Models

Two end-to-end machine learning mini-projects covering classification and regression using scikit-learn pipelines with hyperparameter tuning — applied to real-world tabular datasets.

---

## Mini-Project 1 — Accident Severity Classification (FARS Dataset)

**Dataset:** FARS (Fatality Analysis Reporting System) — 100,000+ US road traffic accident records, 20 features, multi-class severity label.

### Challenge
Strongly imbalanced multi-class classification problem. Preprocessing required careful handling of categorical variables, missing values, and class imbalance.

### Pipelines Implemented

| Pipeline | Algorithm | Preprocessing | Notes |
|----------|-----------|--------------|-------|
| 1a | Logistic Regression (Standard) | StandardScaler + OneHot | Baseline linear model |
| 1b | Logistic Regression (L1 Feature Selection) | RobustScaler + SelectFromModel | Sparse feature subset |
| 2a | Random Forest | Tree-based (no scaling) | Non-linear ensemble |
| 2b | Random Forest + Tuning | RandomizedSearchCV | Optimised n_estimators, max_depth |
| 3 | k-Nearest Neighbours | MinMaxScaler + OneHot | Distance-based, k tuned via GridSearch |
| 4a | Linear SVM | StandardScaler | Margin-based, class_weight="balanced" |
| 4b | SVM + L1 Feature Selection | StandardScaler + SelectFrommodel | Sparse subset |

### Evaluation
- Primary metric: **macro-F1** (balanced across all severity classes)
- Secondary: accuracy, weighted-F1
- All models evaluated on a stratified 80/20 train/test split

---

## Mini-Project 2 — Bacteria Growth Rate Regression

**Dataset:** Experimental data on bacterial growth across strains and conditions (CO2 availability, temperature, etc.). Targets: `a` (max bacteria), `mu` (growth rate), `tau` (lag time), `a0` (initial count).

### Pipelines Implemented (per target variable)

| Algorithm | Notes |
|-----------|-------|
| Ridge Regression | Regularised linear baseline |
| Random Forest | Best performer for target `a` (Val RMSE: 2.19) |
| Support Vector Regressor (SVR) | RBF kernel, tuned C & epsilon |
| k-Nearest Neighbours | Tuned n_neighbors |

### Workflow
- EDA with ydata-profiling, histograms, boxplots, correlation heatmaps
- Train / Validation / Test split
- GridSearchCV / RandomizedSearchCV for hyperparameter tuning
- Evaluation: RMSE, MAE, R² on validation and test sets

---

## Tech Stack
- Python, scikit-learn, Pandas, NumPy
- Matplotlib, Seaborn, ydata-profiling
- Google Colab

## Files
| File | Description |
|------|-------------|
| `CSC8635_2025_Completed (1).ipynb` | Both classification and regression mini-projects |
