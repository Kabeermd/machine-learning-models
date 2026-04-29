# Machine Learning Models — Classification & Regression

Two scikit-learn projects: accident severity classification on the FARS dataset and bacteria growth regression, both built with full preprocessing pipelines.

## Project 1 — Accident Severity Classification (FARS Dataset)

Multi-class classification predicting accident severity using the Fatality Analysis Reporting System (FARS) dataset from the US National Highway Traffic Safety Administration.

### Models Compared

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbours (KNN)

### Methodology

- scikit-learn `Pipeline` objects combining preprocessing and model
- `StandardScaler` for feature normalisation
- `GridSearchCV` for hyperparameter tuning
- Evaluation: Accuracy, F1-score (macro), confusion matrix

---

## Project 2 — Bacteria Growth Regression

Regression modelling to predict bacteria colony growth rate under varying experimental conditions (temperature, pH, etc.).

### Models Compared

- Linear Regression (baseline)
- Ridge / Lasso Regression
- Random Forest Regressor

### Methodology

- Feature engineering and outlier handling
- Cross-validation with RMSE and R² evaluation

---

## Files

| File | Description |
|------|-------------|
| `ml_classification_regression_models.ipynb` | Full pipeline for both tasks |

## Tech Stack

- Python 3
- scikit-learn
- pandas, NumPy
- Matplotlib, seaborn
