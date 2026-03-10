# ⚡ Smart Grid Stability Prediction via Neural Network & Random Forest

A data analytics and machine learning project that predicts whether a 
smart electrical grid system is **stable or unstable** using two models:
a Neural Network classifier and a Random Forest Regressor.

Built as a group assignment for Data Analytics & Machine Learning 
(ITS69304) at Taylor's University.

---

## 📊 Results

| Model | Metric | Score |
|---|---|---|
| Neural Network | Test Accuracy | 93.45% |
| Neural Network | Test Precision | 95.98% |
| Neural Network | AUC-ROC | 0.9849 |
| Random Forest | R² Score | 0.89 |
| Random Forest | Test RMSE | 0.0124 |

---

## 🔧 Preprocessing Pipeline

- Outlier detection & removal (IQR method)
- Label encoding for categorical target
- 60/20/20 train/validation/test split
- SMOTE oversampling to fix class imbalance
- MaxAbsScaler feature scaling (preserves sign of values)
- ANOVA F-score feature selection (top 8 features)

---

## 🏗️ Models

**Neural Network** — Sequential model with ReLU hidden layers and 
sigmoid output for binary classification

**Random Forest Regressor** — Ensemble of 200 decision trees 
(max_depth=15) predicting continuous stability values

---

## 🔧 Tech Stack

- Python, TensorFlow, Keras
- scikit-learn (SMOTE, SelectKBest, RandomForestRegressor)
- NumPy, Pandas, Matplotlib, Seaborn
- Google Colab

---

## 📁 Dataset

`grid_data.csv` — 10,000 observations, 14 features representing 
a simulated smart electrical grid with supplier/consumer nodes.
