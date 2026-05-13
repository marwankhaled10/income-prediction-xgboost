# 💰 Income Prediction — Census Adult Dataset

A machine learning project that predicts whether an individual's annual income exceeds **$50K** based on census demographic data, using an optimized **XGBoost** classifier with full preprocessing pipeline and hyperparameter tuning.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Methodology](#️-methodology)
- [Results](#-results)
- [Technologies Used](#️-technologies-used)
- [How to Run](#-how-to-run)

---

## 🔍 Overview

This project tackles a classic **binary classification** problem from the UCI Machine Learning Repository. The goal is to predict whether a person earns more or less than $50K/year using features such as age, education, occupation, and work hours.

Key highlights:
- Exploratory Data Analysis (EDA) with class imbalance detection
- Robust preprocessing pipeline with `ColumnTransformer`
- Handled class imbalance using `scale_pos_weight`
- Hyperparameter tuning with `GridSearchCV` (5-fold cross-validation)
- Final model: **XGBoost Classifier**

---

## 📊 Dataset

| Property       | Detail                            |
|----------------|-----------------------------------|
| Source         | [UCI Adult Census Dataset]([[https://archive.ics.uci.edu/ml/datasets/adult](https://www.kaggle.com/datasets/uciml/adult-census-income)](https://www.kaggle.com/datasets/uciml/adult-census-income)) |
| Rows           | 32,561                            |
| Features       | 14                                |
| Target         | `income` (≤50K / >50K)           |
| Missing Values | Encoded as `?` → handled via `dropna` |

### Features

| Feature | Type |
|---|---|
| age | Numerical |
| workclass | Categorical |
| fnlwgt | Numerical |
| education | Categorical |
| education.num | Numerical |
| marital.status | Categorical |
| occupation | Categorical |
| relationship | Categorical |
| race | Categorical |
| sex | Categorical |
| capital.gain | Numerical |
| capital.loss | Numerical |
| hours.per.week | Numerical |
| native.country | Categorical |

---

## 📁 Project Structure

```
income-prediction/
│
├── adult.csv                  # Dataset
├── income-prediction.ipynb    # Main notebook
└── README.md
```

---

## ⚙️ Methodology

### 1. Exploratory Data Analysis
- Inspected shape, data types, null values
- Visualized income class distribution (bar chart)
- Identified class imbalance between `<=50K` and `>50K`

### 2. Data Preprocessing
- Replaced `?` with `NaN` and dropped missing rows
- Separated features (`X`) and target (`y`)
- Applied `LabelEncoder` to target column
- Used `ColumnTransformer` with `OneHotEncoder` for categorical columns
- Numerical columns passed through as-is

### 3. Model Training
- Train/test split: **80% / 20%** (stratified)
- Built a `Pipeline` combining preprocessor + XGBoost model
- Handled class imbalance with `scale_pos_weight`

### 4. Hyperparameter Tuning

```python
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [3, 5, 7],
    "model__learning_rate": [0.01, 0.1, 0.2]
}
```

Used `GridSearchCV` with **5-fold cross-validation** and `accuracy` scoring.

---

## 📈 Results

| Metric | Value |
|---|---|
| Best CV Accuracy | See notebook output |
| Test Accuracy | See notebook output |
| Evaluation | Classification report + Confusion matrix |

> Run the notebook to see full evaluation results including precision, recall, and F1-score per class.

---

## 🛠️ Technologies Used

| Tool | Purpose |
|---|---|
| Python 3 | Core language |
| Pandas & NumPy | Data manipulation |
| Scikit-learn | Preprocessing, pipelines, evaluation |
| XGBoost | Gradient boosted classifier |
| Matplotlib | Visualization |
| GridSearchCV | Hyperparameter optimization |
| Jupyter Notebook | Development environment |

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/marwankhaled10/income-prediction.git
cd income-prediction
```

### 2. Install dependencies
```bash
pip install pandas numpy scikit-learn xgboost matplotlib jupyter
```

### 3. Launch the notebook
```bash
jupyter notebook income-prediction.ipynb
```

---

## 👤 Author

**Marwan Khaled**
- GitHub: [@marwankhaled10](https://github.com/marwankhaled10)
- LinkedIn: [marwan-khaled](https://www.linkedin.com/in/marwan-khaled-562b10370)
- Email: marwankhaled213@gmail.com
