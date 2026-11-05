# Machine Learning Algorithms — Coursework Repository

This repository contains implementation code, analysis, and evaluations for two laboratory assignments completed as part of the **Machine Learning Algorithms (P176B104)** course.

---

## 📌 Repository Overview

The objective of these labs is to gain hands-on experience building foundational supervised machine learning models from scratch, comparing custom implementations with standard `scikit-learn` models, and performing complete exploratory data analysis (EDA) and evaluation workflows.

---

## 📂 Laboratory Work Summaries

### 🧪 Laboratory Work 1: Linear Regression

Focused on data pre-processing, exploratory data analysis, and building multi-variable linear regression models using mathematical foundations.

* **Data Quality Analysis & EDA:**
  * Statistical summary for continuous features (missing values, min/max, quartiles, mean, std) and categorical features (cardinality, modes, percentages).
  * Visualization of distributions (histograms), outlier detection, feature correlation matrix, and relationship plots (SPLOM/scatter plots).
  * Data preprocessing: handling missing values, scaling/normalization, categorical encoding, and feature engineering based on training split parameters.
* **Model Implementation:**
  * **From Scratch:** Multi-variable linear regression solved via **Ordinary Least Squares (OLS)** and **Gradient Descent**.
  * **Library Model:** Comparison with `scikit-learn` linear regression implementations.
* **Evaluation & Hypotheses Testing:**
  * Hypothesis testing for coefficient significance.
  * $K$-fold cross-validation ($k \ge 5$) comparing full feature set vs. significant feature subsets.
  * Performance evaluation using standard regression metrics ($MSE$, $MAE$, $R^2$).
  * Interface for single-record inference in raw data format.

---

### 🧪 Laboratory Work 2: Decision Trees & Random Forest

Focused on tree-based architectures for both regression and classification tasks, custom tree splitting logic, and ensemble methods.

* **1. Data Preprocessing & Splitting:**
  * Partitioning dataset into training, validation, and holdout test sets ($\ge 10\%$).
* **2. Regression Decision Tree:**
  * **Custom Implementation:** Handles non-binary categorical features directly (n-ary split per category) without prior numerical encoding. Multi-threshold testing for continuous features.
  * Parameter tuning across depth and min-split thresholds evaluated against validation metrics ($MSE$, $MAE$, $MAPE$).
  * Comparison against `scikit-learn`'s `DecisionTreeRegressor`.
  * Visual representation of decision criteria up to depth 3.
* **3. Classification Decision Tree:**
  * Target discretization into 5 balanced classes.
  * **Custom Implementation:** Direct categorical handling and multi-threshold evaluation.
  * Evaluation via accuracy, precision, recall, F1-score, and confusion matrices.
  * Comparison against `scikit-learn`'s `DecisionTreeClassifier`.
* **4. Regression Random Forest (Ensemble):**
  * **Custom Implementation:** Ensemble of custom regression trees trained across combinations of estimator counts ($N \ge 27$ total grid search trials).
  * $K$-fold cross-validation analysis with performance visualization (box plots across tree counts and leaf depth).
  * Comparison against `scikit-learn`'s `RandomForestRegressor`.
  * Holdout test predictions and inference interface for unseen samples in raw format.

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed along with the core data science stack:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Running the Code

1. Clone the repository:
   ```bash
   git clone https://github.com/Dennnnnnnnnn/Machine-Learning-1
   cd Machine-Learning-1
   ```
2. Navigate to the desired laboratory notebook or script directory to execute the code.

---

## 📊 Summary of Tech Stack

* **Language:** Python
* **Libraries:** `NumPy`, `Pandas`, `Matplotlib`, `Seaborn`, `scikit-learn`