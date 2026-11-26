# Machine Learning Algorithms - Coursework Repository

This repository contains implementation code, data analysis, and model evaluations for laboratory assignments completed as part of the **Machine Learning Algorithms** course at **Kaunas University of Technology (KTU)**.

## 📌 Repository Overview

The objective of these labs is to implement foundational supervised machine learning algorithms from scratch using `NumPy`, compare custom implementations against standard machine learning/deep learning frameworks (`scikit-learn` and `PyTorch`), and conduct end-to-end exploratory data analysis (EDA) on the **Melbourne Housing Market** dataset (target variable: `Price`).

---

## 📂 Laboratory Work Summaries

### 🧪 Laboratory Work 1: Linear Regression
Focused on exploratory data analysis, preprocessing, and building multi-variable linear regression models from mathematical foundations.

* **Data Quality Analysis & EDA:**
  * Statistical summary for continuous features (`Rooms`, `Price`, `Propertycount`, `Distance`, `Postcode`) and categorical features (`Suburb`, `Address`, `Type`, `Method`, `SellerG`, `Date`, `Regionname`, `CouncilArea`).
  * Distribution analysis, outlier detection, feature correlation matrices, and relationship plots.
  * Preprocessing: handling missing target rows, feature scaling/normalization, encoding, and derived feature extraction using training split parameters.
* **Model Implementation & Evaluation:**
  * **From Scratch:** Multi-variable linear regression solved via **Ordinary Least Squares (OLS)** and **Gradient Descent**.
  * **Library Comparison:** Benchmarked against `scikit-learn` linear regression.
  * Hypothesis testing for coefficient significance, $K$-fold cross-validation ($k \ge 5$), and single-record raw inference.

---

### 🧪 Laboratory Work 2: Decision Trees & Random Forest
Focused on tree-based models for regression and classification, custom splitting criteria, and ensemble learning.

* **Data Preprocessing & Splitting:**
  * Partitioned into training, validation, and holdout test splits, removing missing target values, duplicates, and extreme `Price` outliers.
* **Regression & Classification Decision Trees:**
  * **Custom Implementation:** Handles categorical features directly via multi-way splits without prior numerical encoding, and evaluates multiple thresholds for continuous variables.
  * Hyperparameter tuning across tree depth and split thresholds; target discretization into 5 classes for classification.
  * Benchmarked against `scikit-learn`'s `DecisionTreeRegressor` and `DecisionTreeClassifier`.
* **Regression Random Forest:**
  * **Custom Ensemble:** Bagging ensemble of custom regression trees evaluated across 27+ hyperparameter combinations via $K$-fold cross-validation.
  * Benchmarked against `scikit-learn`'s `RandomForestRegressor` with raw-format inference on unseen test records.

---

### 🧪 Laboratory Work 3: Artificial Neural Networks
Focused on building a feed-forward Artificial Neural Network (ANN) with backpropagation from scratch in `NumPy` to predict Melbourne property prices, and comparing it against `PyTorch`.

* **1. Data Preparation (80:10:10 Split):**
  * Removed rows with missing `Price` values, 2 duplicate rows, and 1,832 `Price` outliers (bounds: $[-272,500; 2,107,500]$).
  * Dropped high-cardinality features (`Suburb`, `SellerG`, `Address`, `Postcode`) to prevent overfitting, retained `CouncilArea` and `Regionname`, and extracted numerical `Age` from auction `Date`.
* **2. Custom Feed-Forward ANN (`NumPy`):**
  * Implemented forward pass, MSE loss, backpropagation, gradient descent ($\eta = 0.005$, up to 3,000 epochs), and Xavier/He weight initialization.
  * **Architecture Search (9 Models):** Evaluated hidden neuron budgets of **10**, **20**, and **30** across 1-, 2-, and 3-layer topologies (`[10]`, `[5, 5]`, `[4, 4, 2]`, `[20]`, `[10, 10]`, `[8, 7, 5]`, `[30]`, `[15, 15]`, `[12, 10, 8]`). Single hidden-layer architectures (`[10]`, `[20]`, `[30]`) converged best.
  * **Activation Function Study:** Tested **Sigmoid**, **ReLU**, **Leaky ReLU**, and **Threshold** activations on the top architectures. **Leaky ReLU** achieved the lowest validation MSE across all neuron counts, with **M4 (`[20]`, Leaky ReLU)** achieving the best overall performance (`Val MSE: 1.0611`, `Test MSE: 1.0349`, `Test R²: 0.5560`).
* **3. Framework Comparison & Inference:**
  * Benchmarked custom `NumPy` networks (`[10]`, `[20]`, `[30]` with Leaky ReLU) against equivalent **`PyTorch`** models across `MSE`, `RMSE`, `MAE`, and $R^2$, confirming closely aligned performance.
  * Deployed the best model (`[20]` with Leaky ReLU) for inference on raw test records in original AUD scale.

---

## 🚀 Getting Started

### Prerequisites
Ensure you have Python 3.8+ installed along with the required libraries:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch
```

### Running the Code
1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
   cd YOUR_REPOSITORY_NAME
   ```
2. Open and run the respective Jupyter Notebooks or Python scripts for Lab 1, Lab 2, or Lab 3.

---

## 📊 Tech Stack
* **Language:** Python
* **Core Libraries:** `NumPy`, `Pandas`, `Matplotlib`, `Seaborn`
* **ML / DL Frameworks:** `scikit-learn`, `PyTorch`
