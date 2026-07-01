# Iris Classification: Decision Tree vs. Random Forest vs. SVM

A comparative study of three classic machine learning classifiers — **Decision Tree**, **Random Forest**, and **Support Vector Machine (SVM)** — applied to the classic Iris flower dataset. Each model is tuned via randomized hyperparameter search and evaluated on held-out training and test accuracy to compare generalization performance.

---

## Table of Contents

- [Problem Overview](#problem-overview)
- [Dataset](#dataset)
- [Approach](#approach)
  - [1. Exploratory Data Analysis](#1-exploratory-data-analysis)
  - [2. Train/Test Split](#2-traintest-split)
  - [3. Decision Tree Classifier](#3-decision-tree-classifier)
  - [4. Random Forest Classifier](#4-random-forest-classifier)
  - [5. Support Vector Machine (SVM)](#5-support-vector-machine-svm)
  - [6. Model Comparison](#6-model-comparison)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [How to Run](#how-to-run)
- [Key Takeaways](#key-takeaways)

---

## Problem Overview

The goal is to classify iris flowers into one of three species — *setosa*, *versicolor*, and *virginica* — based on four numerical measurements: sepal length, sepal width, petal length, and petal width. This project trains and tunes three different classifiers on the same data and compares how well each generalizes from training to unseen test data.

## Dataset

The well-known **Iris dataset** (150 samples, 3 balanced classes of 50 samples each, 4 numeric features), loaded directly via `sklearn.datasets.load_iris()`. The dataset has no missing values.

## Approach

### 1. Exploratory Data Analysis

Before modeling, the dataset was explored to understand feature distributions and relationships:

- Histograms of each feature's distribution.
- A correlation heatmap across all numerical features.
- A scatter plot of petal length vs. petal width, colored by species.

**Findings**: petal length and petal width show the clearest separation between species and are highly correlated with each other and with the target class, while sepal measurements overlap more between classes and are less discriminative. The scatter plot of petal dimensions confirmed visually distinct clusters per species, suggesting the classification task should be relatively tractable.

### 2. Train/Test Split

The four numeric features were used as inputs (`X`), with the species label as the target (`y`). The data was split into **70% training / 30% test** using a fixed random seed for reproducibility.

### 3. Decision Tree Classifier

A `DecisionTreeClassifier` was tuned using `RandomizedSearchCV` (5-fold cross-validation) over:

- `max_depth`: [2, 3, 5, 10]
- `min_samples_split`: [5, 10, 15]
- `min_samples_leaf`: [2, 5, 10]
- `criterion`: ['gini', 'entropy']

The best-found hyperparameters were used to refit a final Decision Tree, which was then evaluated on both the training and test sets (classification report + confusion matrix for each).

### 4. Random Forest Classifier

A `RandomForestClassifier` was tuned similarly via `RandomizedSearchCV` (5-fold cross-validation) over a broader search space:

- `n_estimators`: 5 values spanning 50–400 trees
- `max_features`: ['sqrt', 'log2']
- `max_depth`: 5 values spanning 10–100, plus unlimited depth
- `min_samples_split`: [2, 5, 10, 15]
- `min_samples_leaf`: [2, 5, 10, 15]
- `bootstrap`: [True, False]

As with the Decision Tree, the best hyperparameters were used to refit a final model, evaluated on both splits.

### 5. Support Vector Machine (SVM)

Since SVMs are sensitive to feature scale, the input features were first standardized using `StandardScaler` (fit on the training set, applied to both splits). An `SVC` classifier was then tuned via `RandomizedSearchCV` (5-fold cross-validation, 20 iterations) over:

- `C`: sampled uniformly between 0.1 and 1.1
- `kernel`: ['linear', 'rbf', 'poly', 'sigmoid']
- `gamma`: ['scale', 'auto']
- `degree`: [2, 3, 4] (for the polynomial kernel)

The best model from the search was evaluated on both the standardized training and test sets.

### 6. Model Comparison

Training and test accuracy for all three tuned classifiers were plotted side-by-side as a grouped bar chart, making it easy to visually compare both overall performance and the training/test accuracy gap (a proxy for overfitting) across models.

## Results

- All three classifiers achieved strong performance, which is expected given the Iris dataset's small size, lack of missing values, and largely well-separated classes.
- The **Decision Tree** performed slightly worse than the other two, with ~96% training accuracy and ~93% test accuracy.
- **Random Forest** and **SVM** both outperformed the Decision Tree, particularly on the test set, with Random Forest achieving marginally higher training accuracy than SVM.
- The gap between Random Forest/SVM and the single Decision Tree on the test set is consistent with the ensemble/margin-based nature of the former two, which tend to generalize better and are more robust to overfitting than a single tree.

## Repository Structure

```
.
└── DT_RF_SVM_Iris.ipynb   # Full notebook: EDA, model training, tuning, and comparison
```

## How to Run

1. Clone the repository and open the notebook in Jupyter, Google Colab, or your preferred environment.
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn scipy
   ```
3. Run all cells in order. No external dataset download is required — the Iris dataset is loaded directly from `scikit-learn`.

## Key Takeaways

- **EDA guided expectations before modeling**: visualizing feature correlations and class separability (especially for petal measurements) gave a strong prior that this classification task would be relatively easy for most standard classifiers.
- **`RandomizedSearchCV` offers an efficient middle ground** between manual tuning and exhaustive grid search, especially useful when the hyperparameter space (like the Random Forest's) is large.
- **Feature scaling matters for distance/margin-based models**: the SVM pipeline explicitly standardizes features before training, unlike the tree-based models, which are scale-invariant.
- **Ensemble and margin-based methods generalized slightly better** than a single Decision Tree in this case, illustrating the classic bias-variance tradeoff between simple interpretable models and more robust, higher-capacity ones — even on an "easy" dataset like Iris.
