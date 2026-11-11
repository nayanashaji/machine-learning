# Machine Learning Lab

A collection of Machine Learning experiments and implementations exploring fundamental algorithms, statistical methods, model evaluation techniques, and neural networks using Python.

This repository contains practical implementations of machine learning concepts, including experiments using popular Python libraries and selected algorithms implemented from scratch.

##  Topics Covered

### 1. Regression
- Linear Regression
- Polynomial Regression
- Ridge and Lasso Regression
- Bias–Variance Tradeoff

### 2. Classification
- Logistic Regression
- Decision Trees (ID3)
- K-Nearest Neighbors (KNN)
- Naive Bayes
  - Multinomial Naive Bayes
  - Bernoulli Naive Bayes
- Support Vector Machines (SVM)
  - Linear SVM
  - Kernel-based SVM

### 3. Statistical Learning
- Maximum Likelihood Estimation (MLE)
- Maximum A Posteriori (MAP) Estimation
- Multinomial Distribution
- Dirichlet Priors

### 4. Neural Networks
- Multilayer Perceptron (MLP)
- Activation Functions
- Hyperparameter Tuning

### 5. Clustering
- K-Means Clustering
- Agglomerative Clustering

### 6. Ensemble Learning and Model Evaluation
- Bagging and Boosting
- Bootstrap Sampling
- Cross-Validation
- Model Performance Evaluation

##  Repository Structure

```text
machine-learning/
│
├── datasets/                    # Datasets used in experiments
│
├── from-scratch-impl/           # Algorithm implementations from scratch
│
├── activation_functions.py
├── agglomerative_kmeans.py
├── bagging_boosting.py
├── bias_variance.py
├── bootstrap.py
├── decision_tree_library.py
├── hyperparameter_tuning.py
├── kmeans.py
├── knn_library.py
├── linear_regression_library.py
├── logistic_regression_decision_tree.py
├── logistic_regression_library.py
├── mle_map.py
├── mle_map_multinomial.py
├── mlp.py
├── naive_bayes_library.py
├── polynomial_library.py
├── ridge_lasso.py
├── svm_kernels.py
├── svm_library.py
│
├── requirements.txt
└── README.md
```

##  Technologies Used

- **Python** – Primary programming language
- **NumPy** – Numerical computations
- **Pandas** – Data manipulation and preprocessing
- **Matplotlib / Seaborn** – Data visualization
- **Scikit-learn** – Machine learning algorithms, datasets, and evaluation
- **SciPy** – Scientific computing

##  Getting Started

### Prerequisites

- Python 3.x
- pip
- Git

### 1. Clone the repository

```bash
git clone https://github.com/nayanashaji/machine-learning.git
cd machine-learning
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows:**
```bash
.venv\Scripts\activate
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run an experiment

Each experiment can be executed independently.

For example:

```bash
python linear_regression_library.py
```

Replace the filename with the experiment you want to run.

##  Datasets

The repository includes a `datasets/` directory and experiments using datasets such as:

- California Housing
- Auto MPG
- Diabetes
- Breast Cancer Wisconsin
- 20 Newsgroups
- Pima Indians Diabetes
- Fashion-MNIST
- Iris
- Wine Quality
- Adult Income
- Titanic
- Digits
- Mall Customers

Some experiments may load datasets directly through libraries such as Scikit-learn.

##  Learning Objectives

Through these experiments, the repository explores:

- Understanding the mathematical foundations of machine learning algorithms.
- Training and evaluating regression and classification models.
- Comparing different algorithms and their performance.
- Understanding statistical estimation techniques such as MLE and MAP.
- Exploring clustering and dimensionality reduction concepts.
- Studying ensemble methods and model generalization.
- Understanding the impact of hyperparameters on model performance.
- Implementing machine learning algorithms from scratch to understand their underlying mechanisms.

---
