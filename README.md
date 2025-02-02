# Breast Cancer Prediction Using Support Vector Classifier and Regressor

## Overview

This project implements a machine learning solution for breast cancer prediction by combining two powerful techniques from the Support Vector Machine (SVM) family: a Support Vector Classifier (SVC) and a Support Vector Regressor (SVR). The SVC is used for the binary classification task of predicting whether a tumor is malignant or benign, while the SVR is applied to predict a continuous outcome such as risk scores or tumor severity metrics.

By leveraging robust SVM-based models and performing thorough feature engineering and hyperparameter tuning, this project aims to deliver an accurate, interpretable, and efficient diagnostic tool that can assist in early detection and treatment planning for breast cancer patients.

## Motivation

Breast cancer is one of the leading causes of death among women worldwide. Early diagnosis and accurate prognosis are crucial for improving treatment outcomes. However, manual analysis of medical data and imaging is time consuming and prone to human error. Machine learning methods—particularly those based on SVMs—offer several advantages:

- **High Accuracy:** SVMs are effective in high-dimensional spaces and can handle non-linear decision boundaries with kernel tricks.
- **Robustness:** These models are less prone to overfitting in many practical scenarios.
- **Interpretability:** With proper feature selection and model evaluation, SVMs can highlight the most significant factors affecting the prediction.

This project leverages these strengths to build a system that not only classifies breast tumors accurately but also provides a regression-based risk assessment to support clinical decision-making.

## Dataset

The dataset used to train the model includes multiple features extracted from digitized images of fine needle aspirate (FNA) of breast masses. Typical features include:

- **Radius, Texture, Perimeter, Area:** Descriptive measurements of cell nuclei.
- **Smoothness, Compactness, Concavity:** Indicators of how irregular or clustered the cells are.
- **Additional Statistical Measures:** Mean, standard error, and worst (largest) values for each feature.

The SVC model uses these features to classify tumors as malignant or benign, while the SVR is used to predict a continuous outcome (such as a risk score or tumor aggressiveness).

## Methodology

### 1. Data Preprocessing
- **Data Cleaning:** Handle missing values, remove duplicates, and correct any inconsistencies.
- **Normalization/Standardization:** Scale features using methods like StandardScaler or MinMaxScaler so that each attribute contributes equally.
- **Feature Selection/Engineering:** Evaluate the relevance of different features using techniques such as correlation analysis, recursive feature elimination, or other statistical methods.

### 2. Model Development

#### Support Vector Classifier (SVC)
- **Objective:** Classify tumors as malignant or benign.
- **Implementation:** Utilize scikit-learn’s `SVC` with an appropriate kernel (e.g., radial basis function (RBF)). Hyperparameters (C, gamma, etc.) are tuned via grid search or cross-validation.
- **Performance:** The SVC model achieves an accuracy of **97.9%**, demonstrating its effectiveness in accurately classifying breast tumors.

#### Support Vector Regressor (SVR)
- **Objective:** Predict a continuous outcome such as a risk score or tumor severity metric.
- **Implementation:** Use scikit-learn’s `SVR` with a chosen kernel and optimized hyperparameters.
- **Performance:** The SVR model achieves an R² score of **87.5%**, indicating strong performance in predicting continuous outcomes.

### 3. Experimental Setup
- **Programming Language:** Python
- **Libraries:** scikit-learn, Pandas, NumPy, Matplotlib, Seaborn
- **Environment:** Jupyter Notebook (or your preferred IDE)

### 4. Results and Discussion
- **Evaluation:** 
  - For classification, evaluation metrics include accuracy, precision, recall, F1-score, confusion matrices, and ROC curves.
  - For regression, evaluation metrics include mean squared error (MSE) and the R² score.
- **Insights:** Feature importance analysis identifies which features (e.g., radius, texture, perimeter) are most predictive of cancer severity.
- **Key Metrics:** 
  - **SVC Accuracy:** 97.9%
  - **SVR R² Score:** 87.5%
- **Discussion:** The results show that SVM-based methods are highly effective for both classification and regression tasks in breast cancer diagnosis. Future improvements might include integrating additional datasets, employing ensemble methods, or refining feature selection strategies.
