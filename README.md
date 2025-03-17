# Credit Card Fraud Detection

## Project Overview
This project focuses on detecting fraudulent credit card transactions using machine learning techniques. Given a highly imbalanced dataset, the objective is to accurately classify fraudulent transactions while minimizing false positives and false negatives.

## Dataset
- The dataset consists of credit card transactions made by European cardholders in September 2013.
- It includes transactions over two days, containing 284,807 transactions, out of which 492 are fraudulent.
- The dataset is highly imbalanced, with fraud cases accounting for only 0.172% of all transactions.

## Problem Statement
Credit risk is associated with the possibility of a client failing to meet contractual obligations, such as mortgages, credit card debts, and other types of loans. The goal is to predict whether a given transaction is real or fraudulent.

## Solution Approach
The project implements two main techniques to handle the imbalanced dataset:

### 1. Ensemble Technique
- **Preprocessing**:
  - Checked for missing values (none found).
  - Separated features and target labels.
  - Split the dataset into training and testing sets.
- **Model Training**:
  - Used 10 different classifiers: Random Forest, Decision Tree, SVC, KNN, Logistic Regression, GaussianNB, AdaBoost, Gradient Boosting, CatBoost, and XGBoost.
  - Recorded accuracy, F1-score, and Kappa score for comparison.
  - Identified **XGBoost** as the best-performing classifier.
- **Hyperparameter Tuning**:
  - Used **GridSearchCV** to find the optimal parameters for XGBoost.
  - Trained the model with the best parameters, improving accuracy.

### 2. Resampling Technique
- **Preprocessing**:
  - Normalized 'Time' and 'Amount' using **RobustScaler**.
- **Handling Class Imbalance**:
  - Applied **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the dataset.
  - Used a combination of **SMOTE and Random Undersampling** for better results.
- **Dimensionality Reduction**:
  - Applied **PCA, LDA, and ICA** to visualize data and improve model performance.
  - PCA was tested despite being unsupervised; LDA and ICA performed better for this supervised task.
- **Model Training and Evaluation**:
  - Used 10 classifiers on three datasets (raw, SMOTE-resampled, and SMOTE + undersampled).
  - Evaluated models based on **F1-score and Kappa score** instead of accuracy due to class imbalance.
  - Identified **XGBoost** as the best-performing algorithm for resampled data as well.

## Results & Observations
- **XGBoost** provided the best results across both Ensemble and Resampling techniques.
- **LDA performed better** than PCA for dimensionality reduction in this supervised setting.
- The combination of **SMOTE and Random Undersampling** improved fraud detection performance.
- **Evaluation metrics like F1-score and Kappa score** provided a more accurate measure of model effectiveness.

## Visualizations
The following plots were generated for better understanding and analysis:
- Count plot for class distribution.
- Distribution plot for 'Time' and 'Amount' after normalization.
- Scatter plots for raw, PCA-reduced, LDA-reduced, and ICA-reduced datasets.
- Scatter plots for SMOTE-resampled and SMOTE + undersampled datasets.

## Conclusion
This project successfully implements multiple machine learning techniques to detect fraudulent credit card transactions. By using ensemble learning and resampling techniques, we achieve better fraud detection while addressing dataset imbalance. The results indicate that XGBoost, combined with proper data preprocessing and dimensionality reduction, is an effective approach for credit card fraud detection.

## Future Work
- Collect additional real-world data to enhance model generalization.
- Explore deep learning techniques such as neural networks.
- Implement real-time fraud detection with streaming data.
- Further optimize hyperparameters for improved performance.

---
**Developed by:**  
- Chaitravi Kane
