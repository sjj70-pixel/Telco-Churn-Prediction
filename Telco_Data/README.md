# Telecom Customer Churn Prediction

## Objective

Predict customer churn using logistic regression and random forest models, and identify the key drivers of churn.

## Dataset

IBM Telco Customer Churn dataset.

## Methods

* Data cleaning and preprocessing (handling missing values, encoding categorical variables)
* Stratified train/validation/test split
* Cross-validation using ROC-AUC
* Threshold tuning using a validation set (target recall ≈ 75%)
* Final evaluation on a held-out test set

## Key Findings

* Tenure is the strongest predictor: longer tenure reduces churn risk
* Contract type is critical: month-to-month contracts increase churn, while long-term contracts reduce it
* Internet type matters: fiber customers churn more than DSL customers
* Both models achieve similar performance (~0.83–0.84 ROC-AUC), suggesting churn is largely driven by simple, linear relationships; logistic regression is preferred for its interpretability and efficiency

## Tools

Python, pandas, scikit-learn, matplotlib