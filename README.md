# multiple-classification-models
multiple-classification-models
## Problem Statement

The objective of this project is to predict whether a credit card client will default on payment in the next month using classification machine learning models.

## Dataset Description

Dataset Name: UCI Credit Card Default Dataset  
Source: Kaggle  

- Number of instances: 30,000  
- Number of features: 23  
- Target Variable: default.payment.next.month  
- Type: Binary Classification (0 = No Default, 1 = Default)

## Machine Learning Models Used

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Naive Bayes (GaussianNB)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

## Evaluation Metrics

The following evaluation metrics were used for each model:

- Accuracy
- Precision
- Recall
- F1 Score
- AUC Score
- Matthews Correlation Coefficient (MCC)

## Model Comparison Table

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------|----------|-----|-----------|--------|----|-----|
| Logistic Regression |  |  |  |  |  |  |
| Decision Tree |  |  |  |  |  |  |
| KNN |  |  |  |  |  |  |
| Naive Bayes |  |  |  |  |  |  |
| Random Forest |  |  |  |  |  |  |
| XGBoost |  |  |  |  |  |  |

## Observations

- Logistic Regression performs well for linear relationships.
- Decision Tree may overfit on training data.
- KNN performance depends on value of K.
- Naive Bayes assumes feature independence.
- Random Forest improves stability and reduces overfitting.
- XGBoost generally provides strong performance due to boosting.

## Streamlit Application

The web application allows:

- Uploading dataset (CSV)
- Selecting classification model
- Viewing evaluation metrics
- Viewing confusion matrix and classification report

