import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.title("Credit Card Default Prediction")
st.write("Machine Learning Assignment - Multiple Classification Models")

uploaded_file = st.file_uploader("Upload UCI Credit Card CSV", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Drop ID column if exists
    if "ID" in df.columns:
        df = df.drop("ID", axis=1)

    target_column = "default.payment.next.month"

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model_name = st.selectbox("Select Model", [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ])

    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)

    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier()

    elif model_name == "KNN":
        model = KNeighborsClassifier()

    elif model_name == "Naive Bayes":
        model = GaussianNB()

    elif model_name == "Random Forest":
        model = RandomForestClassifier()

    else:
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    st.subheader("Evaluation Metrics")

    st.write("Accuracy:", accuracy_score(y_test, predictions))
    st.write("Precision:", precision_score(y_test, predictions))
    st.write("Recall:", recall_score(y_test, predictions))
    st.write("F1 Score:", f1_score(y_test, predictions))
    st.write("AUC Score:", roc_auc_score(y_test, predictions))
    st.write("MCC Score:", matthews_corrcoef(y_test, predictions))

    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y_test, predictions))

    st.subheader("Classification Report")
    st.text(classification_report(y_test, predictions))
