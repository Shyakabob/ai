import numpy as np
import pandas as pd
import streamlit as st
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


@st.cache_data
def load_iris():
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="target")
    target_names = iris.target_names
    return X, y, target_names


def train_model(model_name: str, X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=500)
    elif model_name == "SVC (RBF)":
        model = SVC(probability=True)
    elif model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=200, random_state=42)
    elif model_name == "KNN":
        model = KNeighborsClassifier(n_neighbors=5)
    else:
        raise ValueError("Unknown model")

    model.fit(X_train_s, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_s))
    return model, scaler, acc


def main():
    st.title("Iris Classifier Playground")
    st.write("Train a model and make predictions interactively.")

    X, y, target_names = load_iris()

    with st.sidebar:
        st.header("Model and Inputs")
        model_name = st.selectbox(
            "Select model", ["Logistic Regression", "SVC (RBF)", "Random Forest", "KNN"]
        )
        st.markdown("---")
        sepal_length = st.slider("sepal length (cm)", float(X.iloc[:, 0].min()), float(X.iloc[:, 0].max()), 5.1, 0.1)
        sepal_width = st.slider("sepal width (cm)", float(X.iloc[:, 1].min()), float(X.iloc[:, 1].max()), 3.5, 0.1)
        petal_length = st.slider("petal length (cm)", float(X.iloc[:, 2].min()), float(X.iloc[:, 2].max()), 1.4, 0.1)
        petal_width = st.slider("petal width (cm)", float(X.iloc[:, 3].min()), float(X.iloc[:, 3].max()), 0.2, 0.1)

    model, scaler, acc = train_model(model_name, X, y)
    st.success(f"Validation accuracy: {acc:.3f}")

    sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    sample_s = scaler.transform(sample)
    pred = int(model.predict(sample_s)[0])
    probs = getattr(model, "predict_proba", None)

    st.subheader("Prediction")
    st.write(f"Predicted class: **{target_names[pred]}**")
    if probs is not None:
        proba = model.predict_proba(sample_s)[0]
        prob_df = pd.DataFrame({"class": target_names, "probability": proba})
        st.bar_chart(prob_df.set_index("class"))

    st.markdown("---")
    st.caption("Data: scikit-learn Iris. This app retrains on each change for simplicity.")


if __name__ == "__main__":
    main()


