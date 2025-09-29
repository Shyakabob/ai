import pathlib
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_iris_dataframe() -> Tuple[pd.DataFrame, pd.Series]:
    iris = datasets.load_iris()
    features = pd.DataFrame(iris.data, columns=iris.feature_names)
    target = pd.Series(iris.target, name="target")
    return features, target


def train_test_split_scaled(
    features: pd.DataFrame, target: pd.Series
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    X_train, X_test, y_train, y_test = train_test_split(
        features.values, target.values, test_size=0.2, random_state=42, stratify=target
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(max_iter=500, multi_class="auto")
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: LogisticRegression, X_test: np.ndarray, y_test: np.ndarray
) -> None:
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", round(acc, 4))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


def save_artifacts(model: LogisticRegression, scaler: StandardScaler, path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path / "iris_logreg.joblib")
    joblib.dump(scaler, path / "scaler.joblib")
    print(f"Saved model and scaler to: {path.resolve()}")


def demo_prediction(model: LogisticRegression, scaler: StandardScaler) -> None:
    sample = np.array([[5.1, 3.5, 1.4, 0.2]])
    sample_scaled = scaler.transform(sample)
    pred = model.predict(sample_scaled)[0]
    print(f"\nDemo prediction for [5.1, 3.5, 1.4, 0.2]: class {pred}")


def main() -> None:
    print("Loading Iris dataset...")
    features, target = load_iris_dataframe()
    print("Splitting and scaling data...")
    X_train, X_test, y_train, y_test, scaler = train_test_split_scaled(features, target)
    print("Training Logistic Regression model...")
    model = train_model(X_train, y_train)
    print("Evaluating model...")
    evaluate_model(model, X_test, y_test)
    print("Saving artifacts...")
    save_artifacts(model, scaler, pathlib.Path("models"))
    demo_prediction(model, scaler)


if __name__ == "__main__":
    main()


