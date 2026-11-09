import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

columns = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigree", "Age", "Outcome"
]

df = pd.read_csv("datasets/diabetes.csv")

X = df.drop("Outcome", axis=1).values
y = df["Outcome"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def train_logistic(X, y, lr=0.01, epochs=3000):

    w = np.zeros(X.shape[1])
    b = 0

    for _ in range(epochs):

        p = sigmoid(X @ w + b)

        dw = (X.T @ (p - y)) / len(y)
        db = np.mean(p - y)

        w -= lr * dw
        b -= lr * db

    return w, b


def predict(X, w, b):
    return (sigmoid(X @ w + b) >= 0.5).astype(int)


# Without scaling
w, b = train_logistic(X_train, y_train)

pred_scratch = predict(
    X_test, w, b
)

print("FROM SCRATCH - WITHOUT SCALING")
print("Accuracy:", accuracy_score(y_test, pred_scratch))

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

w_s, b_s = train_logistic(
    X_train_scaled,
    y_train
)

pred_scratch_scaled = predict(
    X_test_scaled,
    w_s,
    b_s
)

print("\nFROM SCRATCH - WITH SCALING")
print("Accuracy:",
      accuracy_score(y_test, pred_scratch_scaled))

model = LogisticRegression(max_iter=1000)

model.fit(
    X_train,
    y_train
)

pred_builtin = model.predict(X_test)

print("\nBUILT-IN - WITHOUT SCALING")
print("Accuracy:",
      accuracy_score(y_test, pred_builtin))


# Built-in with scaling
model_scaled = LogisticRegression(
    max_iter=1000
)

model_scaled.fit(
    X_train_scaled,
    y_train
)

pred_builtin_scaled = model_scaled.predict(
    X_test_scaled
)

print("\nBUILT-IN - WITH SCALING")
print("Accuracy:",
      accuracy_score(y_test, pred_builtin_scaled))

def metrics(y, pred):

    return [
        accuracy_score(y, pred),
        precision_score(y, pred, zero_division=0),
        recall_score(y, pred, zero_division=0),
        f1_score(y, pred, zero_division=0)
    ]


results = pd.DataFrame(
    [
        metrics(y_test, pred_scratch),
        metrics(y_test, pred_scratch_scaled),
        metrics(y_test, pred_builtin),
        metrics(y_test, pred_builtin_scaled)
    ],
    columns=[
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],
    index=[
        "Scratch - No Scaling",
        "Scratch - Scaling",
        "Library - No Scaling",
        "Library - Scaling"
    ]
)

print("\nRESULT COMPARISON")
print(results)