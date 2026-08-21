import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

columns = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week",
    "native_country", "income"
]

df = pd.read_csv(
    "adult.data",
    names=columns,
    skipinitialspace=True
)

# Remove missing values
df = df.replace("?", np.nan).dropna()

# Convert target
df["income"] = df["income"].str.strip().map({
    "<=50K": 0,
    ">50K": 1
})

# Encode categorical features
for col in df.select_dtypes("object").columns:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop("income", axis=1)
y = df["income"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logistic = LogisticRegression(max_iter=1000)

logistic.fit(X_train_scaled, y_train)

y_pred_logistic = logistic.predict(X_test_scaled)

tree = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=6,
    random_state=42
)

tree.fit(X_train, y_train)

y_pred_tree = tree.predict(X_test)

def evaluate(y_true, y_pred):
    return [
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred, zero_division=0),
        recall_score(y_true, y_pred, zero_division=0),
        f1_score(y_true, y_pred, zero_division=0)
    ]


results = pd.DataFrame(
    [
        evaluate(y_test, y_pred_logistic),
        evaluate(y_test, y_pred_tree)
    ],
    columns=["Accuracy", "Precision", "Recall", "F1 Score"],
    index=["Logistic Regression", "Decision Tree"]
)

print("\nMODEL COMPARISON")
print(results)