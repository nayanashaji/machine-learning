import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
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

# Convert target to 0 and 1
df["income"] = df["income"].str.strip().map({
    "<=50K": 0,
    ">50K": 1
})

# Encode categorical features
for col in df.select_dtypes("object").columns:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop("income", axis=1).values
y = df["income"].values

features = columns[:-1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

def entropy(y):
    counts = np.bincount(y)
    p = counts[counts > 0] / len(y)
    return -np.sum(p * np.log2(p))


def best_split(X, y):

    best_gain = 0
    best_feature = None
    best_threshold = None

    for f in range(X.shape[1]):

        for t in np.unique(X[:, f]):

            left = y[X[:, f] <= t]
            right = y[X[:, f] > t]

            if len(left) == 0 or len(right) == 0:
                continue

            gain = entropy(y) - (
                len(left) / len(y) * entropy(left)
                + len(right) / len(y) * entropy(right)
            )

            if gain > best_gain:
                best_gain = gain
                best_feature = f
                best_threshold = t

    return best_feature, best_threshold


def build_tree(X, y, depth=0, max_depth=6):

    # Stop if all samples belong to one class
    if len(np.unique(y)) == 1 or depth == max_depth:
        return Counter(y).most_common(1)[0][0]

    feature, threshold = best_split(X, y)

    if feature is None:
        return Counter(y).most_common(1)[0][0]

    left = X[:, feature] <= threshold
    right = ~left

    return (
        feature,
        threshold,
        build_tree(X[left], y[left], depth + 1, max_depth),
        build_tree(X[right], y[right], depth + 1, max_depth)
    )


def predict(x, tree):

    # Leaf node
    if not isinstance(tree, tuple):
        return tree

    feature, threshold, left, right = tree

    if x[feature] <= threshold:
        return predict(x, left)
    else:
        return predict(x, right)


# Train from scratch
tree = build_tree(X_train, y_train)

# Predict
y_pred_scratch = np.array([
    predict(x, tree) for x in X_test
])

model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

y_pred_builtin = model.predict(X_test)

def results(y_true, y_pred):

    return [
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred, zero_division=0),
        recall_score(y_true, y_pred, zero_division=0),
        f1_score(y_true, y_pred, zero_division=0)
    ]


comparison = pd.DataFrame(
    [
        results(y_test, y_pred_scratch),
        results(y_test, y_pred_builtin)
    ],
    columns=[
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],
    index=[
        "From Scratch",
        "Sklearn"
    ]
)

print("\nMODEL COMPARISON")
print(comparison)

importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=False)

print("\nFEATURE IMPORTANCE")
print(importance)

plt.figure(figsize=(20, 10))

plot_tree(
    model,
    feature_names=features,
    class_names=["<=50K", ">50K"],
    filled=True,
    max_depth=3
)

plt.title("Decision Tree - Adult Income")
plt.show()