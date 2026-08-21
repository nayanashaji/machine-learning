import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def knn_predict(X_train, y_train, X_test, k):

    predictions = []

    for x in X_test:

        # Euclidean distance
        distances = np.sqrt(
            np.sum((X_train - x) ** 2, axis=1)
        )

        # Get k nearest points
        nearest = np.argsort(distances)[:k]

        # Majority class
        classes, counts = np.unique(
            y_train[nearest],
            return_counts=True
        )

        predictions.append(
            classes[np.argmax(counts)]
        )

    return np.array(predictions)

k_values = [3, 5, 7, 9]

scratch_results = []
library_results = []

for k in k_values:

    # From scratch
    pred_scratch = knn_predict(
        X_train,
        y_train,
        X_test,
        k
    )

    scratch_acc = accuracy_score(
        y_test,
        pred_scratch
    )

    # Built-in
    model = KNeighborsClassifier(n_neighbors=k)

    model.fit(X_train, y_train)

    pred_library = model.predict(X_test)

    library_acc = accuracy_score(
        y_test,
        pred_library
    )

    scratch_results.append(scratch_acc)
    library_results.append(library_acc)

results = pd.DataFrame({
    "K": k_values,
    "From Scratch": scratch_results,
    "Built-in": library_results
})

print("\nKNN RESULTS")
print(results.to_string(index=False))