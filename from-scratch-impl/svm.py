import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

iris = load_iris()

# First two features for 2D visualization
X = iris.data[:, :2]

# Setosa = +1, Non-Setosa = -1
y = np.where(iris.target == 0, 1, -1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_all = scaler.transform(X)

def svm_smo(X, y, C=1.0, tol=0.001, max_passes=20):

    n = len(X)

    # Lagrange multipliers
    alpha = np.zeros(n)

    b = 0.0

    # Linear kernel
    K = X @ X.T

    passes = 0

    while passes < max_passes:

        changed = 0

        for i in range(n):

            Ei = np.sum(
                alpha * y * K[:, i]
            ) + b - y[i]

            # KKT condition
            if ((y[i] * Ei < -tol and alpha[i] < C) or
                (y[i] * Ei > tol and alpha[i] > 0)):

                # Select another point j
                j = np.random.choice(
                    [x for x in range(n) if x != i]
                )

                Ej = np.sum(
                    alpha * y * K[:, j]
                ) + b - y[j]

                old_ai = alpha[i]
                old_aj = alpha[j]

                # Bounds for alpha[j]
                if y[i] != y[j]:

                    L = max(
                        0,
                        alpha[j] - alpha[i]
                    )

                    H = min(
                        C,
                        C + alpha[j] - alpha[i]
                    )

                else:

                    L = max(
                        0,
                        alpha[i] + alpha[j] - C
                    )

                    H = min(
                        C,
                        alpha[i] + alpha[j]
                    )

                if L == H:
                    continue

                # Second derivative
                eta = (
                    2 * K[i, j]
                    - K[i, i]
                    - K[j, j]
                )

                if eta >= 0:
                    continue

                # Update alpha[j]
                alpha[j] -= (
                    y[j] * (Ei - Ej)
                ) / eta

                alpha[j] = np.clip(
                    alpha[j],
                    L,
                    H
                )

                if abs(
                    alpha[j] - old_aj
                ) < 1e-5:

                    alpha[j] = old_aj
                    continue

                # Update alpha[i]
                alpha[i] += (
                    y[i] * y[j]
                    * (old_aj - alpha[j])
                )

                # Calculate two possible b values
                b1 = (
                    b - Ei
                    - y[i] * (alpha[i] - old_ai)
                    * K[i, i]
                    - y[j] * (alpha[j] - old_aj)
                    * K[i, j]
                )

                b2 = (
                    b - Ej
                    - y[i] * (alpha[i] - old_ai)
                    * K[i, j]
                    - y[j] * (alpha[j] - old_aj)
                    * K[j, j]
                )

                # Select b
                if 0 < alpha[i] < C:
                    b = b1

                elif 0 < alpha[j] < C:
                    b = b2

                else:
                    b = (b1 + b2) / 2

                changed += 1

        if changed == 0:
            passes += 1
        else:
            passes = 0

    # Calculate weight vector
    w = np.sum(
        (alpha * y)[:, None] * X,
        axis=0
    )

    return w, b, alpha


w, b, alpha = svm_smo(
    X_train,
    y_train,
    C=1
)

def predict(X, w, b):

    return np.where(
        X @ w + b >= 0,
        1,
        -1
    )


y_pred_scratch = predict(
    X_test,
    w,
    b
)

print("FROM SCRATCH SVM")
print("Accuracy:",
      accuracy_score(y_test, y_pred_scratch))

model = SVC(
    kernel="linear",
    C=1
)

model.fit(
    X_train,
    y_train
)

y_pred_builtin = model.predict(
    X_test
)

print("\nBUILT-IN SVM")
print("Accuracy:",
      accuracy_score(y_test, y_pred_builtin))

def plot_svm(ax, w, b, title, support_vectors):

    # Setosa
    ax.scatter(
        X_all[y == 1, 0],
        X_all[y == 1, 1],
        label="Setosa"
    )

    # Non-Setosa
    ax.scatter(
        X_all[y == -1, 0],
        X_all[y == -1, 1],
        label="Non-Setosa"
    )

    # X values
    x = np.linspace(
        -2.5,
        2.5,
        300
    )

    # Decision boundary
    decision = (
        -(w[0] * x + b)
        / w[1]
    )

    # +1 margin
    margin_positive = (
        -(w[0] * x + b - 1)
        / w[1]
    )

    # -1 margin
    margin_negative = (
        -(w[0] * x + b + 1)
        / w[1]
    )

    # Decision boundary
    ax.plot(
        x,
        decision,
        linewidth=2,
        label="Decision Boundary"
    )

    # Margins
    ax.plot(
        x,
        margin_positive,
        "--",
        label="Margin"
    )

    ax.plot(
        x,
        margin_negative,
        "--"
    )

    # Support vectors
    ax.scatter(
        support_vectors[:, 0],
        support_vectors[:, 1],
        s=100,
        facecolors="none",
        edgecolors="black",
        linewidths=1.5,
        label="Support Vectors"
    )

    ax.set_title(title)

    ax.set_xlabel(
        "Sepal Length (scaled)"
    )

    ax.set_ylabel(
        "Sepal Width (scaled)"
    )

    # EXACT SAME SCALE
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)

    ax.set_aspect("equal")

    ax.grid(alpha=0.2)

    ax.legend()

# Scratch SVM support vectors
scratch_sv = X_train[
    alpha > 1e-4
]

# Built-in support vectors
builtin_sv = model.support_vectors_

fig, axes = plt.subplots(
    1,
    2,
    figsize=(16, 7)
)

plot_svm(
    axes[0],
    w,
    b,
    "SVM From Scratch",
    scratch_sv
)

plot_svm(
    axes[1],
    model.coef_[0],
    model.intercept_[0],
    "Built-in Linear SVM",
    builtin_sv
)

fig.suptitle(
    "SVM on Iris Dataset: Setosa vs Non-Setosa",
    fontsize=16
)

plt.tight_layout()
plt.show()