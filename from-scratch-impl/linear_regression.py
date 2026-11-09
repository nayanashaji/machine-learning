import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
housing = fetch_california_housing(as_frame=True)
df = housing.frame
print(df.head())
X_original = df[['AveRooms']].values
y = df['MedHouseVal'].values
scaler = StandardScaler()
X = scaler.fit_transform(X_original)
X_train, X_test, y_train, y_test, X_plot_train, X_plot_test = train_test_split(
    X,
    y,
    X_original,
    test_size=0.2,
    random_state=42
)
tf.random.set_seed(42)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(1)
])
model.compile(
    optimizer=tf.keras.optimizers.SGD(
        learning_rate=0.01
    ),
    loss="mse"
)
model.fit(
    X_train,
    y_train,
    epochs=500,
    verbose=0
)
tf_pred = model.predict(
    X_test,
    verbose=0
).flatten()
tf_mse = mean_squared_error(
    y_test,
    tf_pred
)
tf_r2 = r2_score(
    y_test,
    tf_pred
)
weights = model.layers[0].get_weights()
tf_weight = weights[0][0][0]
tf_bias = weights[1][0]
x_train = X_train.flatten()
x_test = X_test.flatten()
w = 0.0
b = 0.0
learning_rate = 0.01
epochs = 1000
n = len(x_train)
for epoch in range(epochs):
    # Prediction
    y_pred = w * x_train + b
        # Gradients
    dw = (-2 / n) * np.sum(x_train * (y_train - y_pred))
    db = (-2 / n) * np.sum(y_train - y_pred)
        # Update Parameters
    w = w - learning_rate * dw
    b = b - learning_rate * db
    # Predictions
    gd_pred = w * x_test + b
    gd_mse = mean_squared_error(
    y_test,
    gd_pred
)
gd_r2 = r2_score(
    y_test,
    gd_pred
)
# Add bias column
X_train_ne = np.c_[
    np.ones(len(X_train)),
    X_train
]
# θ = (XᵀX)⁻¹ Xᵀy
theta = np.linalg.inv(
    X_train_ne.T @ X_train_ne
) @ X_train_ne.T @ y_train
# Test Data
X_test_ne = np.c_[
    np.ones(len(X_test)),
    X_test
]
ne_pred = X_test_ne @ theta
ne_mse = mean_squared_error(
    y_test,
    ne_pred
)
ne_r2 = r2_score(
    y_test,
    ne_pred
)
mean = scaler.mean_[0]
std = scaler.scale_[0]
def original_parameters(weight, bias):
    coefficient = weight / std
    intercept = bias - (weight * mean / std)
    return coefficient, intercept
# TensorFlow
tf_coefficient, tf_intercept = original_parameters(
    tf_weight,
    tf_bias
)
# Gradient Descent
gd_coefficient, gd_intercept = original_parameters(
    w,
    b
)
# Normal Equation
ne_coefficient, ne_intercept = original_parameters(
    theta[1],
    theta[0]
)
print("\n")
print("=" * 100)
print("COMPARISON OF MODELS")
print("=" * 100)
print(
    f"{'Method':<22}"
    f"{'Coefficient':<15}"
    f"{'Intercept':<15}"
    f"{'MSE':<15}"
    f"{'R² Score'}"
)
print("-" * 100)
print(
    f"{'TensorFlow':<22}"
    f"{tf_coefficient:<15.5f}"
    f"{tf_intercept:<15.5f}"
    f"{tf_mse:<15.5f}"
    f"{tf_r2:.5f}"
)
print(
    f"{'Gradient Descent':<22}"
    f"{gd_coefficient:<15.5f}"
    f"{gd_intercept:<15.5f}"
    f"{gd_mse:<15.5f}"
    f"{gd_r2:.5f}"
)
print(
    f"{'Normal Equation':<22}"
    f"{ne_coefficient:<15.5f}"
    f"{ne_intercept:<15.5f}"
    f"{ne_mse:<15.5f}"
    f"{ne_r2:.5f}"
)
# Sort x values so regression lines are smooth
index = np.argsort(X_plot_test.flatten())
x_plot = X_plot_test.flatten()[index]
# Corresponding normalized values
x_scaled = X_test.flatten()[index]
# TensorFlow prediction
tf_line = model.predict(
    x_scaled.reshape(-1, 1),
    verbose=0
).flatten()
# Gradient Descent prediction
gd_line = w * x_scaled + b
# Normal Equation prediction
ne_line = theta[1] * x_scaled + theta[0]
fig, ax = plt.subplots(
    1,
    3,
    figsize=(18, 5),
    sharex=True,
    sharey=True
)
titles = [
    f"TensorFlow\nMSE={tf_mse:.3f}, R²={tf_r2:.3f}",
    f"Gradient Descent\nMSE={gd_mse:.3f}, R²={gd_r2:.3f}",
    f"Normal Equation\nMSE={ne_mse:.3f}, R²={ne_r2:.3f}"
]
predictions = [
    tf_line,
    gd_line,
    ne_line
]
colors = [
    "red",
    "green",
    "purple"
]
for i in range(3):
    ax[i].scatter(
        X_plot_test,
        y_test,
        color="blue",
        s=12,
        label="Actual Data"
    )
    ax[i].plot(
        x_plot,
        predictions[i],
        color=colors[i],
        linewidth=2,
        label="Regression Line"
    )
    ax[i].set_title(titles[i])
    ax[i].set_xlabel("Average Rooms")
    ax[i].grid(True)
    ax[i].legend()
    ax[0].set_ylabel("Median House Value")
    xmin = X_plot_test.min()
xmax = X_plot_test.max()
ymin = min(
    y_test.min(),
    tf_line.min(),
    gd_line.min(),
    ne_line.min()
)
ymax = max(
    y_test.max(),
    tf_line.max(),
    gd_line.max(),
    ne_line.max()
)
for axis in ax:
    axis.set_xlim(xmin, xmax)
    axis.set_ylim(ymin, ymax)
    plt.tight_layout()
plt.show()