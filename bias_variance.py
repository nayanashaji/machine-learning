"""

Investigate the bias-variance tradeoff using polynomial regression on the Boston Housing
dataset. Plot the training and validation errors for various polynomial degrees and discuss
the tradeoff between bias and variance.
Tasks:
● Load and preprocess the Boston Housing dataset.
● Implement polynomial regression with varying degrees.
● Plot training and validation errors for each degree.
● Discuss the bias-variance tradeoff and its impact on model performance.

"""
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data=load_diabetes()
X=data.data
y=data.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

train_err=[]
test_err=[]
degrees=range(1,11)

for d in degrees:
    poly=PolynomialFeatures(degree=d)
    X_train_p=poly.fit_transform(X_train)
    X_test_p=poly.transform(X_test)
    model=LinearRegression()
    model.fit(X_train_p,y_train)
    train_pred=model.predict(X_train_p)
    test_pred=model.predict(X_test_p)
    train_err.append(mean_squared_error(y_train,train_pred))
    test_err.append(mean_squared_error(y_test,test_pred))

for degree, train_error, test_error in zip(degrees, train_err, test_err):
    print(
        "Degree:", degree,
        "Train MSE:", train_error,
        "Validation MSE:", test_error
    )

plt.plot(degrees,train_err,label="training error")
plt.plot(degrees,test_err,label="testing error")
plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Squared Error")
plt.title("Bias-Variance tradeoff")
plt.legend()
plt.grid()
plt.show()

# import numpy as np
# import matplotlib.pyplot as plt

# from sklearn.datasets import fetch_openml
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression
# from sklearn.pipeline import make_pipeline
# from sklearn.metrics import mean_squared_error


# # Load Boston Housing dataset
# data = fetch_openml(
#     name="boston",
#     version=1,
#     as_frame=True
# )

# df = data.frame

# print(df.head())

# # Use RM (average number of rooms) as input
# X = df[["RM"]].astype(float)
# y = df["MEDV"].astype(float)

# # Train-validation split
# X_train, X_val, y_train, y_val = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# degrees = range(1, 11)

# train_errors = []
# val_errors = []

# # Try different polynomial degrees
# for degree in degrees:

#     model = make_pipeline(
#         PolynomialFeatures(degree),
#         LinearRegression()
#     )

#     model.fit(X_train, y_train)

#     train_pred = model.predict(X_train)
#     val_pred = model.predict(X_val)

#     train_errors.append(
#         mean_squared_error(y_train, train_pred)
#     )

#     val_errors.append(
#         mean_squared_error(y_val, val_pred)
#     )

# # Print errors
# for degree, train_error, val_error in zip(
#     degrees, train_errors, val_errors
# ):
#     print(
#         "Degree:", degree,
#         "Train MSE:", train_error,
#         "Validation MSE:", val_error
#     )

# # Plot training and validation errors
# plt.plot(degrees, train_errors, marker="o", label="Training Error")
# plt.plot(degrees, val_errors, marker="o", label="Validation Error")

# plt.xlabel("Polynomial Degree")
# plt.ylabel("MSE")
# plt.title("Bias-Variance Tradeoff")
# plt.legend()
# plt.show()