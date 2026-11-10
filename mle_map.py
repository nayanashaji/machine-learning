'''
Estimate the parameters of a logistic regression model using MLE and MAP on the Breast 
Cancer Wisconsin dataset. Compare the results and discuss the effects of regularization. 
Tasks: 
● Load and preprocess the dataset. 
● Implement logistic regression with MLE. 
● Apply MAP estimation with different regularization priors (L1 and L2 
regularization). 
● Compare the performance and parameter estimates with MLE and MAP. 

'''

import matplotlib.pyplot as plt

from sklearn.datasets  import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data=load_breast_cancer()
X=data.data
y=data.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

model=LogisticRegression(penalty=None)
model.fit(X_train,y_train)
mle_pred=model.predict(X_test)

print("\nMLE\nAccuracy: ",accuracy_score(y_test,mle_pred),"\nParameter estimate:",model.coef_[0][:5])

model=LogisticRegression(penalty="l2")
model.fit(X_train,y_train)
l2_pred=model.predict(X_test)

print("\nL2\nAccuracy: ",accuracy_score(y_test,l2_pred),"\nParameter estimate:",model.coef_[0][:5])

model=LogisticRegression(penalty="l1",solver="liblinear")
model.fit(X_train,y_train)
l1_pred=model.predict(X_test)

print("\nL1\nAccuracy: ",accuracy_score(y_test,l1_pred),"\nParameter estimate:",model.coef_[0][:5])

# import pandas as pd
# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, f1_score

# # Load dataset
# data = load_breast_cancer()

# df = pd.DataFrame(data.data, columns=data.feature_names)
# df["target"] = data.target

# print(df.head())

# # Features and target
# X = df[data.feature_names]
# y = df["target"]

# # Split dataset
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # Feature scaling
# scaler = StandardScaler()

# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # ---------------- MLE ----------------
# # Very large C -> approximately no regularization

# mle = LogisticRegression(
#     C=1e10,
#     max_iter=1000
# )

# mle.fit(X_train, y_train)
# pred_mle = mle.predict(X_test)

# print("\nMLE")
# print("Accuracy:", accuracy_score(y_test, pred_mle))
# print("F1 Score:", f1_score(y_test, pred_mle))


# # ---------------- MAP with L2 ----------------
# # L2 regularization corresponds to Gaussian prior

# map_l2 = LogisticRegression(
#     penalty="l2",
#     C=1.0,
#     max_iter=1000
# )

# map_l2.fit(X_train, y_train)
# pred_l2 = map_l2.predict(X_test)

# print("\nMAP - L2 Regularization")
# print("Accuracy:", accuracy_score(y_test, pred_l2))
# print("F1 Score:", f1_score(y_test, pred_l2))


# # ---------------- MAP with L1 ----------------
# # L1 regularization corresponds to Laplace prior

# map_l1 = LogisticRegression(
#     penalty="l1",
#     solver="liblinear",
#     C=1.0,
#     max_iter=1000
# )

# map_l1.fit(X_train, y_train)
# pred_l1 = map_l1.predict(X_test)

# print("\nMAP - L1 Regularization")
# print("Accuracy:", accuracy_score(y_test, pred_l1))
# print("F1 Score:", f1_score(y_test, pred_l1))