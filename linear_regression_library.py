'''
Implement linear regression with one variable on the California Housing dataset to predict 
housing prices based on a single feature (e.g., the average number of rooms per dwelling). 
Tasks:  
● Load and preprocess the datase. 
● Implement linear regression using both gradient descent and the normal equation. 
● Evaluate the model performance using metrics such as Mean Squared Error 
(MSE) and R-squared. 
● Visualize the fitted line along with the data points. 
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data=fetch_california_housing()

df=pd.DataFrame(data.data,columns=data.feature_names)

print(df.head())

df["Prices"]=data.target

X=df[["AveRooms"]]
y=df[["Prices"]]

X_train, X_test, y_train, y_test=train_test_split(
    X,y,
    test_size=0.2,
    random_state=42
)

scaler=StandardScaler()

X_train_s=scaler.fit_transform(X_train)
X_test_s=scaler.transform(X_test)

gd=SGDRegressor(max_iter=5000)

gd.fit(X_train_s,y_train)

gd_pred=gd.predict(X_test_s)

lr=LinearRegression()

lr.fit(X_train_s,y_train)

lr_pred=lr.predict(X_test_s)

print("\nGradient Descent\nMSE: ",mean_squared_error(y_test,gd_pred),"\nR2 score: ",r2_score(y_test,gd_pred))

print("\nNormal Equation\nMSE: ",mean_squared_error(y_test,lr_pred),"\nR2 score: ",r2_score(y_test,lr_pred))

plt.scatter(X_test_s,y_test,label="Data")

order=np.argsort(X_test_s.ravel())

plt.plot(
    X_test_s.ravel()[order],
    gd_pred[order],
    label="Gradient Descent"
)

plt.plot(
    X_test_s.ravel()[order],
    lr_pred[order],
    label="Normal Equation"
)

plt.xlabel("Average rooms")
plt.ylabel("House Price")
plt.title("Linear Regression")
plt.legend()
plt.show()