'''
Implement polynomial regression on the Auto MPG dataset to predict miles per gallon 
(MPG) based on engine displacement. Compare polynomial regression results with linear 
regression. 
Tasks: 
● Load and preprocess the dataset. 
● Implement polynomial regression of varying degrees. 
● Compare the polynomial regression models with linear regression using metrics 
such as MSE and R-squared. 
● Visualize the polynomial fit. 
'''

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

data=load_diabetes()
X=data.data
y=data.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

mse=[]
r2=[]
degrees=range(1,11)

for d in degrees:
    poly=PolynomialFeatures(degree=d)
    X_train_p=poly.fit_transform(X_train)
    X_test_p=poly.transform(X_test)
    model=LinearRegression()
    model.fit(X_train_p,y_train)
    test_pred=model.predict(X_test_p)
    mse.append(mean_squared_error(y_test,test_pred))
    r2.append(r2_score(y_test,test_pred))

for degree, ms, r in zip(degrees, mse,r2):
    print(
        "Degree:", degree,
        "MSE:", ms,
        "R2 score:",r
    )

plt.subplot(1,2,1)
plt.plot(degrees,mse,label="mse")
plt.title("MSE")
plt.xlabel("Polynomial Degree")
plt.ylabel("MSE")
plt.legend()
plt.grid()

plt.subplot(1,2,2)
plt.plot(degrees,r2,label="r2")
plt.title("R2 score")
plt.xlabel("Polynomial Degree")
plt.ylabel("R2 score")
plt.legend()
plt.grid()
plt.show()