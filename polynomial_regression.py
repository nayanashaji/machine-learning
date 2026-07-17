import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
cols=["mpg","cylinders","displacement","horsepower","weight",
      "acceleration","model_year","origin","car_name"]
df=pd.read_csv("auto-mpg.data",
               delim_whitespace=True,
               names=cols,
               na_values="?").dropna()
X=df[["displacement"]].values
y=df["mpg"].values
X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42)
linear=LinearRegression()
linear.fit(X_train,y_train)
linear_pred=linear.predict(X_test)
linear_mse=mean_squared_error(y_test,linear_pred)
linear_r2=r2_score(y_test,linear_pred)
linear_coef=linear.coef_[0]
linear_intercept=linear.intercept_
x_train=X_train.flatten()
x_test=X_test.flatten()
X2_train=np.column_stack((
    np.ones(len(x_train)),
    x_train,
    x_train**2
))
X2_test=np.column_stack((
    np.ones(len(x_test)),
    x_test,
    x_test**2
))
theta2=np.linalg.inv(
    X2_train.T@X2_train
)@X2_train.T@y_train
deg2_pred=X2_test@theta2
deg2_mse=mean_squared_error(y_test,deg2_pred)
deg2_r2=r2_score(y_test,deg2_pred)
X3_train=np.column_stack((
    np.ones(len(x_train)),
    x_train,
    x_train**2,
    x_train**3
))
X3_test=np.column_stack((
    np.ones(len(x_test)),
    x_test,
    x_test**2,
    x_test**3
))
theta3=np.linalg.inv(
    X3_train.T@X3_train
)@X3_train.T@y_train
deg3_pred=X3_test@theta3
deg3_mse=mean_squared_error(y_test,deg3_pred)
deg3_r2=r2_score(y_test,deg3_pred)
idx=np.argsort(x_test)
x=x_test[idx]
linear_curve=linear.predict(x.reshape(-1,1))
curve2=np.column_stack((
    np.ones(len(x)),
    x,
    x**2
))@theta2
curve3=np.column_stack((
    np.ones(len(x)),
    x,
    x**2,
    x**3
))@theta3
print("\n")
print("="*100)
print(f'{"Model":<20}{"Coefficient(s)":<30}{"Intercept":<15}{"MSE":<15}{"R²"}')
print("="*100)
print(f'{"Linear":<20}{str(np.round(linear_coef,4)):<30}{linear_intercept:<15.4f}{linear_mse:<15.4f}{linear_r2:.4f}')
print(f'{"Degree 2":<20}{str(np.round(theta2[1:],4)):<30}{theta2[0]:<15.4f}{deg2_mse:<15.4f}{deg2_r2:.4f}')
print(f'{"Degree 3":<20}{str(np.round(theta3[1:],4)):<30}{theta3[0]:<15.4f}{deg3_mse:<15.4f}{deg3_r2:.4f}')
fig,ax=plt.subplots(1,3,figsize=(18,5),sharex=True,sharey=True)
titles=[
    f"Linear\nMSE={linear_mse:.3f}\nR²={linear_r2:.3f}",
    f"Degree 2\nMSE={deg2_mse:.3f}\nR²={deg2_r2:.3f}",
    f"Degree 3\nMSE={deg3_mse:.3f}\nR²={deg3_r2:.3f}"]
curves=[linear_curve,curve2,curve3]
colors=["red","green","purple"]
ymin=min(y_test.min(),linear_curve.min(),curve2.min(),curve3.min())
ymax=max(y_test.max(),linear_curve.max(),curve2.max(),curve3.max())
for i in range(3):
    ax[i].scatter(X_test,y_test,color="blue",s=12)
    ax[i].plot(x,curves[i],color=colors[i],linewidth=2)
    ax[i].set_title(titles[i])
    ax[i].set_xlabel("Displacement")
    ax[i].set_ylabel("MPG")
    ax[i].set_xlim(X_test.min(),X_test.max())
    ax[i].set_ylim(ymin,ymax)
    ax[i].grid(True)
plt.tight_layout()
plt.show()