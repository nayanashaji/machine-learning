'''
Implement Ridge and Lasso regression on the Diabetes dataset. Compare the performance 
of these regularized models with standard linear regression. 
Tasks: 
 
 
● Load and preprocess the dataset. 
● Implement Ridge and Lasso regression. 
● Tune hyperparameters using cross-validation. 
● Compare performance metrics (MSE, R-squared) with standard linear regression. 

'''

import matplotlib.pyplot as plt

from sklearn.datasets  import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import Ridge,Lasso
from sklearn.metrics import mean_squared_error,r2_score

data=load_diabetes()
X=data.data
y=data.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

model=GridSearchCV(Ridge(),{"alpha":[0.1,0.01]})
model.fit(X_train,y_train)
r_pred=model.predict(X_test)

print("\nRidge Regression\nMSE: ",mean_squared_error(y_test,r_pred),"\nR2 score: ",r2_score(y_test,r_pred))

model=GridSearchCV(Lasso(),{"alpha":[0.1,0.01]})
model.fit(X_train,y_train)
l_pred=model.predict(X_test)

print("\nLasso Regression\nMSE: ",mean_squared_error(y_test,l_pred),"\nR2 score: ",r2_score(y_test,l_pred))

# from sklearn.datasets import load_diabetes
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import LinearRegression, Ridge, Lasso
# from sklearn.metrics import mean_squared_error, r2_score

# # Load dataset
# data = load_diabetes()
# X, y = data.data, data.target

# # Split dataset
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # Linear Regression
# models = {
#     "Linear Regression": LinearRegression(),
#     "Ridge Regression": GridSearchCV(
#         Pipeline([
#             ("scaler", StandardScaler()),
#             ("model", Ridge())
#         ]),
#         {"model__alpha": [0.01, 0.1, 1, 10, 100]},
#         cv=5,
#         scoring="neg_mean_squared_error"
#     ),
#     "Lasso Regression": GridSearchCV(
#         Pipeline([
#             ("scaler", StandardScaler()),
#             ("model", Lasso(max_iter=10000))
#         ]),
#         {"model__alpha": [0.01, 0.1, 1, 10, 100]},
#         cv=5,
#         scoring="neg_mean_squared_error"
#     )
# }

# # Train and evaluate
# for name, model in models.items():
#     if name == "Linear Regression":
#         model.fit(X_train, y_train)
#     else:
#         model.fit(X_train, y_train)
#         print(name, "Best parameters:", model.best_params_)

#     predictions = model.predict(X_test)

#     print("\n", name)
#     print("MSE:", mean_squared_error(y_test, predictions))
#     print("R2 Score:", r2_score(y_test, predictions))