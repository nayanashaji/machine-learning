"""
Implement bagging and boosting ensemble methods on the Titanic dataset. Compare the
performance of both methods in terms of accuracy, precision, recall, and F1-score.
Discuss how each method improves model performance and their respective strengths and
weaknesses.
Tasks:
● Load and preprocess the Titanic dataset.
● Implement bagging using a base classifier (e.g., decision tree) and evaluate
performance.
● Implement boosting using a boosting algorithm (e.g., AdaBoost) and evaluate
performance.
● Compare performance metrics and discuss the strengths and weaknesses of each
method.

"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier,AdaBoostClassifier
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score

df=pd.read_csv("titanic.csv")

X=df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]
y=df[["Survived"]]

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2
)

num_col=["Pclass", "Age", "SibSp", "Parch", "Fare"]
cat_col=["Sex"]

preprocessor=ColumnTransformer(
    [("num",SimpleImputer(strategy="median"),num_col),
     ("cat",Pipeline([
         ("imputer",SimpleImputer(strategy="most_frequent")),
         ("encoder",OneHotEncoder(handle_unknown="ignore"))
     ]),cat_col)]
)

#Bagging

bagging=Pipeline([
    ("preprocessor",preprocessor),
    ("bagging",BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=50
    ))
])

bagging.fit(X_train,y_train)
bagging_pred=bagging.predict(X_test)

print("\nBagging")
print("Accuracy:", accuracy_score(y_test, bagging_pred))
print("Precision:", precision_score(y_test, bagging_pred))
print("Recall:", recall_score(y_test, bagging_pred))
print("F1 Score:", f1_score(y_test, bagging_pred))

# Boosting

boosting=Pipeline([
    ("preprocessor",preprocessor),
    ("boosting",AdaBoostClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=50
    ))
])

boosting.fit(X_train,y_train)
boosting_pred=boosting.predict(X_test)

print("\nBoosting")
print("Accuracy:", accuracy_score(y_test, boosting_pred))
print("Precision:", precision_score(y_test, boosting_pred))
print("Recall:", recall_score(y_test, boosting_pred))
print("F1 Score:", f1_score(y_test, boosting_pred))

# import pandas as pd

# from sklearn.model_selection import train_test_split
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import OneHotEncoder

# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier

# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# # Load Titanic dataset
# df = pd.read_csv("titanic.csv")

# print(df.head())

# # Select useful features
# X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]
# y = df["Survived"]

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # Numerical and categorical columns
# num_cols = ["Age", "SibSp", "Parch", "Fare", "Pclass"]
# cat_cols = ["Sex"]

# # Preprocessing
# preprocessor = ColumnTransformer([
#     ("num", SimpleImputer(strategy="median"), num_cols),
#     ("cat", Pipeline([
#         ("imputer", SimpleImputer(strategy="most_frequent")),
#         ("encoder", OneHotEncoder(handle_unknown="ignore"))
#     ]), cat_cols)
# ])

# # ---------------- BAGGING ----------------

# bagging = Pipeline([
#     ("preprocess", preprocessor),
#     ("model", BaggingClassifier(
#         estimator=DecisionTreeClassifier(random_state=42),
#         n_estimators=50,
#         random_state=42
#     ))
# ])

# bagging.fit(X_train, y_train)

# pred_bagging = bagging.predict(X_test)

# print("\nBagging")
# print("Accuracy:", accuracy_score(y_test, pred_bagging))
# print("Precision:", precision_score(y_test, pred_bagging))
# print("Recall:", recall_score(y_test, pred_bagging))
# print("F1 Score:", f1_score(y_test, pred_bagging))


# # ---------------- BOOSTING ----------------

# boosting = Pipeline([
#     ("preprocess", preprocessor),
#     ("model", AdaBoostClassifier(
#         estimator=DecisionTreeClassifier(max_depth=1),
#         n_estimators=50,
#         random_state=42
#     ))
# ])

# boosting.fit(X_train, y_train)

# pred_boosting = boosting.predict(X_test)

# print("\nBoosting")
# print("Accuracy:", accuracy_score(y_test, pred_boosting))
# print("Precision:", precision_score(y_test, pred_boosting))
# print("Recall:", recall_score(y_test, pred_boosting))
# print("F1 Score:", f1_score(y_test, pred_boosting))