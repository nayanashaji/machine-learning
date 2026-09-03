import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data=load_iris()

df=pd.DataFrame(data.data, columns=data.feature_names)

df["op"]=data.target

X=df[data.feature_names]
y=df["op"]

X_train, X_test, y_train, y_test=train_test_split(
    X,y,
    test_size=0.2
)

scaler=StandardScaler()

X_train_s=scaler.fit_transform(X_train)
X_test_s=scaler.transform(X_test)

model=DecisionTreeClassifier(
    criterion="entropy"
)
model.fit(X_train_s,y_train)
pred=model.predict(X_test_s)

importance=pd.Series(
    model.feature_importances_,
    index=X.columns,
).sort_values(ascending=False)

print("Accuracy Score: ",accuracy_score(y_test,pred))

print("\nFeature Importance: ",importance)