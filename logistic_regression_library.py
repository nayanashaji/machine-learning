import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

data=load_iris()

df=pd.DataFrame(data.data,columns=data.feature_names)

print(df.head())

df["op"]=data.target

X=df[["sepal length (cm)"]]
y=df["op"]

X_train, X_test, y_train, y_test=train_test_split(
    X,y,
    test_size=0.2,
    random_state=42
)

scaler=StandardScaler()

X_train_s=scaler.fit_transform(X_train)
X_test_s=scaler.transform(X_test)

model_w=LogisticRegression()
model_w.fit(X_train,y_train)
model_w_pred=model_w.predict(X_test)

model_s=LogisticRegression()
model_s.fit(X_train_s,y_train)
model_s_pred=model_s.predict(X_test_s)

print("Without Scaling\nAccuracy: ",accuracy_score(y_test,model_w_pred),"Precision: ",precision_score(y_test,model_w_pred,average="weighted"),"F1 score: ",f1_score(y_test,model_w_pred,average="weighted"),"Recall score: ",recall_score(y_test,model_w_pred,average="weighted"))

print("With Scaling\nAccuracy: ",accuracy_score(y_test,model_s_pred),"Precision: ",precision_score(y_test,model_s_pred,average="weighted"),"F1 score: ",f1_score(y_test,model_s_pred,average="weighted"),"Recall score: ",recall_score(y_test,model_s_pred,average="weighted"))