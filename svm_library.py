'''
Implement a Linear Support Vector Machine (SVM) to classify the Iris dataset. Visualize 
the decision boundary and discuss how the margin is determined. 
Tasks: 
● Load and preprocess the Iris dataset. 
● Implement a Linear SVM for binary classification (e.g., classify Setosa vs. Non
Setosa). 
● Visualize the decision boundary and margin. 
● Discuss the concept of the margin and how it influences classification.
'''

import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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

model=SVC(kernel="linear")
model.fit(X_train_s,y_train)
pred=model.predict(X_test_s)

print("Accuracy Score: ",accuracy_score(y_test,pred))