'''
Implement the K-Nearest Neighbors (KNN) algorithm for image classification using the 
Fashion MNIST dataset. Experiment with different values of K and analyze their impact 
on model performance. 
Tasks: 
● Load and preprocess the Fashion MNIST dataset. 
● Implement KNN for multi-class classification. 
● Experiment with different values of K and evaluate performance. 
● Discuss the impact of different K values on model accuracy and computational 
efficiency. 
'''

import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data=load_iris()

df=pd.DataFrame(data.data, columns=data.feature_names)

df["op"]=data.target

X=df[["sepal length (cm)"]]
y=df["op"]

scaler=StandardScaler()

X_train, X_test, y_train, y_test=train_test_split(
    X,y,
    test_size=0.2
)

X_train_s=scaler.fit_transform(X_train)
X_test_s=scaler.transform(X_test)

for k in [3,5,7]:
    model=KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_s,y_train)
    pred=model.predict(X_test_s)

    print("\nK= ",k,"\nAccuracy= ",accuracy_score(y_test,pred))