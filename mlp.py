"""
Implement and train a Multilayer Feed-Forward Network (MLP) on the Wine Quality
dataset. Experiment with different numbers of hidden layers and neurons, and discuss how
these choices affect the network’s performance.
Tasks:
● Load and preprocess the Wine Quality dataset.
● Design and implement an MLP with varying architectures (different hidden layers
and neurons).
● Train and evaluate the network.
● Discuss the impact of architecture choices on performance.

"""
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

data=load_wine()
df=pd.DataFrame(data.data,columns=data.feature_names)
print(df.head())

X=data.data
y=data.target

X_train,X_test,y_train,y_test=train_test_split(
    X,y,
    test_size=0.2
)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

architectures=[
    (10,),
    (20,),
    (10,20),
    (30,20,10)
]

for architecture in architectures:
    model=MLPClassifier(hidden_layer_sizes=architecture)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    accuracy=accuracy_score(y_test,y_pred)
    print("\nArchitecture: ",architecture)
    print("\nAccuracy score: ",accuracy)