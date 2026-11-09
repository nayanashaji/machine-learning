'''
Implement and compare the performance of SVM classifiers with linear, polynomial, and 
RBF kernels on the Fashion MNIST dataset. Analyze the advantages and disadvantages of 
each kernel type. 
 
 
Tasks: 
● Load and preprocess the Fashion MNIST dataset. 
● Implement SVM with linear, polynomial, and RBF kernels. 
● Compare the classification performance for each kernel. 
● Discuss the strengths and weaknesses of each kernel type.
'''

import time
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. Load dataset
data = load_breast_cancer()

X = data.data
y = data.target

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Standardize features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models=["linear","poly","rbf"]

for m in models:
    model=SVC(kernel=m)
    start=time.time()
    model.fit(X_train,y_train)
    end=time.time()
    pred=model.predict(X_test)
    print("\nKernel:", m)
    print("Accuracy:", accuracy_score(y_test, pred))
    print("Training and prediction time:",end-start)


# import time
# import numpy as np
# from tensorflow.keras.datasets import fashion_mnist
# from sklearn.svm import SVC
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score

# # Load dataset
# (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# # Use a subset to reduce training time
# X_train = X_train[:10000].reshape(10000, -1)
# y_train = y_train[:10000]
# X_test = X_test[:2000].reshape(2000, -1)
# y_test = y_test[:2000]

# # Normalize pixel values
# X_train = X_train / 255.0
# X_test = X_test / 255.0

# # Standardize features
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Compare kernels
# kernels = ["linear", "poly", "rbf"]

# for kernel in kernels:
#     model = SVC(kernel=kernel, degree=3)

#     start = time.time()
#     model.fit(X_train, y_train)
#     predictions = model.predict(X_test)
#     elapsed = time.time() - start

#     print("\nKernel:", kernel)
#     print("Accuracy:", accuracy_score(y_test, predictions))
#     print("Training and prediction time:", round(elapsed, 2), "seconds")