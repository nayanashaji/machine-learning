'''
Implement and compare hierarchical (agglomerative) and partitional (K-means) clustering 
algorithms on the Mall Customers dataset. Discuss the strengths and weaknesses of each 
method based on clustering results and evaluation metrics. 
Tasks: 
● Load and preprocess the Mall Customers dataset. 
● Apply both hierarchical (agglomerative) and K-means clustering. 
● Compare results using metrics such as inertia, silhouette score, and clustering visualization. 
● Discuss the advantages and disadvantages of each clustering method.

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

data=load_wine()
df = pd.DataFrame(data.data,columns=data.feature_names)

X = data.data[:,:2]
y=data.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

model = KMeans(n_clusters=5)
k_pred=model.fit_predict(X)

print("K-Means")
print("Inertia:", model.inertia_)
print("Silhouette Score:", silhouette_score(X, k_pred))

model= AgglomerativeClustering(n_clusters=5)
a_pred=model.fit_predict(X)

inertia=0

for cluster in np.unique(a_pred):
    points=X[a_pred==cluster]
    center=points.mean(axis=0)
    inertia+=np.sum((points-center)**2)

print("\nAgglomerative Clustering")
print("Inertia:", inertia)
print("Silhouette Score:", silhouette_score(X, a_pred))

plt.xlabel("Alcohol")
plt.ylabel("Malic Acid")
plt.scatter(X[:,0],X[:,1],c=k_pred)
plt.title("KMeans")
plt.show()

plt.xlabel("Alcohol")
plt.ylabel("Malic Acid")
plt.scatter(X[:,0],X[:,1],c=a_pred)
plt.title("Agglomerative")
plt.show()