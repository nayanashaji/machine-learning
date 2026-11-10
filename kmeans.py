'''
Implement and apply K-means clustering to the Digits dataset. Experiment with different 
numbers of clusters and evaluate the clustering results using metrics such as inertia and 
silhouette score. Analyze how the choice of K affects clustering performance. 
Tasks: 
● Load and preprocess the Digits dataset. 
● Implement K-means clustering with various numbers of clusters. 
● Evaluate clustering performance using inertia and silhouette score. 
● Analyze the impact of the number of clusters on clustering quality.

'''

import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

digits = load_digits()
X = digits.data

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k_values = [2, 5, 8, 10, 12]

inertias = []
silhouette_scores = []

for k in k_values:
    model = KMeans(n_clusters=k)

    pred = model.fit_predict(X_scaled)

    inertia = model.inertia_
    silhouette = silhouette_score(X_scaled, pred)

    inertias.append(inertia)
    silhouette_scores.append(silhouette)

    print("\nNumber of clusters:", k)
    print("Inertia:", inertia)
    print("Silhouette Score:", silhouette)

plt.plot(k_values, inertias, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Inertia vs K")
plt.grid(True)
plt.show()

plt.plot(k_values, silhouette_scores, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs K")
plt.grid(True)
plt.show()