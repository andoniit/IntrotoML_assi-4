import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load the datasets
dataset_1 = pd.read_csv('lab04_dataset_1.csv')
dataset_2 = pd.read_csv('lab04_dataset_2.csv')

# 1. KMeans clustering for dataset 1
silhouette_scores = []
for k in range(2, 7):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(dataset_1)
    silhouette_scores.append(silhouette_score(dataset_1, labels))

    plt.figure(figsize=(8, 6))
    plt.scatter(dataset_1['x1'], dataset_1['x2'], c=labels, cmap='viridis', marker='o', edgecolor='k')
    plt.title(f'KMeans Clustering with K = {k}')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

# Plot silhouette scores
plt.figure(figsize=(8, 6))
plt.plot(range(2, 7), silhouette_scores, marker='o')
plt.title('Silhouette Score vs. Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()

# 2. KMeans clustering for dataset 2
for k in range(2, 5):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(dataset_2)

    plt.figure(figsize=(8, 6))
    plt.scatter(dataset_2['x1'], dataset_2['x2'], c=labels, cmap='viridis', marker='o', edgecolor='k')
    plt.title(f'KMeans Clustering with K = {k}')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

# 3. SpectralClustering for dataset 2
for k in range(2, 5):
    spectral = SpectralClustering(n_clusters=k, random_state=42, affinity='nearest_neighbors')
    labels = spectral.fit_predict(dataset_2)

    plt.figure(figsize=(8, 6))
    plt.scatter(dataset_2['x1'], dataset_2['x2'], c=labels, cmap='viridis', marker='o', edgecolor='k')
    plt.title(f'Spectral Clustering with K = {k}')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

# 4. DBSCAN for dataset 2
dbscan = DBSCAN(eps=0.15, min_samples=10)
labels = dbscan.fit_predict(dataset_2)

plt.figure(figsize=(8, 6))
plt.scatter(dataset_2['x1'], dataset_2['x2'], c=labels, cmap='viridis', marker='o', edgecolor='k')
plt.title('DBSCAN Clustering')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

# Count the number of clusters (excluding noise points)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f'Number of clusters: {n_clusters}')
