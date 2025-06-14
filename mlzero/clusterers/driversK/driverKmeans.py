import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from mlzero.clusterers.k_means import Kmeans

# Generate toy dataset
centroids = [(-5, -5), (5, 5), (1, 1)]
cluster_std = [1, 1, 1]

X, y = make_blobs(n_samples=300, centers=centroids, cluster_std=cluster_std, n_features=2, random_state=42)

# Initialize object of Kmeans
kmeans = Kmeans(n_clusters=3, max_iterations=100)
X_color, final_centroids = kmeans.fit_predict(X)
print(final_centroids)

# Plot points
plt.scatter(X[X_color == 0][:, 0], X[X_color == 0][:, 1], color='blue', label='Cluster 0')
plt.scatter(X[X_color == 1][:, 0], X[X_color == 1][:, 1], color='green', label='Cluster 1')
plt.scatter(X[X_color == 2][:, 0], X[X_color == 2][:, 1], color='orange', label='Cluster 2')

# Final centroids
plt.scatter(final_centroids[:, 0], final_centroids[:, 1], color='black', label='Centroids', marker='X', s=200)

plt.title("K-means Clustering (3 Clusters)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()
