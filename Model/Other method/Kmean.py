import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  # We take only the first two features for simplicity.

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=1)
kmeans.fit(X)

# Predict the cluster labels
y_kmeans = kmeans.predict(X)

# Plotting the results
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

# Plotting the cluster centers
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X', label='Centroids')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('K-Means Clustering on Iris Dataset')
plt.legend()
plt.show()
