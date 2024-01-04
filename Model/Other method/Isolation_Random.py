from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from sklearn import datasets

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  # We take only the first two features for simplicity.

# Apply Isolation Forest for anomaly detection
iso_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=1)
iso_forest.fit(X)

# Predict if a data point is an anomaly (-1 for anomalies, 1 for normal points)
is_anomaly = iso_forest.predict(X)

# Plotting the results
plt.figure(figsize=(8, 6))
# Normal points
plt.scatter(X[is_anomaly == 1, 0], X[is_anomaly == 1, 1], c='blue', label='Normal Data')
# Anomalies
plt.scatter(X[is_anomaly == -1, 0], X[is_anomaly == -1, 1], c='red', label='Anomalies')

plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Anomaly Detection on Iris Dataset')
plt.legend()
plt.show()
