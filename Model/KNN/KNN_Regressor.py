import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)

# Choose one feature for simplicity (e.g., 'petal-length') and convert species names to numbers for regression
feature_column = 'petal-length'
X = dataset[[feature_column]].values
species_to_num = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
y = dataset['class'].map(species_to_num).values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create two KNN regressor instances: one with uniform weights, another with distance weights
knn_uniform = KNeighborsRegressor(n_neighbors=5, weights='uniform')
knn_distance = KNeighborsRegressor(n_neighbors=5, weights='distance')

# Train the models
knn_uniform.fit(X_train, y_train)
knn_distance.fit(X_train, y_train)

# Generate a range of values that covers the input feature space for the chosen feature
X_range = np.linspace(X_train.min(), X_train.max(), 500).reshape(-1, 1)

# Predict using both models
predictions_uniform = knn_uniform.predict(X_range)
predictions_distance = knn_distance.predict(X_range)
# Plot the actual data and the predictions
plt.figure(figsize=(14, 6))

# Define colors for each class
colors = {0: 'red', 1: 'green', 2: 'blue'}
class_names = {0: 'Iris-setosa',1:'Iris-versicolor',2:'Iris-virginica'}
# Plot for uniform weights
plt.subplot(1, 2, 1)
for class_val, color in colors.items():
    idx = np.where(y_train == class_val)
    plt.scatter(X_train[idx], y_train[idx], color=color, label=class_names[class_val])
plt.plot(X_range, predictions_uniform, color='black', label='prediction')
plt.title('KNeighborsRegressor (k = 5, weights = \'uniform\')')
plt.xlabel(feature_column)
plt.ylabel('Class')
plt.legend()

# Plot for distance weights
plt.subplot(1, 2, 2)
for class_val, color in colors.items():
    idx = np.where(y_train == class_val)
    plt.scatter(X_train[idx], y_train[idx], color=color, label=class_names[class_val])
plt.plot(X_range, predictions_distance, color='black', label='prediction')
plt.title('KNeighborsRegressor (k = 5, weights = \'distance\')')
plt.xlabel(feature_column)
plt.ylabel('Class')
plt.legend()

plt.tight_layout()
plt.show()
