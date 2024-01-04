import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

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

# Generate a range of values that covers the input feature space for the chosen feature
X_range = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)  # Use the range of the entire dataset

# Create a Random Forest Regressor instance and train it
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=1)
rf_regressor.fit(X_train, y_train)

# Predict using the Random Forest model
predictions_rf = rf_regressor.predict(scaler.transform(X_range))  # Scale X_range before prediction

# Define colors for each class
colors = {0: 'red', 1: 'green', 2: 'blue'}
class_names = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}

# Plot the actual data and the predictions
plt.figure(figsize=(7, 6))  # Adjusted figure size for a single plot

# Plot for Random Forest Regressor
for class_val, color in colors.items():
    idx = np.where(y_train == class_val)
    plt.scatter(X_train[idx], y_train[idx], color=color, label=class_names[class_val])
plt.plot(scaler.transform(X_range), predictions_rf, color='black', label='Random Forest Prediction')
plt.title('RandomForestRegressor (n_estimators = 100)')
plt.xlabel(feature_column)
plt.ylabel('Class')
plt.legend()

plt.show()
