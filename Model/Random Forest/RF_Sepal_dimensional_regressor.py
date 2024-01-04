import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)

# Filter the dataset for 'Iris-setosa' class
setosa_dataset = dataset[dataset['class'] == 'Iris-setosa']

# Choose 'sepal-length' as the feature
feature_column = 'sepal-length'
X = setosa_dataset[[feature_column]].values
y = setosa_dataset['sepal-width'].values  # Assuming you want to predict 'sepal-width'

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a Random Forest Regressor instance and train it
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=1)
rf_regressor.fit(X_train, y_train)

# Generate a range of values that covers the input feature space for the chosen feature
X_range = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)

# Predict using the Random Forest model
predictions_rf = rf_regressor.predict(scaler.transform(X_range))

# Plot the actual data and the predictions
plt.figure(figsize=(7, 6))
plt.scatter(X_train, y_train, color='red', label='Iris-setosa Data')
plt.plot(scaler.transform(X_range), predictions_rf, color='black', label='Random Forest Prediction')
plt.title('RandomForestRegressor for Iris-setosa (n_estimators = 100)')
plt.xlabel(feature_column)
plt.ylabel('Sepal Width')
plt.legend()
plt.show()
