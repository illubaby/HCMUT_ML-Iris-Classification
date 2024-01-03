from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Encoding species names into numerical values
encoder = LabelEncoder()

iris_data['species'] = encoder.fit_transform(iris_data['species'])

# Splitting the dataset into training and testing sets
X = iris_data.drop('species', axis=1)
y = iris_data['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Now, X_train and y_train can be used for training the model,
# and X_test and y_test for testing it.
