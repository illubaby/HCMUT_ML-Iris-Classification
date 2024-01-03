# Data Preprocessing
# X = dataset.iloc[:, :-1].values
# y = dataset.iloc[:, -1].values

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Model Selection
# model = KNeighborsClassifier(n_neighbors=5)

# # Model Training and Evaluation
# model.fit(X_train, y_train)
# predictions = model.predict(X_test)

# print(confusion_matrix(y_test, predictions))
# print(classification_report(y_test, predictions))
