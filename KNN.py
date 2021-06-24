
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 21, stratify = y)

knn = KNeighborsClassifier(n_neighbors=5)

# Fitting the classifier to the data
knn.fit(X_train, y_train)

# Predicting the labels for the training data X: y_pred
y_pred = knn.predict(X_test)

new_prediction = knn.predict(X_new)
print(\"Prediction:\\n {}\".format(y_pred))

#Evaluation
knn.score(X_test, y_test)
