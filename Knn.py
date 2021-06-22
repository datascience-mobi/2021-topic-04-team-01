from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# X_train and X_test = the pictures in the training and testing data
# y_train and y_test = the labels of the pictures (these will be the names of the folders for each person)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 21, stratify = y)

# Create a k-NN classifier with 5 neighbors
knn = KNeighborsClassifier(n_neighbors = 5)

# Fit the classifier to the data
knn.fit(X_train, y_train)

# Predicting the labels for the training data X: y_pred
y_pred = knn.predict(X_test)



