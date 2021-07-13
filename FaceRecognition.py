from Classes import imagetomatrix as img
from Classes import datasplit as ds
from Classes import pcaalgorithm as pca
from Classes import knearestneighbors as knn

"""
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.metrics import accuracy_score
"""

# Get the paths and labels of the data
train_percent = 80

yaleB = ds.MetaData(train_percent)

# Load the the paths as image matrices
image_width, image_height = 168, 192

training_set = img.ImageToMatrixClass(yaleB.training_paths, image_width, image_height)
testing_set = img.ImageToMatrixClass(yaleB.testing_paths, image_width, image_height)
train_matrix = training_set.matrix
test_matrix = testing_set.matrix

"""
# Probe with the ready-made functions
pca = PCA(n_components=0.9)
train = pca.fit_transform(train_matrix)
test = pca.transform(test_matrix)

print(train.shape, test.shape)

KNN = knn(6)
train_knn = KNN.fit(train,  yaleB.training_labels)
predicted_labels = KNN.predict(test)
score = accuracy_score(yaleB.testing_labels, predicted_labels)
print(score, "From the KNN function")
"""

# Do PCA dimension reduction
pca_images = pca.PCA(train_matrix, 99)
pca_model = pca_images.fit_svd()

train_tr = pca_images.transform(train_matrix)
test_tr = pca_images.transform(test_matrix)
print(train_tr.shape, test_tr.shape)

# Load KNN
train_knn = knn.KNearestNeighbors(train_tr, yaleB.training_labels, k=8)
predicted_labels = train_knn.predict(test_tr)


# Check prediction
correct = 0
for i in range(predicted_labels.shape[0]):
    if predicted_labels[i] == yaleB.testing_labels[i]:
        correct += 1
    else:
        continue

print("That is the accuracy percent", (correct/predicted_labels.shape[0])*100)
