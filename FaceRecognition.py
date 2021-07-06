from Classes import imagetomatrix as img
from Classes import datasplit as ds
from Classes import pcaalgorithm as pca
from Classes import knn

# Get the paths and labels of the data
train_percent = 80

yaleB = ds.MetaData(train_percent)

training_paths = yaleB.training_paths
training_labels = yaleB.training_labels

testing_paths = yaleB.testing_paths
testing_labels = yaleB.testing_labels

# Load the the paths as image matrices
image_width, image_height = 168, 192

training_set = img.ImageToMatrixClass(training_paths, image_width, image_height)
train_matrix = training_set.get_matrix()

testing_set = img.ImageToMatrixClass(testing_paths, image_width, image_height)
test_matrix = testing_set.get_matrix()

# Do PCA dimension reduction
pca_images = pca.PCA(train_matrix, 90)

components = pca_images.fit_svd()
train_tr = pca_images.transform(train_matrix)
test_tr = pca_images.transform(test_matrix)

print(train_tr.shape)
print(test_tr.shape)

# Load KNN


