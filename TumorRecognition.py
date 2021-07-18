from Classes import imagetomatrix as img
from Classes import datasplit as ds
from Classes import pcaalgorithm as pca
from Classes import knearestneighbors as knn
from Classes import visualising as vis

# Get the paths and labels of the data
image_height, image_width = 256, 256
Brain_tumor = img.ImageToMatrix("Brain_tumor_images", file_extension=".jpg",
                                image_height=image_height, image_width=image_width)
image_matrix = Brain_tumor.matrix()


# Displaying some of the images
vis.small_set_show(Brain_tumor.no_images_per_person, image_matrix, image_height, image_width, 2)

# Splitting the data into testing and training
train_matrix, test_matrix, training_labels, testing_labels = ds.train_test_split(image_matrix,
                                                                                 Brain_tumor.labels,
                                                                                 train_percent=80, seed=7)

# Do PCA dimension reduction
pca_images = pca.PCA(train_matrix, 99)
pca_model = pca_images.fit_svd()

# Displaying the mean face
vis.vector_show(pca_images.mean_face, image_height, image_width)
# Displaying some of components of the model
vis.show_components(pca_model, image_height, image_width)

train_tr = pca_images.transform(train_matrix)
test_tr = pca_images.transform(test_matrix)
print(train_tr.shape, test_tr.shape)

# Load KNN
train_knn = knn.KNearestNeighbors(train_tr, training_labels, k=2)
predicted_labels = train_knn.predict(test_tr)

# Visualize
vis.k_barplot(8, train_tr, training_labels, test_tr, testing_labels,)
