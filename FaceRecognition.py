from datasplit import DatasetClass
from imagetomatrix import ImageToMatrixClass
from pcaalgorithm import PCA
from sklearn.decomposition import PCA as pca
# reco_type = "image"
import numpy as np
# required images for training so it should be 80% of 64
# integer of 0,8 * 5 = 64 is 51
no_of_images_of_one_person = 51

yaleB = DatasetClass(no_of_images_of_one_person)

training_paths = yaleB.training_paths
training_labels = yaleB.training_labels
no_images_training = yaleB.no_images_training

testing_paths = yaleB.testing_paths
testing_labels = yaleB.testing_labels
no_images_testing = yaleB.no_images_testing

# if want the targets as a column vector, make them as a column vector in the datasplit
image_targets = np.asmatrix(yaleB.target_labels_training).T

image_width, image_height = 168, 192
training_set = ImageToMatrixClass(training_paths, image_width, image_height)
image_matrix = training_set.get_matrix()

pca_images = PCA(image_matrix, 90)
print("this is the normalized matrix", pca_images.norm_matrix.shape)
print("this is the covariance matrix", pca_images.cov_matrix.shape)
p_values = pca_images.n_components()
print(p_values)

tr_matrix = pca_images.fit_transform()
print(tr_matrix.shape)

obj = pca(0.9)
PC = obj.fit_transform(image_matrix)
print(PC.shape)


