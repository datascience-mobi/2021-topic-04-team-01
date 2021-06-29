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

'''
# img_width, img_height = 50, 50
# ImageToMatrixClassObj = ImageToMatrixClass(images_path_for_training, img_width, img_height)
# img_matrix = ImageToMatrixClassObj.get_matrix()

# PCA_class_obj = PCA(img_matrix, labels_for_training, image_targets,
# no_of_elements_for_training, img_width, img_height, quality_percent=90)
# new_coordinates = PCA_class_obj.reduce_dim()

# Recognizing

 if reco_type == "image":
    # correct = 0
    # wrong = 0
    # i = 0
    # for img_path in images_path_for_testing:
      #  img = PCA_class_obj.img_from_path(img_path)
      #  PCA_class_obj.show_image("Recognize Image", img)
      # new_cords_for_image = PCA_class_obj.new_coords(img)

      #  found_name = PCA_class_obj.recognize_face(new_cords_for_image)
      #  target_index = labels_for_testing[i]
        # original_name = image_targets[target_index]

        # if found_name is original_name:
          #  correct += 1
           # print("Correctly matched", "Label: ", found_name)
        # else:
          #  wrong += 1
          # print("Wrongly matched", "Label: ", original_name)
        # i += 1
    # print("Total correct: ", correct)
    # print("Toatal wrong: ", wrong)
    # print("Accuracy: ", correct / (correct + wrong) * 100)
'''