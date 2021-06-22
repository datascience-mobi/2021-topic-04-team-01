from skimage import io
import os
import numpy as np
from sklearn.decomposition import PCA

# Access files from the Training_set
training_dir = "Dataset/Training_set/"

for path, dirs, files in os.walk(training_dir):
    image_number = 0
    for image in files:
        if image_number == 0:
            # Loads the 1. image and turns it into a vector
            filename = os.path.join(training_dir, image)
            initial_faces = io.imread(filename, as_gray=True)
            faces = initial_faces.flatten()
            print("Initial image added", faces.shape)
        else:
            # Concatenates the image vectors into an array
            filename = os.path.join(training_dir, image)
            array_face = io.imread(filename, as_gray=True)
            new_face = array_face.flatten()
            faces = np.vstack((new_face, faces))
        image_number += 1

print("The dataset is loaded successfully", faces.shape)

# The transposed array has participants as columns and pixels as rows
faces = faces.transpose()
print("The dataset has been transposed", faces.shape)

# 90% of the variance in the dataset is expressed through the principal components
pca = PCA(0.9)
converted_faces = pca.fit_transform(faces.data)

print("The dataset is reduced to", converted_faces.shape)
# The transposed matrix has people as rows and pixels as columns
pca_people = converted_faces.transpose()
print("The dataset has been transposed", pca_people.shape)