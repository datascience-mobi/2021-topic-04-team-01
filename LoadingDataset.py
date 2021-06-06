from skimage import io
import os
import numpy as np

# Access files from the Training_set
training_dir = "Dataset/Training_set/"

for path, dirs, files in os.walk(training_dir):
    i = 0
    for f in files:
        if i == 0:
            filename = os.path.join(training_dir,f)
            # Open image0 and turn it into an array
            faces = io.imread(filename,as_gray=True)
            copy_faces = faces
        else:
            filename = os.path.join(training_dir, f)
            # Open any other image different than image0
            new_face = io.imread(filename, as_gray=True)
            print(new_face)
            copy_new_face = new_face
        # This part is used so that we can display the images
        # The copy is made so that we can use the faces variable for further processing
            if i % 51 == 0:
                # For every participant take their 1. picture and stack it
                # horizontally to the 1. picture of the previous participant
                array = np.hstack((copy_faces, copy_new_face))
                copy_faces = array
        i += 1


io.imshow(copy_faces)
io.show()


# Show the type and shape of the resulting object
print(type(faces))
print(faces.shape)

# Calculating the mean face
mean_face = np.mean(faces, axis=1)
print(mean_face)













