from skimage import io
# if we want to resize we have this
from skimage.transform import resize
import numpy as np


class ImageToMatrixClass:
    """
    Opens image paths and converts them into a numpy array, eg. image_matrix\n
    image_paths are the path names of each image\n
    image_width is desired width of the image\n
    image_height is the desired height of the image
    """
    # image_width and image_height can be changed
    # image_paths is the same as the training_paths or testing_paths
    def __init__(self, image_paths, image_width, image_height):
        self.image_paths = image_paths
        self.images_width = image_width
        self.images_height = image_height
        self.images_size = image_width * image_height

    def get_matrix(self):
        # rows are people and columns are pixels
        # row size is the number of images
        row = len(self.image_paths)
        # column size is the size of the image in pixels
        col = self.images_size
        # creating an empty matrix
        img_mat = np.zeros((row, col))

        img_number = 0
        for image in self.image_paths:
            # remove the colour channels
            gray = io.imread(image, as_gray=True)
            # as they are already resized we try to resize them
            try:
                new_size = (self.images_height, self.images_width)
                # gray_resized has elements as floats, gray has them as integers
                gray_resized = resize(gray, new_size, preserve_range=True)
            except:
                break
            # ravel puts them into vectors
            vec = gray_resized.ravel()
            # fill the first row with the vector of the pixels
            # stack them vertically so that each new row is a new image
            img_mat[img_number, :] = vec
            # the rows are people and the columns are pixels
            # an image is a row vector
            img_number += 1
        print("The dataset is loaded successfully into an image_matrix", img_mat.shape)
        print("The number of rows is the number of images in the dataset", img_mat.shape[0])
        print("The number of columns is the number of pixels in an image", img_mat.shape[1])
        return img_mat
