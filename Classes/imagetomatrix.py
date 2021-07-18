import pathlib as pl
import numpy as np
from skimage import io
# if we want to resize we have this
from skimage.transform import resize


class ImageToMatrix:
    """
            Loads the file paths for each image of the dataset and puts them in a numpy array eg. image_matrix.

            Labels of images are also loaded \n (they correspond to the folder name of each image)

             data: a string with the folder name

             image_height: desired height of the image

             image_width:  desired width of the image
            """
    def __init__(self, data, file_extension=".pgm", image_height=192, image_width=168):

        data_folder = str("./"+data)
        to_glob = str("*" + str(file_extension))
        self.data = pl.Path(data_folder)
        self.image_width = image_width
        self.image_height = image_height
        self.image_size = image_width * image_height

        self.titles = []
        self.paths = []
        self.labels = []
        self.no_images_per_person = []

        person_number = 0
        for person_name in self.data.glob("*"):
            # For visualisation
            titles = str(person_name)
            titles = titles.replace((str(data)), "")
            self.titles += [str(titles)]
            img_number = 0
            # Check if it is an image
            for img_name in person_name.glob(to_glob):
                # Skip the ambient files
                if str(img_name).find('Ambient') != -1:
                    continue
                label = str(img_name.parent)
                label = label.replace((str(self.data)), "")

                self.paths += [str(img_name)]
                self.labels += [str(label)]

                if len(self.no_images_per_person) > person_number:
                    self.no_images_per_person[person_number] += 1
                else:
                    self.no_images_per_person += [1]
                # increase after every image
            img_number += 1
            # increases after every folder
            person_number += 1
        self.paths = np.asarray(self.paths)
        self.labels = np.asarray(self.labels)

    def matrix(self):
        """
        Turns image paths into an image matrix

        :return: image matrix with all the images
        """

        # rows are people and columns are pixels
        # row size is the number of images
        row = len(self.paths)
        # column size is the size of the image in pixels
        col = self.image_size
        # creating an empty matrix
        image_matrix = np.zeros((row, col))

        img_number = 0
        for image in self.paths:
            # remove the colour channels
            gray = io.imread(image, as_gray=True)
            # as they are already resized we try to resize them
            try:
                new_size = (self.image_height, self.image_width)
                # gray_resized has elements as floats, gray has them as integers
                gray_resized = resize(gray, new_size, preserve_range=True)
            except:
                break
            # ravel puts them into vectors
            vec = gray_resized.ravel()
            # fill the first row with the vector of the pixels
            # stack them vertically so that each new row is a new image
            image_matrix[img_number, :] = vec
            # the rows are people and the columns are pixels
            # an image is a row vector
            img_number += 1
        print("The dataset is loaded successfully into an image_matrix", image_matrix.shape)
        return image_matrix
