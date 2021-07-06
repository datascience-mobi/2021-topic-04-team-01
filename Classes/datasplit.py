import pathlib as pl
import numpy as np

class MetaData:
    """
    Loads the file paths for each image of the dataset
    and splits them into testing and training. Labels of the training
    and testing set are also loaded \n (they correspond to the folder name of each image)
    \n The train_percent determines how much of the images go into training
    """
    # variable number of required images for training
    # this class has the attributes/ modules
    # image path (name of the folder + the name of the image)
    # labels (person number, corresponding to the folder number)
    # number of images (how many pictures per person go into the group)

    def __init__(self, train_percent):

        self.data = pl.Path("./CroppedYale")
        self.train_percent = train_percent/100

        self.training_paths = []
        self.training_labels = []
        self.no_images_training = []

        self.testing_paths = []
        self.testing_labels = []
        self.no_images_testing = []

        self.target_labels_training = []

        def size(data):
            """
            Gives how much images should each person have in training.
            """
            folder = data.joinpath("./yaleB01")
            pgm_files = folder.glob("*.pgm")

            image_number = 0
            for image in pgm_files:
                # Skip the ambient files
                if str(image).find('Ambient') != -1:
                    continue
                image_number += 1
            required_no = int(self.train_percent * image_number)

            return required_no

        person_number = 0
        # go to each dir in CroppedYale
        for person_name in self.data.glob("*"):
            img_number = 0
            # Check if it is an image
            for img_name in person_name.glob("*.pgm"):
                # Skip the ambient files
                if str(img_name).find('Ambient') != -1:
                    continue
                label = str(img_name.parent)
                label = label.replace((str(self.data)), "")

                # If the number is under the required for training put in training
                if img_number < size(self.data):
                    self.training_paths += [str(img_name)]
                    self.training_labels += [str(label)]
                    if len(self.no_images_training) > person_number:
                        self.no_images_training[person_number] += 1
                    else:
                        self.no_images_training += [1]
                else:
                    # when the required number of images for training has been reached
                    self.testing_paths += [str(img_name)]
                    self.testing_labels += [str(label)]
                    if len(self.no_images_testing) > person_number:
                        self.no_images_testing[person_number] += 1
                    else:
                        self.no_images_testing += [1]
                # increase after every image
                img_number += 1
            # increases after every folder
            person_number += 1

        self.training_paths = np.asarray(self.training_paths)
        self.training_labels = np.asarray(self.training_labels)

        self.testing_paths = np.asarray(self.testing_paths)
        self.testing_labels = np.asarray(self.testing_paths)

        print("Training image_paths are loaded successfully", self.no_images_training)
        print("Testing image_paths are loaded successfully", self.no_images_testing)