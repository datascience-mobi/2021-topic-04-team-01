import os


class DatasetClass:
    # variable number of required images for training
    # this class has the attributes/ modules
    # image path (name of the folder + the name of teh image)
    # labels (person number, corresponding to the folder number)
    # number of images (how many pictures per person go into the group)

    def __init__(self, required_no):
        dir = "../CroppedYale/"

        self.training_paths = []
        self.training_labels = []
        self.no_images_training = []

        self.testing_paths = []
        self.testing_labels = []
        self.no_images_testing = []

        self.target_labels_training = []

        # the variable is used for labeling
        person_number = 0
        # go to each dir in CroppedYale
        for person_name in os.listdir(dir):
            dir_path = os.path.join(dir, person_name)
            # check if it is a directory
            if os.path.isdir(dir_path):
                # check if the directory has enough images for training
                if len(os.listdir(dir_path)) >= required_no:
                    # image count is img_number = -2 because the first 2 files are not .pgm
                    img_number = -2
                    # inside a folder of a person, for example 'yaleB01'
                    for img_name in os.listdir(dir_path):

                        if img_name.endswith('.pgm'):
                            # check if it is an image
                            # skips files that are not .pgm
                            # add the image_name to the dir with person_name
                            img_path = os.path.join(dir_path, img_name)
                            # skip the ambient files
                            # the function gives you an error for -1
                            # the ambient files skip this image
                            if img_name.find('Ambient') != -1:
                                continue

                            if img_number < required_no:
                                self.training_paths += [img_path]
                                self.training_labels += [person_number]
                                # the number of images already in the array is less the required number
                                # for raining increase the images in the array by one
                                if len(self.no_images_training) > person_number:
                                    # we are increase the number of pictures of the person
                                    # it starts at 0 than it increases by 1 until it reaches person_number
                                    # for person 2 the value of no_of_images_for_training will be 2
                                    self.no_images_training[person_number] += 1
                                else:
                                    # start counting when not present
                                    self.no_images_training += [1]

                                # if the loop has started
                                if img_number == 0:
                                    self.target_labels_training += [person_name]
                                    # this array will contain the dir names yaleB01 and thus forth

                            # when the required number of images for training has been reached
                            # the other images go to testing
                            else:
                                self.testing_paths += [img_path]
                                self.testing_labels += [person_number]

                                if len(self.no_images_testing) > person_number:
                                    self.no_images_testing[person_number] += 1
                                else:
                                    self.no_images_testing += [1]
                        # increase after every image
                        img_number += 1
                    # after every folder increases
                    person_number += 1
        print("Training image_paths are loaded successfully", self.no_images_training)
        print("Testing image_paths are loaded successfully", self.no_images_testing)