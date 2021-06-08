import numpy as np
from pathlib import Path
from skimage import io
import os

# get paths of each file in folder named Training_set
training_dir = Path("Dataset/Training_set/")

# Loading the dataset into an array for further processing
def load(directory):
    for training_dir, dirs, files in os.walk(directory):
        for f in files:
            filename = os.path.join(training_dir,f)
            image_array = io.imread(filename, as_gray=True)
            # as_gray turns the images into grayscale
            # although they were grayscale already the im.read function reads them as RGB
            # (3 channels per pixel)
    print("Dataset loaded successfully into image_array")
    return

def visualize(directory):
    for training_dir, dirs, files in os.walk(directory):
    # starting with -1 so when we add the first picture it has the value 0
    # i represents an index for every picture
        i = -1
    # 38 people with 52 pictures these numbers can be represented with a parameter
    # that way we can generalize for other datasets
        while i < (38*52):
            for f in files:
                i += 1
            # used to define the array people so we can use it in the function
                if i == 0:
                    filename = os.path.join(training_dir, f)
                    people = io.imread(filename, as_gray=True)
            # Every subject has a number of pictures in the training_set for us that is 51
            # We are taking only the first pictures of a person to visualize
            # i+51 is used so that we don't take the 0th person
                if (i+51) % 51 == 0:
                # The idea here is to make an array which has two columns of 19 picture
                    if i % 19 != 0:
                        filename = os.path.join(training_dir, f)
                        next_person = io.imread(filename, as_gray=True)
                        vertical_array = np.vstack((people, next_person))
                        people = vertical_array
                # else:
                #   filename = os.path.join(training_dir, f)
                #   person2 = io.imread(filename, as_gray=True)
                #   image_array = np.hstack((person0, person2))
                #   person0 = image_array
                #   We need a function to tell the computer to move to the 0 picture
                #   when it needs to concatenate horizontally

            # assigns a number to each filename
    io.imshow(people)
    io.show()
    return

def meanface(array_name):
    meanf = np.mean(array_name, axis=1)
    print('The mean face of the dataset is', meanf)
    return

load(training_dir)
meanface(image_array)
