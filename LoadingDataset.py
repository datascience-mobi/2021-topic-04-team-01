import numpy
from pathlib import Path
import cv2
import os

# get paths of each file in folder named Training_set
path = Path("/Users/gogoa/PycharmProjects/2021-topic-04-team-01/Dataset/Training_set/")

for path, dirs, files in os.walk(path):
    for f in files:
        filename = os.path.join(path,f)
        with open(filename, 'r'):
            # open every image and turn it into an array
            an_image = cv2.imread(filename)
            # ".getdata" doesn't work on an_image here
            image_array = numpy.array(an_image)

print(image_array)







