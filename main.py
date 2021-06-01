import PIL
from PIL import Image
import os
import numpy

for path, dirs, files in os.walk("C:\\Users\\Gabi\\PyCharm Project\\2021-topic-04-team-01\\Dataset\\Training_set"):
    for f in files:
        fileName = os.path.join(path,f)
        with open(fileName, "r") as myFile:
            an_image = PIL.Image.open(fileName)
            image_sequence = an_image.getdata()
            image_array = numpy.array(image_sequence)
            print (image_array)


