import numpy as np


# trainimages_pca: transformed training images using pca
# new_image: tasted data image, array, after pca

def euclidean_distance(trainimages_pca, new_image):
    distance = np.sqrt(np.sum(np.square(trainimages_pca - new_image)))
    return distance

# trainlabels = string with the names of pictures, vector
# k = number of neighbors


def Knn(trainimages_pca, trainlabels, new_image, k):
    for train_image in trainimages_pca:
        dist = euclidean_distance(train_image, new_image)
        neighbors = np.sort(dist)[:k]  # sorting the array, first element is the smallest
    return neighbors
