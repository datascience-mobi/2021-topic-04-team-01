import numpy as np
from collections import Counter

from numpy import linalg
# trainimages_pca: transformed training images using pca
# new_image: tasted data image, array, after pca


def euclidean_distance(trainimages_pca, test_image):
    distance = np.linalg.norm(trainimages_pca-test_image, axis=0)
    # distance = np.sqrt(np.sum(np.square(trainimages_pca - test_image)))
    return distance


class KNearestNeighbors:

    def __init__(self, Trainimages_pca, trainlabels, k=6):
        self.k = k
        self.Trainimages_pca = Trainimages_pca
        self.trainlabels = trainlabels

    def predict(self, new_image):

        # predict the labels of each images from the testing set
        predicted_labels = [self._predict(new_image)]
        return np.array(predicted_labels)

    def _predict(self, new_image):

        # compute the distances between the new image and all other images in the training set, so trainimages_pca
        distances = [euclidean_distance(trainimages_pca, new_image) for trainimages_pca in self.Trainimages_pca]

        # get the k nearest neighbors, the smallest distances, get the nearest labels
        k_indices = np.argsort(distances)[:self.k]   # sort the distances and find the 5 nearest pictures from the training set
        k_nearest_labels = [self.trainlabels[i] for i in k_indices]  # indices correspond to the labels, but the labels are numbers not names
        # print("these are the indices", k_indices)
        # print("these are the nearest labels", k_nearest_labels)
        # we want to get the most common class label, majority vote
        most_common = Counter(k_nearest_labels).most_common(1)  # we want to have only one most common item
        print("this is the most common class label, name of the person", most_common)
        return most_common[0][0]  # to get the tuple and the first item of it
