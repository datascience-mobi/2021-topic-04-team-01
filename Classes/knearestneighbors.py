import numpy as np
from collections import Counter

def euclidean_distance(train_images_pca, test_image):
    """
        Euclidean distance formula.
    """
    # distance = np.linalg.norm(train_images_pca - test_image, axis=0)
    distance = np.sqrt(np.sum(np.square(train_images_pca - test_image)))
    return distance


class KNearestNeighbors:
    """ K-Nearest Neighbors (KNN).
            Calculates the euclidean distance of each image of the testing set to each image of the training set.
            \n
            The train_images_pca are the images after dimensionality reduction with pca.
            \n
            The training_labels are the labels (folder names) of each image of the training set.
            \n
            k is the number of neighbors which we use to the determine the label of the testing image.
        """
    def __init__(self, train_images_pca, training_labels, k=6):
        self.k = k
        self.train_images_pca = train_images_pca
        self.training_labels = training_labels

    def predict(self, test_images_pca):
        """
        Predicts the labels of each image of the testing set by taking the smallest k-distances.
        """
        predicted_labels = []
        for new_image in test_images_pca:
            # predict the labels of each images from the testing set
            predicted_labels += [self._predict(new_image)]
            prediction_matrix = np.array(predicted_labels)
        return prediction_matrix

    def _predict(self, new_image):
        """
            Predicts the label of an image of the testing set by taking the smallest k-distances.
        """

        # compute the distances between the new image and all other images in the training set, so train_images_pca
        distances = [euclidean_distance(train_image, new_image) for train_image in self.train_images_pca]

        # get the k nearest neighbors, the smallest distances, get the nearest labels
        k_indices = np.argsort(distances)[:self.k]
        # sort the distances and find the 5 nearest pictures from the training set
        k_nearest_labels = [self.training_labels[i] for i in k_indices]
        # indices correspond to the labels
        # we want to get the most common class label, majority vote
        most_common = Counter(k_nearest_labels).most_common(1)  # we want to have only one most common item
        return most_common[0][0]  # to get the tuple and the first item of it
