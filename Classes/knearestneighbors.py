import numpy as np
from collections import Counter


def euclidean_distance(train_images_pca, test_image):
    """Euclidean distance formula.

    :param train_images_pca: train images after dimensionality reduction with pca.
    :param test_image: image from the testing set
    :return: distance between the image and every image of the training set
    """

    distance = np.sqrt(np.sum(np.square(train_images_pca - test_image)))
    return distance


def k_accuracy(predicted_labels, testing_labels):
    """ Calculates accuracy of the model. Can be used as an attribute predicted_labels.accuracy(testing_labels).

    :param predicted_labels: labels predicted with the KNN model
    :param testing_labels: labels of the testing images obtained by the splitting

    :return: what part of the images are correctly labeled in percent
    """
    correct = 0
    for i in range(predicted_labels.shape[0]):
        if predicted_labels[i] == testing_labels[i]:
            correct += 1
        else:
            continue
    accuracy_percent = (correct / predicted_labels.shape[0]) * 100
    return accuracy_percent


class KNearestNeighbors:
    """ K-Nearest Neighbors (KNN).
            Calculates the euclidean distance of each image of the testing set to each image of the training set.

            train_images_pca: train images after dimensionality reduction with pca.

            training_labels: are the labels (folder names) of each image of the training set.

            k is the number of neighbors which we use to the determine the label of the testing image.
        """
    def __init__(self, train_images_pca, training_labels, k=2):
        self.k = k
        self.train_images_pca = train_images_pca
        self.training_labels = training_labels

    def predict(self, test_images_pca):
        """
        Predicts the labels of each image of the testing set by taking the smallest k-distances.

        :param test_images_pca: test images after dimensionality reduction with pca.

        :return: matrix with the predicted labels of each image of the testing set
        """

        predicted_labels = []
        for new_image in test_images_pca:
            # predict the labels of each images from the testing set
            predicted_labels += [self.predict_one(new_image)]
        return np.array(predicted_labels)

    def predict_one(self, new_image):
        """
         Predicts the label of an image of the testing set by taking the smallest k-distances.

        :param new_image: new image from the testing set
        :return: label of the image
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
