from Classes import knearestneighbors as knn
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import accuracy_score


def set_show(no_images_per_person, image_matrix, image_height, image_width):
    """Shows a representation of the dataset

    :param no_images_per_person: how many pictures each person has.
    :param image_matrix: is the matrix of the images from the  dataset.
    :param image_height: image_height describes height of the image
    :param image_width: image_width describes the width of the image


    :return: showing rows of the image matrix as a picture in a plot
    """
    for i in range(0, 36):
        first_image = i*(no_images_per_person[i])
        vec = (image_matrix[first_image])
        plt.subplot(6, 6, i + 1)
        plt.imshow(vec.reshape(image_height, image_width), cmap="gray")
    plt.show()


def vector_show(vector, image_height, image_width):
    """
    Shows an image from a vector


    :param vector: an image vector to be reshaped into an image
    :param image_height:  image_height describes height of the image
    :param image_width: image_width describes the width of the image


    :return: showing rows of the image matrix as a picture in a plot
    """
    plt.imshow(vector.reshape(image_height, image_width), cmap="gray")
    plt.show()


def show_components(pca_fit_model, image_height, image_width):
    """
    Shows some of the components of the pca model

    :param pca_fit_model: pca_fit_model requires the fitted train_matrix.
    :param image_height: image_height describes height of the image
    :param image_width: image_width describes the width of the image


    :return: showing a picture in a plot
    """
    for i in range(36):
        plt.subplot(6, 6, i + 1)
        plt.imshow(pca_fit_model[i].reshape(image_height, image_width), cmap="gray")
    plt.show()


def k_barplot(k_values, train_images_pca, training_labels, test_images_pca, testing_labels):
    """
        Plots the k values over the accuracy for their KNN model
    :param k_values: the number of k values to be plotted
    :param train_images_pca: train images after dimensionality reduction with pca.
    :param training_labels: labels for training
    :param testing_labels: labels for testing
    :param test_images_pca: test images after dimensionality reduction with pca.
    :return: barplot, for which on x axis are the k values
        and on y axis are the accuracy values.
    """
    accuracy_vec = []
    k_labels = []
    for k in range(1, k_values):
        # the accuracy values
        train_knn = knn.KNearestNeighbors(train_images_pca, training_labels, k)
        predicted_labels = train_knn.predict(test_images_pca)
        k_accuracy = knn.k_accuracy(predicted_labels, testing_labels)
        accuracy_vec += [k_accuracy]
        # the labels
        k_new = "k=" + str(k)
        k_labels += [str(k_new)]
    df = pd.DataFrame(accuracy_vec, k_labels)
    sb.barplot(data=df, x=k_labels, y=accuracy_vec).set(title='Accuracy by different k [%]')
    fig = plt.show()
    return fig


def k_sklearnplot(k_values, train_pca_sklearn, training_labels, test_pca_sklearn, testing_labels):
    """
    Plots the k values over the accuracy for their KNN model.
    Requires sklearn.decomposition for the transformation of the train and test matrices.

    :param k_values: the number of k values to be plotted
    :param train_pca_sklearn: transformed train_matrix by sklearn.decomposition PCA function
    :param training_labels: labels for training
    :param test_pca_sklearn: transformed test_matrix by sklearn.decomposition PCA function
    :param testing_labels: labels for testing


    :return: barplot, for which on x axis are the k values
        and on y axis are the accuracy values.
    """
    k_range = range(1, k_values)
    scores = {}
    scores_list = []
    k_labels = []
    for k in k_range:
        knn1 = KNN(n_neighbors=k)
        knn1.fit(X=train_pca_sklearn, y=training_labels)
        predicted_labels1 = knn1.predict(test_pca_sklearn)
        scores[k] = accuracy_score(testing_labels, predicted_labels1)*100
        scores_list.append(scores[k])
        # the labels
        k_new = "k=" + str(k)
        k_labels += [str(k_new)]

    df = pd.DataFrame(scores_list, k_labels)
    sb.barplot(data=df, x=k_labels, y=scores_list, color="salmon").set(title='Accuracy by different k [%]')
    fig = plt.show()
    return fig
