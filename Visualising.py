import matplotlib.pyplot as plt
from Classes import knearestneighbors as knn
import pandas as pd
import seaborn as sb
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import accuracy_score


def train_show(no_images_training, train_matrix, image_height, image_width):
    """ Shows a representation of the training set
            \n
            The no_images_training is a vector where each element shows how many pictures the person has for training .
            \n
            The train_matrix is the matrix of the training set.
            \n
            image_height, image_width describe how big the image is.
        """
    for i in range(0, 36):
        first_image = i*(no_images_training[i])
        vec = (train_matrix[first_image])
        plt.subplot(6, 6, i + 1)
        plt.imshow(vec.reshape(image_height, image_width), cmap="gray")
    plt.show()


def test_show(no_images_testing, test_matrix, image_height, image_width):
    """ Shows a representation of the training set
            \n
            The no_images_testing is a vector where each element shows how many pictures the person has for testing .
            \n
            The test_matrix is the matrix of the testing set.
            \n
            image_height, image_width describe how big the image is.
        """
    for i in range(0, 36):
        first_image = i*(no_images_testing[i])
        vec = (test_matrix[first_image])
        plt.subplot(6, 6, i + 1)
        plt.imshow(vec.reshape(image_height, image_width), cmap="gray")
    plt.show()


def vector_show(vector, image_height, image_width):
    """ Shows an image from a vector
            \n
            The vector is an image vector to be reshaped into an image
            \n
            image_height, image_width describe how big the image is.
        """
    plt.imshow(vector.reshape(image_height, image_width), cmap="gray")
    plt.show()


def show_components(pca_fit_model, image_height, image_width):
    """ Shows some of the components of the pca model
                \n
                The pca_fit_model requires the fitted train_matrix.
                \n
                image_height, image_width describe how big the image is.
            """
    for i in range(36):
        plt.subplot(6, 6, i + 1)
        plt.imshow(pca_fit_model[i].reshape(image_height, image_width), cmap="gray")
    plt.show()


def k_barplot(k1, train_tr, yaleB, test_tr):
    """
        This function returns a barplot, for which on x axis are the k values
        and on y axis are the accuracy values.
    """
    k_vector = []
    for k in range(1, 16):
        train_knn = knn.KNearestNeighbors(train_tr, yaleB.training_labels, k)
        predicted_labels = train_knn.predict(test_tr)
        k_accuracy = knn.k_accuracy(predicted_labels, yaleB.testing_labels)
        k_vector.append(k_accuracy)
    df = pd.DataFrame(k_vector, k1)
    sb.barplot(data=df, x=k1, y=k_vector).set(title='Accuracy by different k [%]')
    fig = plt.show()
    return fig


def k_sklearnplot(k1, train, yaleB, test):
    """
        This function returns a barplot, for which on x axis are the k values
        and on y axis are the accuracy values.
        The ready-made sklearn classifier was used.
    """
    k_range = range(1, 16)
    scores = {}
    scores_list = []
    for k in k_range:
        knn1 = KNN(n_neighbors=k)
        knn1.fit(X=train, y=yaleB.training_labels)
        predicted_labels1 = knn1.predict(test)
        scores[k] = accuracy_score(yaleB.testing_labels, predicted_labels1)*100
        scores_list.append(scores[k])
    df = pd.DataFrame(scores_list, k1)
    sb.barplot(data=df, x=k1, y=scores_list, color="salmon").set(title='Accuracy by different k [%]')
    fig = plt.show()
    return fig

