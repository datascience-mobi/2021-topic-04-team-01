import matplotlib.pyplot as plt


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
