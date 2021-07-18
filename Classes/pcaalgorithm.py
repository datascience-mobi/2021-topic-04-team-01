import numpy as np
import scipy.linalg as linalg


class PCA:
    """ Principal component analysis (PCA).
        Linear dimensionality reduction of the data to project it to a lower dimensional space.

        train_matrix : image matrix of the training set.

        quality_percent : describes how much of the variance should be described by the model.
    """

    def __init__(self, train_matrix, quality_percent):
        # image_ matrix is from images to matrix
        self.train_matrix = train_matrix
        # how many percent of the data should be represented, converted into a float
        self.quality_percent = quality_percent / 100

        # The real matrix G is our image_matrix
        # It is an n x m matrix
        # n is the number of features/columns, which are the pixels of an image
        # m is the number of samples/rows, which are the people in the dataset
        self.n_samples, self.n_features = train_matrix.shape

        # Centering the data
        self.mean_face = np.mean(self.train_matrix, axis=0)
        # We will add elements of the rows of the image so axis = 0
        # self.mean_face is a row vector mean_face.shape == img_mat[1].shape
        # Z-transformation if we divide by the standard deviation
        self.norm_matrix = (self.train_matrix - self.mean_face)/np.std(train_matrix, axis=0)

    def fit_evd(self):
        """
        Using eigen value decomposition (eigendecomposition) for dimensionality reduction.
        Outputs the components that explain X% of the variance in the data.
        (X% is the quality_percent). TAKES TOO MUCH TIME, BECAUSE OF THE COVARIANCE MATRIX.

        :return: principal components in a form of a matrix
        """

        # EVD only work on square matrices as we need to compute the eigenvalues and eigenvectors
        # For this we compute the covariance matrix K
        # K should be n x n matrix (pixels x pixels)

        # The covariance matrix is nxn
        self.cov_matrix = np.zeros(shape=[self.n_features, self.n_features], dtype='uint8')

        self.cov_matrix = np.cov(self.norm_matrix, rowvar=False)
        # C is a symmetric matrix and so it can be diagonalized:
        eig_val, eig_vec = linalg.eig(self.cov_matrix)

        # Sorting the eigenvectors by decreasing eigenvalues
        # [Start : stop : stepcount] stepcount is reversed
        idx = eig_val.argsort()[::-1]
        eig_val, eig_vec = eig_val[idx], eig_vec[:, idx]

        # Explained_variance tell us how much of the variance in the data each eigen value explains
        explained_variance = eig_val / (self.n_samples - 1)
        # total_var is the total variance in the data
        total_var = explained_variance.sum()
        explained_variance_ratio = explained_variance / total_var
        # The cumulative sum of all ratios
        ratio_cumsum = np.cumsum(explained_variance_ratio)

        # We search in the cumsum for the index of the value which, when added, corresponds to the quality_percent
        # The index of the cumsum gives us the components we need to add to explain X quality percent of our data
        n_components = np.searchsorted(ratio_cumsum, self.quality_percent, side='right') + 1

        self.components = eig_vec[:n_components]
        print("The principal components have been calculated using eigendecomposition", self.components.shape)

        return self.components

    def fit_svd(self):
        """
        Dimensionality reduction using singular value decomposition (SVD).
        Outputs the components that explain X% of the variance in the data.
        (X% is the quality_percent)

        :return:  principal components in a form of a matrix
        """

        # U has the eigenvectors of G.Gt as columns ()
        # S has square roots of the eigenvalues of G.Gt and Gt.G in its diagonal
        # The square roos of the eigenvalues are called singular values
        # V has the eigenvectors of Gt.G as columns ()
        # full_matrices set to false will set the Vt matrix to a shape m x n

        U, S, Vt = linalg.svd(self.norm_matrix, full_matrices=False)

        # Compute the eigenvalues
        eig_val = (S ** 2)

        # Explained_variance tell us how much of the variance in the data each eigen value explains
        explained_variance = eig_val / (self.n_samples - 1)
        # total_var is the total variance in the data
        total_var = explained_variance.sum()
        explained_variance_ratio = explained_variance / total_var
        # The cumulative sum of all ratios
        ratio_cumsum = np.cumsum(explained_variance_ratio)

        # We search in the cumsum for the index of the value which, when added, corresponds to the quality_percent
        # The index of the cumsum gives us the components we need to add to explain X quality percent of our data
        n_components = np.searchsorted(ratio_cumsum, self.quality_percent, side='right') + 1

        self.components = Vt[:n_components]
        print("The principal components have been calculated using svd", self.components.shape)

        return self.components

    def transform(self, image_matrix):
        """Uses the components from the fit functions to transform an Image_matrix (of the training or testing set)

        :param image_matrix: matrix with the images of the training or testing set

        :return: the pixels of the images reduced to the number of principal components
        """

        # Centering the data
        mean = np.mean(image_matrix, axis=0)
        image_matrix = image_matrix - mean

        # Dimension reduction is done by multiplying the original matrix with the components
        transformed_matrix = np.dot(image_matrix, self.components.T)
        return transformed_matrix
