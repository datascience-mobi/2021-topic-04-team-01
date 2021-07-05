import numpy as np
import scipy.linalg as linalg

class PCA:
    # that is called a constructor
    def __init__(self, image_matrix, quality_percent):
        # image_ matrix is from images to matrix
        self.image_matrix = image_matrix
        # how many images are there in the dataset
        self.no_of_images = self.image_matrix.shape[0]
        # how many percent of the data should be represented
        self.quality_percent = quality_percent

        # image matrix has rows as people and columns as pixels
        # We will add elements of the rows of the image so 0
        mean = np.mean(self.image_matrix, axis=0)
        # self.mean_face is a row vector mean_face.shape == img_mat[1].shape
        self.mean_face = np.asmatrix(mean)
        self.norm_matrix = self.image_matrix - self.mean_face
        # We will use the covariance matrix C to preserve variance
        # Using the covariance matrix it is the same whether we use svd or eigenvector decomposition
        # As we are looking at pixels
        # the range and scale of variables is similar and they have the same units of measurement.

        # Transposing the matrix for reduction in computational time
        self.cov_matrix = np.cov(self.norm_matrix.T, rowvar=False)
        # C is a symmetric matrix and so it can be diagonalized:
        self.eig_val, self.eig_vec = linalg.eig(self.cov_matrix)

        # Sorting the eigenvectors by decreasing eigenvalues
        # [Start : stop : stepcount] stepcount is reversed
        idx = self.eig_val.argsort()[::-1]
        self.eig_val, self.eig_vec = self.eig_val[idx], self.eig_vec[:, idx]

        # projections of X on the principal axes are called principal components
        # covariance matrix times the eigenvector matrix

    def n_components(self):
        # Where did we calculate the eig_vals
        sum_original = np.sum(self.eig_val)
        # how well should be able to explain the variability of the data
        # check in the other PCA function
        sum_threshold = sum_original * self.quality_percent / 100
        # sum of the eigenvalues until the threshold
        sum_temp = 0
        # number of eigenvalues
        p = 0
        while sum_temp < sum_threshold:
            sum_temp += self.eig_val[p]
            p += 1
        return p
        # says how much eigenvalues we have used

    def fit(self):
        # singular value decomposition
        # U is unitary array AA.t, S is square roots of the eigenvalues,  Vt is the transposed matrix of A_tA
        # used svd for dim_reduction
        # An m by n matrix M
        # Where U is an m by r matrix, V is an n by r matrix,
        # and S is an r by r matrix; where r is the rank of the matrix M.
        # V is transposed so you can have a r by n matrix in the end
        # full_matrices means they are square m m and n n

        self.mean = np.mean(self.image_matrix, axis=0)
        self.image_matrix -= self.mean

        U, S, Vt = linalg.svd(self.image_matrix, full_matrices=False)
        p = self.n_components()
        # combine only the first p singular values so p is our rank
        # That is where the p should be
        self.components = Vt[:p]
        # dot multiplycation is multiply the elements with eachother
        # self is a matrix so we multiply U-transposed with M
        return self.components

    def transform(self, X):

        self.mean = np.mean(X, axis=0)
        X = X - self.mean
        X_transformed = np.dot(X, self.components.T)
        return X_transformed

    # find new coordinates of any pixel of an image




