import numpy as np
import scipy.linalg as s_linalg
from skimage import io

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
        self.eig_val, self.eig_vec = s_linalg.eig(self.cov_matrix)

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

    def fit_transform(self):
        p = self.n_components()
        # combine only the first p singular values so p is our rank
        # we take the first p eigenvectors from the eigenvector matrix
        self.components = self.eig_vec[:, 0:p]
        # dot multiplication is multiply the elements with each other
        # covariance matrix times the eigenvector matrix gives us the new coordinates of the pixels
        # for matrix multiplication the inner dimensions should be the same 2x3 3x2
        return np.dot(self.cov_matrix, self.components)

    # find new coordinates of any pixel of an image

    def new_cords(self, single_image_path):
        # give you the image presented through the svd/pca
        # make it into a row vector
        img_gray = io.imread(single_image_path, as_gray=True)
        img_vec = img_gray.ravel()
        # mean_face is a row by number of people plus the values for the image
        # (self.mean_face * n) is the sum of all pixel values in the dataset
        # n is the number of people, it should be the number of images in the dataset
        # maybe put no_of_elements instead of labels
        n = self.no_of_images
        new_mean = ((self.mean_face * n) + img_vec) / (n + 1)
        # n+1 means you have increase the number of images by one
        img_vec = img_vec - new_mean

        return self.components.dot(img_vec)

    '''def recognize_face(self, new_cords_of_image):
        classes = len(self.no_of_elements)
        start = 0
        dist = []
        for i in range(classes):
            # check the new_coordinates from the first column to the last column
            # from 0 to the last column to the i th member
            # he does it column-wise
            # number of elements is number of people
            temp_imgs = self.new_coordinates[:, int(start):int(start + self.no_of_elements[i])]
            # temp is a matrix of the images until the i th member
            # only of the class
            # here again in should be 0 instead of 1, row-wise
            # should be the same as up
            mean_temp = np.asmatrix(np.mean(temp_imgs, 1)).T
            # calculate the mean of the images
            # the starting position of the next loop
            start = start + self.no_of_elements[i]
            # we increase start with the i th member so 1 2 3 4
            # it rises every element of matrix M
            # ith the power of the mean and then takes the root of the mean from it
            # dist_ temp is a matrix of the distances between
            # each pixel with the other pixel
            # subtract sign instead of a coma!!!
            dist_temp = np.linalg.norm(new_cords_of_image - mean_temp)
            dist += [dist_temp]
        # returns only the min_ distance for the min_pos so the face we are looking for
        min_pos = np.argmin(dist)
        return self.image_targets[min_pos]

    # add more methods, not mandatory
    # to find an original image
    def new_to_old_cords(self, new_cords):
        # we add the mean_face
        # new cord should be a matrix which has pca already
        return self.mean_face + (np.asmatrix(np.dot(self.new_bases, new_cords))).T

    def show_image(self, label_to_show, old_cords):
        # to have the same size
        old_cords_matrix = np.reshape(old_cords, [self.images_width, self.images_height])
        # as an 8 bit value of colors, integers of 8
        old_cords_integers = np.array(old_cords_matrix, dtype=np.uint8)
        # the new shape is 500 by 500
        resized_image = np.resize(old_cords_integers, (500, 500))
        cv2.imshow(label_to_show, resized_image)
        cv2.waitKey()

    def show_eigenfaces(self, min_pix_int, max_pix_int, eig_face_no):
        # eigenfaces are next to 0
        # change pixel intensity pix_int
        # new.bases is after the svd show only until the eig_number
        # because it counts from 0 we increase by 1
        # eigenvector, whole column is pixels
        ev = self.new_bases[:, eig_face_no: eig_face_no + 1]
        min_orig = np.min(ev)
        max_orig = np.max(ev)
        # so 100 and 0 it would be 100/(min and macs of the image we change the contrast)
        ev = min_pix_int + (((max_pix_int - min_pix_int) / (max_orig - min_orig)) * ev)
        self.show_image("Eigenface " + str(eig_face_no), ev)'''
