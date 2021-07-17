import numpy as np
from itertools import chain


def random_indices(array, train_percent, seed):
    """
    Produces random indices used to split the data.

    :param array: an array, whose size is used for the calculation
    :param train_percent: the percentage of the data that goes into training
    :param seed:  random number used to generate the indices of the split matrices. By using he same number the
    results are the same.


    :return: indices for training and testing
    """

    train_size = train_percent/100
    n_samples = len(array)
    # try here with int() floor
    n_train = int(train_size * n_samples)
    n_test = n_samples - n_train
    n_train, n_test = int(n_train), int(n_test)
    rng = np.random.RandomState(seed)
    print("There are", n_train, "images in training and", n_test, "images in testing")

    # random partition
    permutation = rng.permutation(n_samples)
    # all the permutations from 0 to n_test
    test_ind = permutation[:n_test]
    # all the permutations from n_test to the end
    train_ind = permutation[n_test:(n_test + n_train)]
    # because n_test is included in the second line, no indices duplicate
    yield train_ind, test_ind


def train_test_split(*arrays, train_percent, seed):
    """
    Split arrays or matrices into random train and test subsets.

    :param arrays: all of the given matrices
    :param train_percent: the percentage of the data that goes into training
    :param seed: random number used to generate the indices of the split matrices. By using he same number the
    results are the same.


    :return: A test and train split of each array

    """

    # we start from the 0th element in arrays and with next we continue until the last one
    train_ind, test_ind = next(random_indices(array=arrays[0], train_percent=train_percent, seed=seed))


    return list(chain.from_iterable((a[train_ind], a[test_ind]) for a in arrays))
