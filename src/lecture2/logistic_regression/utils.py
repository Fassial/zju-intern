"""
Created on August 01 02:06, 2020

@author: fassial
"""
import os
import gzip
import pickle
import numpy as np
from PIL import Image

def sigmoid(x):
    """The logistic sigmoid function"""
    return 1 / (1 + np.exp(-x))

def softmax(x, dim = -1):
    """The logistic softmax function"""
    # center data to avoid overflow
    e_x = np.exp(x - np.max(x, axis=dim, keepdims=True))
    return e_x / e_x.sum(axis=dim, keepdims=True)

def one_hot(label, n_samples, n_classes):
    """Onehot function"""
    one_hot = np.zeros((n_samples, n_classes))
    one_hot[np.arange(n_samples), label.T] = 1
    return one_hot

def visualize_image(X, filename = "test.jpg"):
    X = X.reshape(28, 28)*255
    img = Image.fromarray(np.uint8(X)).convert('RGB')
    img.save(filename)

def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        dataset_path = os.path.join(
            os.path.split(__file__)[0],
            "data"
        )
        # check whether dataset exists
        if not os.path.exists(dataset_path): os.mkdir(dataset_path)
        new_path = os.path.join(
            dataset_path,
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('... loading data')

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        return data_x, data_y

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

DATASET = 'mnist.pkl.gz'

def test_softmax():
    # init softmax input
    a = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]])
    # get softmax res
    res = softmax(a)
    print(res)
    res = np.argmax(res, axis=1).reshape((-1,1))
    print(res)

def test_one_hot():
    # init label & n_samples & n_classes
    n_samples = 5; n_classes = 5
    label = np.array([i for i in range(n_samples)])
    # get one_hot
    res = one_hot(
        label = label,
        n_samples = n_samples,
        n_classes = n_classes
    )
    print(res)

if __name__ == "__main__":
    # datasets = load_data(dataset = DATASET)
    test_softmax()
    # test_one_hot()

