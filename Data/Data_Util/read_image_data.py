""" This script is also modeled after Dale's code for assignment 1 """
import sys
import numpy as np
import os
import pickle

projectpath = os.getcwd()
sourcepath = projectpath + '/Data/Util'
datapath = projectpath + '/Data/'


def open_pickle_file(filepath):
    with open(filepath, 'rb') as f:
        pickle_data = pickle.load(f)
    return pickle_data


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def read_train_data(data_set = "MNIST", noise="", plabel=0,
                              p=0, image_shape=[784], one_hot = True, num_validation=0):
    """ Read MNIST or CIFAR10 Data """

        # Getting the paths for the train and test data sets
    label_corruption_level = "p"+str(plabel)
    corruption_level = "p" + str(p)

    if noise == "Gaussian" or noise == "Uniform":
           corruption_level = noise+"_"+corruption_level
    else:
        raise ValueError("Invalid noise type. Specify Gaussian or Uniform for  p > 0.")

    if data_set == "MNIST":
        if np.prod(image_shape) != 784 or len(image_shape) > 2:
            raise ValueError("Invalid shape specified for MNIST images")
        else:
            image_data_path = os.path.join(datapath, "Corrupted_MNIST_Data/")
    elif data_set == "CIFAR10":
        if np.prod(image_shape) != 1024 or len(image_shape) > 2:
            raise ValueError("Invalid shape specified for CIFAR10 images")
        else:
            image_data_path = os.path.join(datapath, "Corrupted_CIFAR10_Data/")
    else:
        raise NameError("Invalid data set name specified")

        # Train and Test set file names
    train_images_data_filename = os.path.join(image_data_path, data_set + "_Train_Data_" + corruption_level)
    train_images_labels_filename = os.path.join(image_data_path, data_set + "_Train_Labels_" + label_corruption_level)

    """ Train and Validation Data Sets"""
    train_images_data = open_pickle_file(train_images_data_filename)
    if len(image_shape) == 1:
        Xin = train_images_data.reshape([train_images_data.shape[0]] + image_shape)
    elif len(image_shape) == 2:
        if data_set == "CIFAR10":
            d = 3
            Xin = train_images_data.reshape([train_images_data.shape[0]] + [d] + image_shape)
            Xin = Xin.transpose([0, 2, 3, 1])
        elif data_set == "MNIST":
            d = 1
            Xin = train_images_data.reshape([train_images_data.shape[0]] + image_shape + [d])

        # Train and Validation Labels
    train_images_labels = open_pickle_file(train_images_labels_filename)
    if one_hot:
        Yin = dense_to_one_hot(train_images_labels, 10)
    else:
        Yin = train_images_labels

        # Validation Set
    Xval = Xin[0:num_validation, ...]
    Yval = Yin[0:num_validation, ...]
    validation_set = (Xval, Yval)
        # Train Set
    Xtrain = Xin[num_validation:, ...]
    Ytrain = Yin[num_validation:, ...]
    train_set = (Xtrain, Ytrain)

    return validation_set, train_set #, test_set

def read_test_data(data_set = "MNIST", noise="", plabel=0, p=0, image_shape=[784], one_hot = True):

        # Getting the paths for the train and test data sets
    label_corruption_level = "p"+str(plabel)
    corruption_level = "p" + str(p)

    if noise == "Gaussian" or noise == "Uniform":
           corruption_level = noise+"_"+corruption_level
    else:
        raise ValueError("Invalid noise type. Specify Gaussian or Uniform for  p > 0.")

    if data_set == "MNIST":
        if np.prod(image_shape) != 784 or len(image_shape) > 2:
            raise ValueError("Invalid shape specified for MNIST images")
        else:
            image_data_path = os.path.join(datapath, "Corrupted_MNIST_Data/")
    elif data_set == "CIFAR10":
        if np.prod(image_shape) != 1024 or len(image_shape) > 2:
            raise ValueError("Invalid shape specified for CIFAR10 images")
        else:
            image_data_path = os.path.join(datapath, "Corrupted_CIFAR10_Data/")
    else:
        raise NameError("Invalid data set name specified")

    test_images_data_filename = os.path.join(image_data_path, data_set + "_Test_Data_" + corruption_level)
    test_images_labels_filename = os.path.join(image_data_path, data_set + "_Test_Labels_" + label_corruption_level)

    """ Test Data Set """
    test_images_data = open_pickle_file(test_images_data_filename)
    if len(image_shape) == 1:
        Xtest = test_images_data.reshape([test_images_data.shape[0]] + image_shape)
    elif len(image_shape) == 2:
        if data_set == "CIFAR10":
            d = 3
            Xtest = test_images_data.reshape([test_images_data.shape[0]] + [d] + image_shape)
            Xtest = Xtest.transpose([0, 2, 3, 1])
        elif data_set == "MNIST":
            d = 1
            Xtest = test_images_data.reshape([test_images_data.shape[0]] + image_shape + [d])
        # Test Labels
    test_images_labels = open_pickle_file(test_images_labels_filename)
    if one_hot:
        Ytest = dense_to_one_hot(test_images_labels)
    else:
        Ytest = test_images_labels
    test_set = (Xtest, Ytest)
    return test_set
