""" Generates the Corrupted Data for MNIST Data Set """

import sys
import os
import numpy as np
import pickle
import Generate_Data.read_mnist
import matplotlib.pyplot as plt

projectpath = os.getcwd() + "/../"
sourcepath = projectpath + 'Generate_Data/'
datapath = sourcepath + 'MNIST_Data/'
outputpath = projectpath + "/Data/Corrupted_MNIST_Data/"

def add_uniform_noise_to_image(filename, outputfile, p, prefix="Train"):
    """ Adds random uniform noise to each of the pixels of """
        # MNIST Data as a (n, 784) shaped array
    images = Generate_Data.read_mnist.read_data(filename)
        # Adding random noise to each pixel with probability p
    images_shape = images.shape
    random_pixels = np.random.binomial(1,p, images_shape).astype(np.float32)
    random_uniform_noise = np.random.uniform(0, images.max(), images_shape)
    random_noise = np.multiply(random_uniform_noise, random_pixels)
    modified_pixels = np.multiply(images, random_pixels)
        # Noisy Images
    noisy_images= np.subtract(np.add(images, random_noise), modified_pixels)
        # Store Data
    print("Storing images with uniform noise (p = "+str(p)+")...")
    pickle.dump(noisy_images, open(outputfile+'MNIST_' + prefix +'_Data_Uniform_p'+str(np.int8(p*10)), 'wb'))

def add_gaussian_noise_to_image(filename, outputfile, p, prefix="Train"):
    """ Adds random uniform noise to each of the pixels of """
    # MNIST Data as a (n, 784) shaped array
    images = Generate_Data.read_mnist.read_data(filename)
        # Adding random noise to each pixel with probability p
    images_shape = images.shape
    images_mean = np.mean(images, 1) # average pixel value per image
    images_std = np.std(images, 1) # variance of pixel value per image
    random_pixels = np.random.binomial(1, p, images_shape).astype(np.float32)
    random_gaussian_noise = np.random.normal(images_mean, images_std,
                                             (images_shape[1], images_shape[0]))
    random_gaussian_noise = np.transpose(random_gaussian_noise)
    random_noise = np.abs(np.multiply(random_gaussian_noise, random_pixels))
    modified_pixels = np.multiply(images, random_pixels)
        # Noisy images
    noisy_images = np.subtract(np.add(images, random_noise), modified_pixels)
        # store data
    print("Storing images with gaussian noise (p = " + str(p) + ")...")
    pickle.dump(noisy_images, open(outputfile + 'MNIST_' + prefix + '_Data_Gaussian_p' + str(np.int8(p * 10)), 'wb'))


def add_noise_to_label(filename, outputfile, p, prefix="Train"):
    """ Switches each label to a random label with a probability p """
        # MNIST data labels as vector of size n
    labels = Generate_Data.read_mnist.extract_labels(filename, one_hot=False)
        # Generating random labels
    random_ints = np.random.randint(labels.min(), labels.max()+1, labels.shape[0]).reshape(labels.shape[0])
    random_binom = np.random.binomial(1, p, labels.shape[0]).astype(np.int64)
    random_labels = np.multiply(random_ints, random_binom)
    modified_labels = np.multiply(labels, random_binom)
        # Noisy Labels
    noisy_labels = np.subtract(np.add(labels, random_labels), modified_labels)
        # Store Labels
    print("Storing random labels (p = " + str(p) + ")...")
    pickle.dump(noisy_labels, open(outputfile+"MNIST_"+prefix+"_Labels_p"+str(np.int8(p*10)), 'wb'))

    # MnistData
train_image_data_filename = datapath + "train-images-idx3-ubyte.gz"
train_image_labels_filename = datapath + "train-labels-idx1-ubyte.gz"
test_image_data_filename = datapath + "t10k-images-idx3-ubyte.gz"
test_image_labels_filename = datapath + "t10k-labels-idx1-ubyte.gz"

image_data = [train_image_data_filename, test_image_data_filename]
image_labels = [train_image_labels_filename, test_image_labels_filename]
prefixes = ['Train', 'Test']

    # Probability of Noise
probabilities = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

if __name__ == "__main__":
    for i in range(2):
        for p in probabilities:
            # add_uniform_noise_to_image(image_data[i], outputpath, p, prefixes[i])
            add_gaussian_noise_to_image(image_data[i], outputpath, p, prefixes[i])
            add_noise_to_label(image_labels[i], outputpath, p, prefixes[i])
