
""" Imports """
import pickle
import numpy as np
import os

""" Defining Paths """
projectpath = os.getcwd() + "/../"
# homepath = '/home/jfernando/'
# projectpath = homepath + 'PycharmProjects/Deep_Learning_Final_Project/'
sourcepath = projectpath + 'Generate_Data/'
datapath = sourcepath + 'CIFAR10_Data/'
outputpath = projectpath + "/Data/Corrupted_CIFAR10_Data/"



def add_uniform_noise_to_image(images, outputfile, p, prefix="Train", image_range=[0.0, 1.0]):
    """ Adds random uniform noise to each of the pixels of each image"""
        # Adding random noise to each pixel with probability p
    images = images.astype(np.float64)
    images *= (image_range[1] - image_range[0]) / images.max()
    images += image_range[0]
    images_shape = images.shape
    random_binom = np.random.binomial(1, p, [images_shape[0], np.int32(images_shape[1]/3)]).astype(np.float32)
    random_pixels = np.column_stack([random_binom, random_binom, random_binom])
    random_uniform_noise = np.random.uniform(0, images.max(), images_shape)
    random_noise = np.multiply(random_uniform_noise, random_pixels)
    modified_pixels = np.multiply(images, random_pixels)
        # Noisy Images
    noisy_images= np.subtract(np.add(images, random_noise), modified_pixels)
        # Store Data
    pickle.dump(noisy_images, open(outputfile+'CIFAR10_' + prefix +'_Data_p'+str(np.int8(p*10)), 'wb'))

def add_gaussian_noise_to_image(images, outputfile, p, prefix="Train", image_range=[0.0, 1.0]):
    """ Adds random gaussian noise to each of the pixels of each images"""
        # Adding random noise to each pixel with probability p
    images = images.astype(np.float64)
    images *= (image_range[1] - image_range[0]) / images.max()
    images += image_range[0]
        # Adding random noise to each pixel with probability p
    images_shape = images.shape
    images = np.reshape(images, (images_shape[0], 32, 32, 3))
    images_mean = np.mean(images, 0)
    images_std = np.std(images, 0)

    pass

def add_noise_to_label(labels, outputfile, p, prefix="Train"):
    """ Switches each label to a random label with a probability p """
        # Generating random labels
    random_ints = np.random.randint(labels.min(), labels.max()+1, labels.shape[0]).reshape(labels.shape[0])
    random_binom = np.random.binomial(1, p, labels.shape[0]).astype(np.int64)
    random_labels = np.multiply(random_ints, random_binom)
    modified_labels = np.multiply(labels, random_binom)
        # Noisy Labels
    noisy_labels = np.subtract(np.add(labels, random_labels), modified_labels)
        # Store Labels
    pickle.dump(noisy_labels, open(outputfile+"CIFAR10_"+prefix+"_Labels_p"+str(np.int8(p*10)), 'wb'))


if __name__ == "__main__":
    image_data = [np.load(open(datapath+"CIFAR10_Train_Data", "rb")),
                  np.load(open(datapath+"CIFAR10_Test_Data", "rb"))]
    image_labels = [np.load(open(datapath+"CIFAR10_Train_Labels", "rb")),
                    np.load(open(datapath+"CIFAR10_Test_Labels", "rb"))]
    prefixes = ["Train", "Test"]
    probabilities = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for i in range(2):
        for p in probabilities:
            add_gaussian_noise_to_image(image_data[i], outputpath, p, prefixes[i])
            add_uniform_noise_to_image(image_data[i], outputpath, p, prefixes[i])
            # add_noise_to_label(image_labels[i], outputpath, p, prefixes[i])
