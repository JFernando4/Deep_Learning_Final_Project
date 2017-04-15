""" Imports """
import os
import sys
from Data.Data_Util import read_image_data

''' Project Paths'''
projectpath = os.getcwd()
sourcepath = os.path.join(projectpath, '/Data/Util')
datapath = os.path.join(projectpath, '/Data/')


sys.path.append(datapath)

import matplotlib.pyplot as plt
import numpy as np

train = list()
for i in range(0, 11):
    temp_valid, temp_train = read_image_data.read_train_data("MNIST", "Gaussian", plabel=0, p=i, image_shape=[784],
                                                             one_hot=True, num_validation=1000)
    train.append(temp_train)

f, axarr = plt.subplots(2, 6)
for i in range(0,11):
    images = train[i][0]
    xindex = i % 6
    yindex = int(np.floor(i/6))
    axarr[yindex, xindex].imshow(images[100].reshape((28, 28)))



# validation2, train2 = read_image_data.read_train_data("MNIST","Gaussian", plabel=10, p=0, image_shape=[784],
#                                                                 one_hot=False, num_validation=1000)
# test = read_image_data.read_test_data("MNIST", noise="Gaussian", plabel=0, p=5, image_shape=[784], one_hot=True)
# images = test[0]
# plt.imshow(images[0].reshape((28,28)))
