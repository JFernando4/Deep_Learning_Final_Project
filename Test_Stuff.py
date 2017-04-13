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

validation, train = read_image_data.read_train_data("MNIST","Gaussian", plabel=0, p=10, image_shape=[784],
                                                                one_hot=False, num_validation=1000)
validation2, train2 = read_image_data.read_train_data("MNIST","Gaussian", plabel=10, p=0, image_shape=[784],
                                                                one_hot=False, num_validation=1000)
# images = train[0]
# plt.imshow(images[0].reshape((28,28)))

# test = read_image_data.read_test_data("MNIST", noise="Gaussian", plabel=0, p=5, image_shape=[784], one_hot=True)
# images = test[0]
# plt.imshow(images[0].reshape((28,28)))





# """ Code based on Dale's assignment 1 code """
# import os
# homepath = '/home/jfernando/'
# projectpath = os.path.join(homepath, 'PycharmProjects/Deep_Learning_Final_Project/')
# sourcepath = projectpath
# datapath = os.path.join(projectpath, "Data/")
#
# import sys
# sys.path.append(projectpath)
#
# import numpy as np
# import tensorflow as tf
# from project_util import experiment
# from project_util import gap_timing
# from project_util import wrap_counting
# from project_util import measurement as meas
# from project_util import losses
# from project_util import models
# from Data.Data_Util import read_image_data
#
#
# # experimental configuration
# REPEATS = 1
# MAX_EPOCH = 100
# BATCH = 100 # minibatch size
# VALID = 5000 # size of validation set
# SEED = None #66478 # None for random seed
# PERMUTE = False # permute pixels
# ECHO = True
#
# # save filenames
# FILELABEL = "MNIST_2Fully_Connected"
# CORRUPTION = "10"
#
# # plot bounds
# YMAX_TRAIN = 10000.0
# YMAX_VALID = 200.0
# YMAX_TEST  = 400.0
#
# epoch = (60000 - VALID)//BATCH
#
#
# # data
#
#   # read MNIST data: vector input format (used by fully connected model)
# data_valid_vec, data_train_vec, data_test_vec = read_image_data.read_data("MNIST", p=10, image_shape=[784],
#                                                                           one_hot=True, num_validation=VALID)
#   # dimensions
# t, n = data_train_vec[0].shape
# te, m = data_test_vec[1].shape
#   # minibatch holders
# xdata_vec = np.zeros([BATCH, n], dtype=np.float32)
# ydata = np.zeros([BATCH, m], dtype=np.float32)
#
# # wrapper: combines model + optimizer into "method" to run with util1.experiment
# def methoddef(name, color, model, optimizer,
#               xdata, ydata, data_train, data_valid, data_test):
#   """method = model + optimizer:
#      wrap a model + optimizer into a method for use with util1.experiment"""
#   method = experiment.method(
#       name, color, model, optimizer, data_train, xdata, ydata)
#   method.meas = [meas.meas_iter(epoch, "step"),
#                  meas.meas_loss(model.x, model.y, data_train,
#                                 model.train_loss, "train_loss", batch=BATCH,
#                                 axes=[0.0, np.inf, 0.0, YMAX_TRAIN]),
#                  meas.meas_loss(model.x, model.y, data_valid,
#                                 model.misclass_err, "valid_err", batch=BATCH,
#                                 axes=[0.0, np.inf, 0.0, YMAX_VALID]),
#                  meas.meas_loss(model.x, model.y, data_test,
#                                 model.misclass_err, "test_err", batch=BATCH,
#                                 axes=[0.0, np.inf, 0.0, YMAX_TEST]),
#                  meas.meas_time("train_time") ]
#   return method
#
#
# # define methods
#
# methods = []
#
# name = "f_f-kl_ml"
# color = "#FBB829" #yellow
# hidden = 1024
# dimensions = (n, hidden, m)
# gate_fun = tf.nn.relu
# loss_fun = lambda z_hat, y: losses.kl_divergence_ml(z_hat, y, tau=0.2)
# model = Models.model_2fully_connected(name, dimensions, gate_fun, loss_fun)
# optimizer = tf.train.AdamOptimizer((1.0 / BATCH) / 30)
# method = methoddef(name, color, model, optimizer, xdata_vec, ydata,
#                    data_train_vec, data_valid_vec, data_test_vec)
# methods.append(method)
#
#
# # run experiment
#
# #methods_use = [methods[0]]
# #methods_use = [methods[1]]
# methods_use = methods
#
# gap_timer = gap_timing.gaptimer(epoch, MAX_EPOCH)
# sampler = wrap_counting.wrapcounter(BATCH, t, seed=SEED)
#
# results = experiment.run_methods_list(
#     methods_use, gap_timer, sampler, REPEATS, ECHO)
#
# means = experiment.summarize(results) # updates, methods, measures
# experiment.print_results(methods_use, means, sys.stdout, CORRUPTION)
# experiment.print_results(methods_use, means, FILELABEL, CORRUPTION)
# experiment.plot_results(methods_use, means, FILELABEL, CORRUPTION)
