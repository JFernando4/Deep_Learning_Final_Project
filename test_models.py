""" Code based on Dale's assignment 1 code """
import os
import sys
import numpy as np
import tensorflow as tf
from project_util import gap_timing, wrap_counting, losses, models, experiment
from project_util import measurement as meas
from Data.Data_Util.read_image_data import read_train_data, read_test_data

''' Project Paths'''
projectpath = os.getcwd()
sourcepath = os.path.join(projectpath, '/Data/Util')
datapath = os.path.join(projectpath, '/Data/')
sys.path.append(projectpath)

""" Constants Definitions """
REPEATS = 1
MAX_EPOCH = 40
BATCH = 100 # minibatch size
VALID = 5000 # size of validation set
SEED = None #66478 # None for random seed
ECHO = True
PIXEL_CORRUPTION = 0
LABEL_CORRUPTION = 1
NOISE_TYPE = "Gaussian"

# save filenames
FILELABEL = "MNIST_cpcpf"
FILETAG = "pc"+str(PIXEL_CORRUPTION)+"_lc"+str(LABEL_CORRUPTION)

# plot bounds
YMAX_TRAIN = 10000.0
YMAX_VALID = 200.0
YMAX_TEST  = 400.0

epoch = (60000 - VALID)//BATCH

""" Reading the data """
#     # read MNIST data: matrix format
# valid_data, train_data = read_train_data("MNIST", noise=NOISE_TYPE, plabel=LABEL_CORRUPTION,
#                                                  p=PIXEL_CORRUPTION, image_shape=[28, 28], one_hot=True,
#                                                  num_validation=VALID)
#
# test_data = read_test_data("MNIST", noise="Gaussian", plabel=0, p=0, image_shape=[28,28], one_hot=True)
#
#     # Dimensions
# t, n1, n2, d0 = train_data[0].shape
# n = n1*n2
# te, m = test_data[1].shape
#
#   # minibatch holders
# xdata = np.zeros([BATCH, n1, n2, d0], dtype=np.float32) # minibatch holder
# ydata = np.zeros([BATCH, m], dtype=np.float32)

    # read MNIST data: matrix format
valid_data, train_data = read_train_data("MNIST", noise=NOISE_TYPE, plabel=LABEL_CORRUPTION,
                                                 p=PIXEL_CORRUPTION, image_shape=[784], one_hot=True,
                                                 num_validation=VALID)

test_data = read_test_data("MNIST", noise="Gaussian", plabel=0, p=0, image_shape=[784], one_hot=True)

  # dimensions
t, n = train_data[0].shape
te, m = test_data[1].shape
  # minibatch holders
xdata_vec = np.zeros([BATCH, n], dtype=np.float32)
ydata = np.zeros([BATCH, m], dtype=np.float32)


# wrapper: combines model + optimizer into "method" to run with util1.experiment
def methoddef(name, color, model, optimizer,
              xdata, ydata, data_train, data_valid, data_test):
  """method = model + optimizer:
     wrap a model + optimizer into a method for use with util1.experiment"""
  method = experiment.method(
      name, color, model, optimizer, data_train, xdata, ydata)
  method.meas = [meas.meas_iter(epoch, "step"),
                 meas.meas_loss(model.x, model.y, data_train,
                                model.train_loss, "train_loss", batch=BATCH,
                                axes=[0.0, np.inf, 0.0, YMAX_TRAIN]),
                 meas.meas_loss(model.x, model.y, data_valid,
                                model.misclass_err, "valid_err", batch=BATCH,
                                axes=[0.0, np.inf, 0.0, YMAX_VALID]),
                 meas.meas_loss(model.x, model.y, data_test,
                                model.misclass_err, "test_err", batch=BATCH,
                                axes=[0.0, np.inf, 0.0, YMAX_TEST]),
                 meas.meas_time("train_time") ]
  return method


# define methods

methods = []

# for i in range(2, 10):
#     name = "cpcpf_kl_ml_q"+str(i)
#     color = "#F02311"  #red
#     f1 = 5 # filter size (f1 x f1)
#     d1 = 64 # filter depth (number of independent filters)
#     f2 = 5
#     d2 = 64
#     dimensions = (n1, n2, d0, f1, d1, f2, d2, m)
#     gate_fun = tf.nn.relu
#     loss_fun = lambda z_hat, y: tf.nn.softmax_cross_entropy_with_logits(logits=z_hat, labels=y)
#     # loss_fun = lambda z_hat, y: losses.kl_divergence_ml(z_hat, y, tau=0.2)
#     model = models.model_cp_cp_f(name, dimensions, gate_fun, loss_fun)
#     optimizer = tf.train.GradientDescentOptimizer(0.1/BATCH * (i/2))
#     method = methoddef(name, color, model, optimizer, xdata, ydata,
#                        train_data, valid_data, test_data)
#     methods.append(method)

name = "ff"
color = "#FBB829" #yellow
hidden = 1024 * 5
dimensions = (n, hidden, m)
gate_fun = tf.nn.relu
loss_fun = lambda z_hat, y: tf.nn.softmax_cross_entropy_with_logits(logits=z_hat, labels=y)
# loss_fun = lambda z_hat, y: losses.kl_divergence_ml(z_hat, y, tau=0.2)
model = models.model_5fully_connected(name, dimensions, gate_fun, loss_fun)
optimizer = tf.train.AdamOptimizer((1.0 / BATCH) / 30)
method = methoddef(name, color, model, optimizer, xdata_vec, ydata,
                   train_data, valid_data, test_data)
methods.append(method)


# run experiment

#methods_use = [methods[0]]
#methods_use = [methods[1]]
methods_use = methods

gap_timer = gap_timing.gaptimer(epoch, MAX_EPOCH)
sampler = wrap_counting.wrapcounter(BATCH, t, seed=SEED)

results = experiment.run_methods_list(
    methods_use, gap_timer, sampler, REPEATS, ECHO)

means = experiment.summarize(results) # updates, methods, measures
experiment.print_results(methods_use, means, sys.stdout, FILETAG)
experiment.print_results(methods_use, means, FILELABEL, FILETAG)
# experiment.plot_results(methods_use, means, FILELABEL, FILETAG)
