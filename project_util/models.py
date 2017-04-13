""" This models are based on Dale's code from assignment 1 """
import os
homepath = '/home/jfernando/'
projectpath = os.path.join(homepath, 'PycharmProjects/Deep_Learning_Final_Project/')
sourcepath = os.path.join(projectpath, '/Util')

import sys
sys.path.append(projectpath)

from project_util import layers
import tensorflow as tf
import numpy as np


class model_2fully_connected():
  """Define an f_f model: input -> fully connected -> output"""
  def __init__(self, name, dimensions, gate_fun, loss_fun, SEED=None):
      # placeholders
    dim_in, hidden, dim_out = dimensions
    self.x = tf.placeholder(tf.float32, shape=(None, dim_in)) # input
    self.y = tf.placeholder(tf.float32, shape=(None, dim_out)) # target
      # layer 1: full
    W_1, b_1, z_hat_1, y_hat_1 = layers.fully_connected(
        name, "layer_1", self.x, dim_in, hidden,
        tf.random_normal_initializer(stddev=1.0/np.sqrt(dim_in+1), seed=SEED),
        gate_fun)
      # layer 2: full
    W_2, b_2, z_hat, y_hat = layers.fully_connected(
        name, "layer_2", y_hat_1, hidden, dim_out,
        tf.random_normal_initializer(stddev=1.0/np.sqrt(hidden+1), seed=SEED),
        tf.nn.softmax)
      # loss
    self.train_loss = tf.reduce_sum(loss_fun(z_hat, self.y))
    self.train_vars = [W_1, b_1, W_2, b_2]
    self.misclass_err = tf.reduce_sum(tf.cast(
        tf.not_equal(tf.argmax(y_hat, 1), tf.argmax(self.y, 1)), tf.float32))


class model_5fully_connected():
  """Define an 5f model: input -> 5 x fully connected -> output"""
  def __init__(self, name, dimensions, gate_fun, loss_fun, SEED=None):
      # placeholders
    dim_in, hidden, dim_out = dimensions
    self.x = tf.placeholder(tf.float32, shape=(None, dim_in)) # input
    self.y = tf.placeholder(tf.float32, shape=(None, dim_out)) # target
      # layer 1: full
    W_1, b_1, z_hat_1, y_hat_1 = layers.fully_connected(
        name, "layer_1", self.x, dim_in, hidden,
        tf.random_normal_initializer(stddev=1.0/np.sqrt(dim_in+1), seed=SEED),
        gate_fun)
      # layer 2: full
    W_2, b_2, z_hat_2, y_hat_2 = layers.fully_connected(
        name, "layer_2", y_hat_1, hidden, hidden,
        tf.random_normal_initializer(stddev=1.0/np.sqrt(hidden+1), seed=SEED),
        tf.nn.softmax)
      # layer 3: full
    W_3, b_3, z_hat_3, y_hat_3 = layers.fully_connected(
        name, "layer_3", y_hat_2, hidden, hidden,
        tf.random_normal_initializer(stddev=1.0 / np.sqrt(hidden + 1), seed=SEED),
        tf.nn.softmax)
      # layer 4: full
    W_4, b_4, z_hat_4, y_hat_4 = layers.fully_connected(
        name, "layer_4", y_hat_3, hidden, hidden,
        tf.random_normal_initializer(stddev=1.0 / np.sqrt(hidden + 1), seed=SEED),
        tf.nn.softmax)
      # layer 5: full
    W_5, b_5, z_hat, y_hat = layers.fully_connected(
        name, "layer_5", y_hat_4, hidden, dim_out,
        tf.random_normal_initializer(stddev=1.0 / np.sqrt(hidden + 1), seed=SEED),
        tf.nn.softmax)

      # loss
    self.train_loss = tf.reduce_sum(loss_fun(z_hat, self.y))
    self.train_vars = [W_1, b_1, W_2, b_2, W_3, b_3, W_4, b_4, W_5, b_5]
    self.misclass_err = tf.reduce_sum(tf.cast(
        tf.not_equal(tf.argmax(y_hat, 1), tf.argmax(self.y, 1)), tf.float32))

class model_3fully_connected():
  """Define an f_f model: input -> fully connected -> output"""
  def __init__(self, name, dimensions, gate_fun, loss_fun, SEED=None):
      # placeholders
    dim_in, hidden, dim_out = dimensions
    self.x = tf.placeholder(tf.float32, shape=(None, dim_in)) # input
    self.y = tf.placeholder(tf.float32, shape=(None, dim_out)) # target
      # layer 1: full
    W_1, b_1, z_hat_1, y_hat_1 = layers.fully_connected(
        name, "layer_1", self.x, dim_in, hidden,
        tf.random_normal_initializer(stddev=1.0/np.sqrt(dim_in+1), seed=SEED),
        gate_fun)
      # layer 2: full
    W_2, b_2, z_hat_2, y_hat_2 = layers.fully_connected(
        name, "layer_2", y_hat_1, hidden, hidden,
        tf.random_normal_initializer(stddev=1.0/np.sqrt(hidden+1), seed=SEED),
        gate_fun)
      # layer 3: full
    W_3, b_3, z_hat, y_hat = layers.fully_connected(
        name, "layer_3", y_hat_2, hidden, dim_out,
        tf.random_normal_initializer(stddev=1.0 / np.sqrt(hidden + 1), seed=SEED),
        tf.nn.softmax)
      # loss
    self.train_loss = tf.reduce_sum(loss_fun(z_hat, self.y))
    self.train_vars = [W_1, b_1, W_2, b_2, W_3, b_3]
    self.misclass_err = tf.reduce_sum(tf.cast(
        tf.not_equal(tf.argmax(y_hat, 1), tf.argmax(self.y, 1)), tf.float32))

class model_cp_cp_f():
  """Define a cp_cp_f model: input -> conv & pool -> conv & pool -> output"""
  def __init__(self, name, dimensions, gate_fun, loss_fun, SEED=None):
      # placeholders
    n1, n2, d0, f1, d1, f2, d2, m = dimensions
    self.x = tf.placeholder(tf.float32, shape=(None, n1, n2, d0)) # input
    self.y = tf.placeholder(tf.float32, shape=(None, m)) # target
      # layer 1: conv
    W_1, b_1, z_hat_1, r_hat_1 = layers.convolution_2d(
        name, "layer_1", self.x, f1, d0, d1,
        tf.random_normal_initializer(stddev=1.0/np.sqrt(f1*f1*d0+1), seed=SEED),
        gate_fun)
      # layer 1.5: pool
    s_hat_1 = tf.nn.max_pool(
        r_hat_1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
      # layer 2: conv
    W_2, b_2, z_hat_2, r_hat_2 = layers.convolution_2d(
        name, "layer_2", s_hat_1, f2, d1, d2,
        tf.random_normal_initializer(stddev=1.0/np.sqrt(f2*f2*d1+1), seed=SEED),
        gate_fun)
      # layer 2.5: pool
    s_hat_2 = tf.nn.max_pool(
        r_hat_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
    shape_2 = s_hat_2.get_shape().as_list()
    y_hat_2 = tf.reshape(s_hat_2, [-1, shape_2[1]*shape_2[2]*shape_2[3]])
      # layer 3: full
    W_3, b_3, z_hat, y_hat = layers.fully_connected(
        name, "layer_3", y_hat_2, (n1*n2*d2)//16, m,
        tf.random_normal_initializer(stddev=1.0/np.sqrt((n1*n2*d2)//16),
                                     seed=SEED),
        tf.nn.softmax)
      # loss
    self.train_loss = tf.reduce_sum(loss_fun(z_hat, self.y))
    self.train_vars = [W_1, b_1, W_2, b_2, W_3, b_3]
    self.misclass_err = tf.reduce_sum(tf.cast(
        tf.not_equal(tf.argmax(y_hat, 1), tf.argmax(self.y, 1)), tf.float32))


class model_cp_f_f():
    """ Define a cp f f model: input -> conv & pool -> 2 x fully connected -> output """
    def __init__(self, name, dimensions, gate_fun, loss_fun, SEED=None):
        # placeholders
        # n1 x n2 = image dimensions
        # d0 = number of channels (rgb or greyscale)
        # d1 = number of independent filters
        # f1 = filter size
        n1, n2, d0, f1, d1, hidden, m = dimensions
        self.x = tf.placeholder(tf.float32, shape=(None, n1, n2, d0))  # input
        self.y = tf.placeholder(tf.float32, shape=(None, m))  # target
        # layer 1: conv
        W_1, b_1, z_hat_1, r_hat_1 = layers.convolution_2d(
            name, "layer_1", self.x, f1, d0, d1,
            tf.random_normal_initializer(stddev=1.0 / np.sqrt(f1 * f1 * d0 + 1), seed=SEED),
            gate_fun)
        # layer 1.5: pool
        s_hat_1 = tf.nn.max_pool(
            r_hat_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        shape_1 = s_hat_1.get_shape().as_list()
        y_hat_1 = tf.reshape(s_hat_1, [-1, shape_1[1] * shape_1[2] * shape_1[3]])
        # layer 2: full
        W_2, b_2, z_hat_2, y_hat_2 = layers.fully_connected(
            name, "layer_2", y_hat_1, (n1 * n2 * d1) // 4, hidden,
            tf.random_normal_initializer(stddev=1.0 / np.sqrt((n1 * n2 * d1) // 4),
                                         seed=SEED),
            gate_fun)
        # layer 3: full
        W_3, b_3, z_hat_3, y_hat_3 = layers.fully_connected(
            name, "layer_3", y_hat_2, hidden, m,
            tf.random_normal_initializer(stddev=1.0 / np.sqrt(hidden + 1), seed=SEED),
            tf.nn.softmax)
        # loss
        self.train_loss = tf.reduce_sum(loss_fun(z_hat_3, self.y))
        self.train_vars = [W_1, b_1, W_2, b_2, W_3, b_3]
        self.misclass_err = tf.reduce_sum(tf.cast(
            tf.not_equal(tf.argmax(y_hat_3, 1), tf.argmax(self.y, 1)), tf.float32))


class model_cp_f_fdc():
    """ Define a cp f f model: input -> conv & pool -> 2 x fully connected -> output """
    def __init__(self, name, dimensions, gate_fun, loss_fun, p, SEED=None):
        # placeholders
        # n1 x n2 = image dimensions
        # d0 = number of channels (rgb or greyscale)
        # d1 = number of independent filters
        # f1 = filter size
        n1, n2, d0, f1, d1, hidden, m = dimensions
        self.x = tf.placeholder(tf.float32, shape=(None, n1, n2, d0))  # input
        self.y = tf.placeholder(tf.float32, shape=(None, m))  # target
        # layer 1: conv
        W_1, b_1, z_hat_1, r_hat_1 = layers.convolution_2d(
            name, "layer_1", self.x, f1, d0, d1,
            tf.random_normal_initializer(stddev=1.0 / np.sqrt(f1 * f1 * d0 + 1), seed=SEED),
            gate_fun)
        # layer 1.5: pool
        s_hat_1 = tf.nn.max_pool(
            r_hat_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        shape_1 = s_hat_1.get_shape().as_list()
        y_hat_1 = tf.reshape(s_hat_1, [-1, shape_1[1] * shape_1[2] * shape_1[3]])
        # layer 2: full
        W_2, b_2, z_hat_2, y_hat_2 = layers.fully_connected(
            name, "layer_2", y_hat_1, (n1 * n2 * d1) // 4, hidden,
            tf.random_normal_initializer(stddev=1.0 / np.sqrt((n1 * n2 * d1) // 4),
                                         seed=SEED),
            tf.nn.softmax)
        # layer 3: full
        W_3, MW_3, b_3, mb_3, z_hat_3, y_hat_3 = layers.drop_connect(
            name, "layer_3", y_hat_2, hidden, m,
            tf.random_normal_initializer(stddev=1.0 / np.sqrt(hidden + 1), seed=SEED),
            tf.nn.softmax, p)
        # loss
        self.train_loss = tf.reduce_sum(loss_fun(z_hat_3, self.y))
        self.train_vars = [W_1, b_1, W_2, b_2, tf.multiply(MW_3, W_3), tf.multiply(mb_3, b_3)]
        self.misclass_err = tf.reduce_sum(tf.cast(
            tf.not_equal(tf.argmax(y_hat_3, 1), tf.argmax(self.y, 1)), tf.float32))