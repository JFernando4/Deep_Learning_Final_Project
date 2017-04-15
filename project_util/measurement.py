
""" Simple framework for capturing various measurements recorded
    during an experiment. Copied from Dale's Code for Assignment 1 """


import tensorflow as tf
import numpy as np
import time


class measure():

  def __init__(self, label, axes=[0.0, np.inf, 0.0, np.inf]):
    self.label = label
    self.axes = axes

  def update(self):
    None

  def reset(self):
    None

  def printout(self, sess, echo):
    val = self.eval(sess)
    if echo:
      print("%s %g" % (self.label, val), "\t")
    return val


class meas_loss(measure):

  def __init__(self, x, y, data, loss, label,
               batch=100, axes=[0.0, np.inf, 0.0, np.inf]):
    self.x = x
    self.y = y
    self.data = data
    self.loss = loss
    self.label = label
    self.batch = batch
    self.axes = axes

  def eval(self, sess):
    loss_total = 0.0
    X, Y = self.data
    end = X.shape[0]
    cur = 0
    while cur < end:
      inds = range(cur, min(cur + self.batch, end))
      loss_total += sess.run(self.loss, feed_dict={self.x: X[inds],
                                                   self.y: Y[inds]})
      cur += self.batch
    return loss_total


class meas_avg_euclid_dist_loss(measure):

    def __init__(self, x, y, data, data_mean, data_sd, model_output, label,
                 batch=100, axes = [0.0, np.inf, 0.0, np.inf]):
        self.x = x
        self.y = y
        self.data = data
        self.data_mean = data_mean
        self.data_sd = data_sd
        self.model_output = model_output
        self.label = label
        self.batch = batch
        self.axes = axes

    def eval(self, sess):
        loss_total = 0
        X, Y = self.data
        means = self.data_mean
        sds = self.data_sd
        modified_X = self.add_noise(X, means, sds)
        end = X.shape[0]
        cur = 0
        while cur < end:
            inds = range(cur, min(cur + self.batch, end))
            model_predict = sess.run(self.model_output, feed_dict={self.x: X[inds],
                                                           self.y: Y[inds]})
            modified_predict = sess.run(self.model_output, feed_dict={self.x: modified_X[inds],
                                                              self.y: Y[inds]})
            loss_total += self.euclid_dist(model_predict, modified_predict)
            cur += self.batch
        return loss_total

    def add_noise(self, X, X_mean, X_sd, pixels=1):
        n, max_idx = X.shape # Total number of observations and pixels in a flattened image
        # random_noise = np.abs(np.transpose(np.random.normal(X_mean, X_sd, [pixels, n])))
        random_indices = np.random.choice(max_idx, [n, pixels])
        noisy_X = np.add(np.zeros(X.shape, dtype=X.dtype), X)
        for i in range(0, n):
            noisy_X[i][random_indices[i]] = 0#random_noise[i]
        return noisy_X

    def euclid_dist(self, x, y): # x and y must have the same shape
        xy_diff = np.subtract(x,y)
        sqrd_diff = np.multiply(xy_diff, xy_diff)
        col_sum = np.sum(sqrd_diff, 1)
        distance = np.sum(np.sqrt(col_sum))
        return distance




class meas_time(measure):

  def __init__(self, label, axes=[0.0, np.inf, 0.0, np.inf]):
    self.label = label
    self.axes = axes
    self.start_time = None

  def update(self):
    if self.start_time == None:
      self.start_time = time.time()

  def reset(self):
    self.start_time = None

  def eval(self, sess):
    return 0 if self.start_time == None else time.time() - self.start_time


class meas_iter(measure):

  def __init__(self, gap, label, axes=[0.0, np.inf, 0.0, np.inf]):
    self.gap = gap
    self.label = label
    self.axes = axes
    self.iter = None

  def update(self):
    if self.iter == None:
      self.iter = 0
    else:
      self.iter += self.gap

  def reset(self):
    self.iter = None

  def eval(self, sess):
    return self.iter


# general

def reset(meas_list):
  for meas in meas_list:
    meas.reset()

def update(meas_list):
  for meas in meas_list:
    meas.update()

def printout(label, meas_list, sess, echo):
  results = np.zeros(len(meas_list))
  i = 0
  if echo:
    print(label, "\t")
  for meas in meas_list:
    results[i] = meas.printout(sess, echo)
    i += 1
  if echo:
    print("\t")
  return results