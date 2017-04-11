
"""Functions for reading MNIST data based on Dales Code"""

import gzip
import os
import numpy as np

def read_revfloat32(bytestream):
  """Read a float32 in reverse byte order"""
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(filename):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth]"""
  print('Extracting', filename)
  with open(filename, 'rb') as f:
    with gzip.GzipFile(fileobj=f) as bytestream:
      magic = read_revfloat32(bytestream)
      if magic != 2051:
        raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                         (magic, filename))
      num_images = read_revfloat32(bytestream)
      rows = read_revfloat32(bytestream)
      cols = read_revfloat32(bytestream)
      buf = bytestream.read(rows * cols * num_images)
      data = np.frombuffer(buf, dtype=np.uint8)
      data = data.reshape(num_images, rows, cols, 1)
      return data


def dense_to_one_hot(labels_dense, num_classes=10):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


def extract_labels(filename, one_hot=False, num_classes=10):
  """Extract the labels into a 1D uint8 numpy array [index]"""
  print('Extracting', filename)
  with open(filename, 'rb') as f:
    with gzip.GzipFile(fileobj=f) as bytestream:
      magic = read_revfloat32(bytestream)
      if magic != 2049:
        raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                         (magic, filename))
      num_items = read_revfloat32(bytestream)
      buf = bytestream.read(num_items)
      labels = np.frombuffer(buf, dtype=np.uint8)
      if one_hot:
        return dense_to_one_hot(labels, num_classes)
      return labels

def read_data(image_data_filename, image_shape=[784], image_range=[0.0, 1.0]):
    """Read MNIST train set and adds some random noise"""
        # train/validation inputs
    if np.prod(image_shape) != 784 or len(image_shape) > 2:
        raise ValueError("Invalid shape specified for MNIST images")
    if len(image_range) != 2 or image_range[0] >= image_range[1]:
        raise ValueError("Invalid range specified for MNIST images")
    Xin_img = extract_images(image_data_filename).astype(np.float32)
    if len(image_shape) == 1:
        Xin = Xin_img.reshape([Xin_img.shape[0]] + image_shape)
    elif len(image_shape) == 2:
        Xin = Xin_img.reshape([Xin_img.shape[0]] + image_shape + [1])
    Xin *= (image_range[1] - image_range[0]) / np.max(Xin)
    Xin += image_range[0]
    return Xin

