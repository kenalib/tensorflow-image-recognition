# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import os
import pickle
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.lib.io import file_io
from tflearn.data_utils import shuffle, to_categorical

from src.model import build_model


def load_data(dir_name, one_hot=False):
    x_train = []
    y_train = []

    for i in range(1, 6):
        filename = os.path.join(dir_name, 'data_batch_' + str(i))
        data, labels = load_batch(filename)
        if i == 1:
            x_train = data
            y_train = labels
        else:
            x_train = np.concatenate([x_train, data], axis=0)
            y_train = np.concatenate([y_train, labels], axis=0)

    filename = os.path.join(dir_name, 'test_batch')
    x_test, y_test = load_batch(filename)

    x_train = np.dstack(
        (x_train[:, :1024], x_train[:, 1024:2048], x_train[:, 2048:])
    ) / 255.
    x_train = np.reshape(x_train, [-1, 32, 32, 3])

    x_test = np.dstack(
        (x_test[:, :1024], x_test[:, 1024:2048], x_test[:, 2048:])
    ) / 255.
    x_test = np.reshape(x_test, [-1, 32, 32, 3])

    if one_hot:
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)


def load_batch(filename):
    obj = file_io.read_file_to_string(filename, binary_mode=True)

    if sys.version_info > (3, 0):  # Python3
        d = pickle.loads(obj, encoding='latin1')
    else:  # Python2
        d = pickle.loads(obj)

    return d["data"], d["labels"]


def main(_):
    epoch = 1
    data_dir = "cifar-10-batches-py"
    (x, y), (x_test, y_test) = load_data(data_dir)
    print("load data done")

    x, y = shuffle(x, y)
    y = to_categorical(y, 10)
    y_test = to_categorical(y_test, 10)

    # Train using classifier
    model = build_model()
    model.fit(x, y, n_epoch=epoch, shuffle=True, validation_set=(x_test, y_test),
              show_metric=True, batch_size=96, run_id='cifar10_cnn')

    model_path = os.path.join("check_point", "model.tfl")
    model.save(model_path)


if __name__ == '__main__':
    tf.app.run(main=main)
