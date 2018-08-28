# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import os

import numpy as np
import tensorflow as tf
from scipy import misc
from scipy import ndimage

from src.model import build_model


def main(_):
    model_path = os.path.join("check_point", "model.tfl")
    num = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    model = build_model()
    model.load(model_path)

    img = ndimage.imread("prediction.jpg", mode="RGB")

    # Scale it to 32x32
    img = misc.imresize(img, (32, 32), interp="bicubic").astype(np.float32, casting='unsafe')

    # Predict
    prediction = model.predict([img])

    print(prediction[0])
    print("This is a %s" % (num[prediction[0].tolist().index(max(prediction[0]))]))


if __name__ == '__main__':
    tf.app.run(main=main)
