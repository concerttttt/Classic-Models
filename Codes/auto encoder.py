# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 08:00:08 2017

@author: LI YUXIN
"""

import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def xavier_init(data_in,data_out,constant = 1):
    low = -constant * np.sqrt(6.0 / (data_in + data_out))
    high = - low
    return tf.random_uniform((data_in,data_out),
                             minval = low,
                             maxval = high,
                             dtype = tf.float32)
                             
class AdditiveGaussianNoiseAutoencoder:
    pass


    