# Adversarial Anomaly Detection
# - Utility functions
#
# Author: Seongho Baek
# Contact: seonghobaek@gmail.com
#
# ==============================================================================


import math
import numpy as np
import tensorflow as tf


def get_batch(X, X_, size):
    # X, X_ must be nd-array
    a = np.random.choice(len(X), size, replace=False)
    return X[a], X_[a]


def get_sequence_batch(X, seq_length, batch_size):
    # print('input dim:', len(X[0]), ', seq len:', seq_length, ', batch_size:', batch_size)
    # X must be nd-array
    a = np.random.choice(len(X)-seq_length, batch_size, replace=False)
    a = a + seq_length

    # print('index: ', a)

    seq = []

    for i in range(batch_size):
        if a[i] < seq_length - 1:
            s = np.random.normal(0.0, 0.1, [seq_length, len(X[0])])
            seq.append(s)
        else:
            s = np.arange(a[i]-seq_length, a[i])
            seq.append(X[s])

    seq = np.array(seq)

    return X[a], seq


def noise_validator(noise, allowed_noises):
    '''Validates the noise provided'''
    try:
        if noise in allowed_noises:
            return True
        elif noise.split('-')[0] == 'mask' and float(noise.split('-')[1]):
            t = float(noise.split('-')[1])
            if t >= 0.0 and t <= 1.0:
                return True
            else:
                return False
    except:
        return False
    pass


def sigmoid_normalize(value_list):
    list_max = float(max(value_list))
    alist = [i/list_max for i in value_list]
    alist = [1/(1+math.exp(-i)) for i in alist]

    return alist


def swish(logit,  name=None):
    with tf.name_scope(name):
        l = tf.multiply(logit, tf.nn.sigmoid(logit))

        return l


def generate_samples(dim, num_inlier, num_outlier, normalize=True):
    inlier = np.random.normal(0.0, 1.0, [num_inlier, dim])

    sample_inlier = []

    if normalize:
        inlier = np.transpose(inlier)

        for values in inlier:
            values = sigmoid_normalize(values)
            sample_inlier.append(values)

        inlier = np.array(sample_inlier).transpose()

    outlier = np.random.normal(1.0, 1.0, [num_outlier, dim])

    sample_outlier = []

    if normalize:
        outlier = np.transpose(outlier)

        for values in outlier:
            values = sigmoid_normalize(values)
            sample_outlier.append(values)

        outlier = np.array(sample_outlier).transpose()

    return inlier, outlier


def add_gaussian_noise(input_layer, mean, std, truncate=True):
    if std < 0.0:
        return input_layer

    if truncate:
        noise = tf.truncated_normal(shape=tf.shape(input_layer), mean=mean, stddev=std, dtype=tf.float32)
    else:
        noise = tf.random_normal(shape=tf.shape(input_layer), mean=mean, stddev=std, dtype=tf.float32)

    return input_layer + noise


def add_uniform_noise(input_layer, min_val, max_val):
    return input_layer + tf.random_uniform(shape=tf.shape(input_layer), minval=min_val, maxval=max_val)
