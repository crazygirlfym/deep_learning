#!/usr/bin/env python
#-*- coding: utf-8 -*-

# File Name: layers.py
# Author: Yanmei Fu
# mail: yanmei2016@iscas.ac.cn
# Created Time: 2018-05-27

## description : some common function of cnn layers

import tensorflow as tf
import numpy as np

from layer_utils import __variable_with_weight_decay
from layer_utils import __variable_summaries

def __conv2d_p(name, x, w=None, num_filters=16, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
               initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0):
    """
    Convolution 2D Wrapper
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, H, W, C).
    :param w: (tf.tensor) pretrained weights (if None, it means no pretrained weights)
    :param num_filters: (integer) No. of filters (This is the output depth)
    :param kernel_size: (integer tuple) The size of the convolving kernel.
    :param padding: (string) The amount of padding required.
    :param stride: (integer tuple) The stride required.
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias. (if not float, it means pretrained bias)
    :return out: The output of the layer. (N, H', W', num_filters)
    """
    with tf.variable_scope(name):
        stride = [1, stride[0], stride[1], 1]
        kernel_shape = [kernel_size[0], kernel_size[1], x.shape[-1], num_filters]

        with tf.name_scope('layer_weights'):
            if w == None:
                w = __variable_with_weight_decay(kernel_shape, initializer, l2_strength)
            __variable_summaries(w)
        with tf.name_scope('layer_biases'):
            if isinstance(bias, float):
                bias = tf.get_variable('biases', [num_filters], initializer=tf.constant_initializer(bias))
            __variable_summaries(bias)
        with tf.name_scope('layer_conv2d'):
            conv = tf.nn.conv2d(x, w, stride, padding)
            out = tf.nn.bias_add(conv, bias)

    return out


def conv2d(name, x, w=None, num_filters=16, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
           initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0,
           activation=None, batchnorm_enabled=False, max_pool_enabled=False, dropout_keep_prob=-1,
           is_training=True):
    """
    This block is responsible for a convolution 2D layer followed by optional (non-linearity, dropout, max-pooling).
    Note that: "is_training" should be passed by a correct value based on being in either training or testing.
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, H, W, C).
    :param num_filters: (integer) No. of filters (This is the output depth)
    :param kernel_size: (integer tuple) The size of the convolving kernel.
    :param padding: (string) The amount of padding required.
    :param stride: (integer tuple) The stride required.
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias.
    :param activation: (tf.graph operator) The activation function applied after the convolution operation. If None, linear is applied.
    :param batchnorm_enabled: (boolean) for enabling batch normalization.
    :param max_pool_enabled:  (boolean) for enabling max-pooling 2x2 to decrease width and height by a factor of 2.
    :param dropout_keep_prob: (float) for the probability of keeping neurons. If equals -1, it means no dropout
    :param is_training: (boolean) to diff. between training and testing (important for batch normalization and dropout)
    :return: The output tensor of the layer (N, H', W', C').
    """

    with tf.variable_scope(name) as scope:
        conv_o_b = __conv2d_p('conv', x=x, w=w, num_filters=num_filters, kernel_size=kernel_size, stride=stride,
                              padding=padding,
                              initializer=initializer, l2_strength=l2_strength, bias=bias)

        if batchnorm_enabled:
            conv_o_bn = tf.layers.batch_normalization(conv_o_b, training=is_training, epsilon=1e-5)
            if not activation:
                conv_a = conv_o_bn
            else:
                conv_a = activation(conv_o_bn)
        else:
            if not activation:
                conv_a = conv_o_b
            else:
                conv_a = activation(conv_o_b)

        def dropout_with_keep():
            return tf.nn.dropout(conv_a, dropout_keep_prob)

        def dropout_no_keep():
            return tf.nn.dropout(conv_a, 1.0)

        if dropout_keep_prob != -1:
            conv_o_dr = tf.cond(is_training, dropout_with_keep, dropout_no_keep)
        else:
            conv_o_dr = conv_a

        conv_o = conv_o_dr
        if max_pool_enabled:
            conv_o = max_pool_2d(conv_o_dr)

    return conv_o



def max_pool_2d(x, size=(2, 2), stride=(2, 2), name='pooling'):
    """
    Max pooling 2D Wrapper
    :param x: (tf.tensor) The input to the layer (N,H,W,C).
    :param size: (tuple) This specifies the size of the filter as well as the stride.
    :param name: (string) Scope name.
    :return: The output is the same input but halfed in both width and height (N,H/2,W/2,C).
    """
    size_x, size_y = size
    stride_x, stride_y = stride
    return tf.nn.max_pool(x, ksize=[1, size_x, size_y, 1], strides=[1, stride_x, stride_y, 1], padding='VALID',
                          name=name)


def avg_pool_2d(x, size=(2, 2), stride=(2, 2), name='avg_pooling', padding='VALID'):
    """
        Average pooling 2D Wrapper
        :param x: (tf.tensor) The input to the layer (N,H,W,C).
        :param size: (tuple) This specifies the size of the filter as well as the stride.
        :param name: (string) Scope name.
        :return: The output is the same input but halfed in both width and height (N,H/2,W/2,C).
    """
    size_x, size_y = size
    stride_x, stride_y = stride
    return tf.nn.avg_pool(x, ksize=[1, size_x, size_y, 1], strides=[1, stride_x, stride_y, 1], padding=padding, name=name)


def flatten(x):
    """
    Flatten a (N,H,W,C) input into (N,D) output. Used for fully connected layers after conolution layers
    :param x: (tf.tensor) representing input
    :return: flattened output
    """
    all_dims_exc_first = np.prod([v.value for v in x.get_shape()[1:]])
    o = tf.reshape(x, [-1, all_dims_exc_first])
    return o



def __dense_p(name, x, w=None, output_dim=128, initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0,
              bias=0.0):
    """
    Fully connected layer
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, D).
    :param output_dim: (integer) It specifies H, the output second dimension of the fully connected layer [ie:(N, H)]
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias. (if not float, it means pretrained bias)
    :return out: The output of the layer. (N, H)
    """
    n_in = x.get_shape()[-1].value
    with tf.variable_scope(name):
        if w == None:
            w = __variable_with_weight_decay([n_in, output_dim], initializer, l2_strength)
        __variable_summaries(w)
        if isinstance(bias, float):
            bias = tf.get_variable("layer_biases", [output_dim], tf.float32, tf.constant_initializer(bias))
        __variable_summaries(bias)
        output = tf.nn.bias_add(tf.matmul(x, w), bias)
        return output


def dense(name, x, w=None, output_dim=128, initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0,
          bias=0.0,
          activation=None, batchnorm_enabled=False, dropout_keep_prob=-1,
          is_training=True
          ):
    """
    This block is responsible for a fully connected followed by optional (non-linearity, dropout, max-pooling).
    Note that: "is_training" should be passed by a correct value based on being in either training or testing.
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, D).
    :param output_dim: (integer) It specifies H, the output second dimension of the fully connected layer [ie:(N, H)]
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias.
    :param activation: (tf.graph operator) The activation function applied after the convolution operation. If None, linear is applied.
    :param batchnorm_enabled: (boolean) for enabling batch normalization.
    :param dropout_keep_prob: (float) for the probability of keeping neurons. If equals -1, it means no dropout
    :param is_training: (boolean) to diff. between training and testing (important for batch normalization and dropout)
    :return out: The output of the layer. (N, H)
    """
    with tf.variable_scope(name) as scope:
        dense_o_b = __dense_p(name='dense', x=x, w=w, output_dim=output_dim, initializer=initializer,
                              l2_strength=l2_strength,
                              bias=bias)

        if batchnorm_enabled:
            dense_o_bn = tf.layers.batch_normalization(dense_o_b, training=is_training, epsilon=1e-5)
            if not activation:
                dense_a = dense_o_bn
            else:
                dense_a = activation(dense_o_bn)
        else:
            if not activation:
                dense_a = dense_o_b
            else:
                dense_a = activation(dense_o_b)

        def dropout_with_keep():
            return tf.nn.dropout(dense_a, dropout_keep_prob)

        def dropout_no_keep():
            return tf.nn.dropout(dense_a, 1.0)

        if dropout_keep_prob != -1:
            dense_o_dr = tf.cond(is_training, dropout_with_keep, dropout_no_keep)
        else:
            dense_o_dr = dense_a

        dense_o = dense_o_dr
    return dense_o


def flatten(x):
    """
    Flatten a (N,H,W,C) input into (N,D) output. Used for fully connected layers after conolution layers
    :param x: (tf.tensor) representing input
    :return: flattened output
    """
    all_dims_exc_first = np.prod([v.value for v in x.get_shape()[1:]])
    o = tf.reshape(x, [-1, all_dims_exc_first])
    return o

def scala_transfer(x, r):
    bsize, a, b, c = x.get_shape().as_list()
    X = tf.reshape(x, (bsize, a, b, c//(r*r), r, r))
    X = tf.transpose(X, (0, 1, 2, 5, 4, 3))
    X = [X[:, i, :, :, :, :] for i in range(a)] # a, [bsize, b, r, r, c/(r*r)
    X = tf.concat(X, 2)  # bsize, b, a*r, r, c/(r*r)
    X = [X[:, i, :, :, :] for i in range(b)]  # b, [bsize, r, r, c/(r*r)
    X = tf.concat(X, 2) # bsize, a*r, b*r, c/(r*r)
    return X

def __depthwise_conv2d_p(name, x, w=None, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
                         initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0):
    with tf.variable_scope(name):
        stride = [1, stride[0], stride[1], 1]
        kernel_shape = [kernel_size[0], kernel_size[1], x.shape[-1], 1]

        with tf.name_scope('layer_weights'):
            if w is None:
                w = __variable_with_weight_decay(kernel_shape, initializer, l2_strength)
            __variable_summaries(w)
        with tf.name_scope('layer_biases'):
            if isinstance(bias, float):
                bias = tf.get_variable('biases', [x.shape[-1]], initializer=tf.constant_initializer(bias))
            __variable_summaries(bias)
        with tf.name_scope('layer_conv2d'):
            conv = tf.nn.depthwise_conv2d(x, w, stride, padding)
            out = tf.nn.bias_add(conv, bias)

    return out


def depthwise_conv2d(name, x, w=None, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
                     initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0, activation=None,
                     batchnorm_enabled=False, is_training=True):

    with tf.variable_scope(name) as scope:
        conv_o_b = __depthwise_conv2d_p(name='conv', x=x, w=w, kernel_size=kernel_size, padding=padding,
                                        stride=stride, initializer=initializer, l2_strength=l2_strength, bias=bias)

        if batchnorm_enabled:
            conv_o_bn = tf.layers.batch_normalization(conv_o_b, training=is_training, epsilon=1e-5)
            if not activation:
                conv_a = conv_o_bn
            else:
                conv_a = activation(conv_o_bn)
        else:
            if not activation:
                conv_a = conv_o_b
            else:
                conv_a = activation(conv_o_b)
    return conv_a


def channel_shuffle(name, x, num_groups):

    ## implemention of "ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices"
    ## 解释下： 分组做卷积操作，假设有M个filter 和N 个feature map 做卷积，然后相加作为一个卷积的结果， 当引入group操作的时候
    ## 将M 个filter 和N个feature map分成g个group， 做卷积的时候， 第一个group 和M/g 个filter 的每一个都和第一个group的N/g个输入做卷积
    ## 在这种操作下， 某个输出的channel仅仅来自于输入channel的一小部分，会有边界效应, 因此引入了不同组之间进行channel shuffle
    with tf.variable_scope(name) as scope:
        n, h, w, c = x.shape.as_list()
        x_reshaped = tf.reshape(x, [-1, h, w, num_groups, c // num_groups])
        x_transposed = tf.transpose(x_reshaped, [0, 1, 2, 4, 3])
        output = tf.reshape(x_transposed, [-1, h, w, c])
        return output


def grouped_conv2d(name, x, w=None, num_filters=16, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
                   initializer=tf.contrib.layers.xavier_initializer(), num_groups=1, l2_strength=0.0, bias=0.0,
                   activation=None, batchnorm_enabled=False, dropout_keep_prob=-1,
                   is_training=True):
    with tf.variable_scope(name) as scope:
        sz = x.get_shape()[3].value // num_groups
        conv_side_layers = [
            conv2d(name + "_" + str(i), x[:, :, :, i * sz:i * sz + sz], w, num_filters // num_groups, kernel_size,
                   padding,
                   stride,
                   initializer,
                   l2_strength, bias, activation=None,
                   batchnorm_enabled=False, max_pool_enabled=False, dropout_keep_prob=dropout_keep_prob,
                   is_training=is_training) for i in
            range(num_groups)]
        conv_g = tf.concat(conv_side_layers, axis=-1)

        if batchnorm_enabled:
            conv_o_bn = tf.layers.batch_normalization(conv_g, training=is_training, epsilon=1e-5)
            if not activation:
                conv_a = conv_o_bn
            else:
                conv_a = activation(conv_o_bn)
        else:
            if not activation:
                conv_a = conv_g
            else:
                conv_a = activation(conv_g)

        return conv_a




