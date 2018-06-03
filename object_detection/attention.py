#!/usr/bin/env python
#-*- coding: utf-8 -*-

# File Name: layers.py
# Author: Yanmei Fu
# mail: yanmei2016@iscas.ac.cn
# Created Time: 2018-06-03


import tensorflow as tf
from layers import dense
from mask_utils import Mask

# sequence attention
def general_attention(name, input, atn_hidden_size):
    """

    :param name:  variable scope name
    :param input: [batch_size, max_time,  hidden_size]
    :return:
    """
    with tf.variable_scope(name):
        # [batch_size, max_time, 1,  hidden_size]
        hidden_size = input.get_shape().as_list()[2]
        atn_in = tf.expand_dims(input, axis=2)
        atn_w = tf.Variable(tf.truncated_normal(shape=[1, 1, hidden_size, atn_hidden_size], stddev=0.1), name="atn_w")
        atn_b = tf.Variable(tf.zeros(shape=[atn_hidden_size]), name="atn_b")
        atn_v = tf.Variable(tf.truncated_normal([1, 1, atn_hidden_size, 1], stddev=0.1),name="atn_v")
        atn_activations = tf.nn.tanh(tf.nn.conv2d(atn_in, atn_w, strides=[1, 1, 1, 1], padding='SAME') + atn_b)
        atn_scores = tf.nn.conv2d(atn_activations, atn_v, strides=[1, 1, 1, 1], padding='SAME')
        atn_probs = tf.nn.softmax(tf.squeeze(atn_scores, [2,3]))
        atn_out = tf.matmul(tf.expand_dims(atn_probs, 1), input)
        out = tf.squeeze(atn_out, [1], name="atn_out")
        return out


def multi_head_attention(name,  Q, K, V, nb_head =8, size_per_head=16, Q_len=None, V_len=None):

    ## linear mapping

    Q = dense(name=name+"q_linear", x=Q, output_dim=nb_head * size_per_head)
    Q = tf.reshape(Q, (-1, tf.shape(Q)[1], nb_head, size_per_head))
    ## [ batch_size, feature_size, nb_head, size_per_head]  -> [batch_size, nb_head, feature_size, size_per_head]
    Q = tf.transpose(Q, [0, 2, 1, 3])

    K = dense(name=name + "K_linear", x=K, output_dim=nb_head * size_per_head)
    K = tf.reshape(K, (-1, tf.shape(K)[1], nb_head, size_per_head))
    ## [ batch_size, feature_size, nb_head, size_per_head]  -> [batch_size, nb_head, feature_size, size_per_head]
    K = tf.transpose(K, [0, 2, 1, 3])

    V = dense(name=name + "V_linear", x=V, output_dim=nb_head * size_per_head)
    V = tf.reshape(V, (-1, tf.shape(V)[1], nb_head, size_per_head))
    ## [ batch_size, feature_size, nb_head, size_per_head]  -> [batch_size, nb_head, feature_size, size_per_head]
    V = tf.transpose(V, [0, 2, 1, 3])


    A = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(float(size_per_head))

    # [batch_size, nb_head, feature_size, size_per_head] -> [batch_size, size_per_head, feature_size, nb_head]
    A = tf.transpose(A, [0, 3, 2, 1])
    A = Mask(A, V_len, mode='add')

    # [batch_size, size_per_head, feature_size, nb_head] ->[batch_size, nb_head, feature_size, size_per_head]
    A = tf.transpose(A, [0, 3, 2, 1])
    A = tf.nn.softmax(A)

    O = tf.matmul(A, V)
    #[batch_size, nb_head, feature_size, size_per_head]  -> [batch_size, feature_size, nb_head, size_per_head]
    O = tf.transpose(O, [0, 2, 1, 3])

    O = tf.reshape(O, (-1, tf.shape(O)[1], nb_head * size_per_head))
    O = Mask(O, Q_len, 'mul')

    return O












