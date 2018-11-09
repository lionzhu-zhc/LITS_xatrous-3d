#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2018/10/13 15:21
# @Author  : LionZhu

# a simple super resolution CNN called SRCNN
#https://blog.csdn.net/Autism_/article/details/79401798

import tensorflow as tf

def build_srcnn(tensor_in, BN_FLAG, KEEPPROB, IMGCHANNEL):
    '''

    :param tensor_in:
    :param BN_FLAG:
    :param BATCHSIZE:
    :param CLASSNUM:
    :param KEEPPROB:
    :param IMGCHANNEL:
    :return:
    '''

    # the first layer
    with tf.variable_scope('layer_1'):
        res = tf.layers.batch_normalization(tensor_in, momentum=0.95, training=BN_FLAG, name='BN')
        res = conv_layer(9, res, out_channel= 64, c_stride=1, keep_prob= KEEPPROB, name= 'conv')
        layer1_res = tf.nn.relu(res)

    with tf.variable_scope('layer_2'):
        res = tf.layers.batch_normalization(layer1_res, momentum=0.95, training=BN_FLAG, name='BN')
        res = conv_layer(1, res, out_channel= 32, c_stride=1, keep_prob= KEEPPROB, name= 'conv')
        layer2_res = tf.nn.relu(res)

    with tf.variable_scope('layer_3'):
        res = tf.layers.batch_normalization(layer2_res, momentum=0.95, training=BN_FLAG, name='BN')
        res = conv_layer(3, res, out_channel= 32, c_stride=1, keep_prob= KEEPPROB, pad= 'SAME', name= 'conv')
        layer3_res = tf.nn.relu(res)

    with tf.variable_scope('layer_4'):
        res = tf.layers.batch_normalization(layer3_res, momentum=0.95, training=BN_FLAG, name='BN')
        res = conv_layer(3, res, out_channel= 32, c_stride=1, keep_prob= KEEPPROB, pad= 'SAME', name= 'conv')
        res = tf.nn.relu(res)

    with tf.variable_scope('layer_10'):
        conv = conv_layer(5, res, out_channel= IMGCHANNEL, c_stride=1, keep_prob= KEEPPROB, name= 'conv')
        # bn_res = tf.layers.batch_normalization(conv, momentum= 0.95, training= BN_FLAG, name= 'BN')
        # layer3_res = tf.nn.relu(bn_res)

    return conv


def conv_layer(k_size, in_put, out_channel, c_stride= 1, keep_prob= 0.8, pad= 'VALID',name= None):
    shap = in_put.get_shape().as_list()
    in_channel = shap[-1]
    with tf.variable_scope(name):
        conv_weights, conv_bias = get_var(k_size, in_channel, out_channel, name)
        conv = tf.nn.conv2d(in_put, conv_weights, strides=[1, c_stride, c_stride, 1], padding= pad)
        conv_addbias = tf.nn.bias_add(conv, conv_bias)
        conv_addbias = tf.nn.dropout(conv_addbias, keep_prob)
        return conv_addbias

def get_var(k_size, in_channel, out_channel, name=None):
    weights = tf.get_variable('w', shape=[k_size, k_size, in_channel, out_channel],
                              initializer=tf.contrib.layers.xavier_initializer())
    bias = tf.get_variable('b', [out_channel], initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))

    return weights, bias