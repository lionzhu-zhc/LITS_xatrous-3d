#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2018/10/17 15:59
# @Author  : LionZhu

# this is a SuperResolution CNN based on Resnet pretrained model

import tensorflow as tf
from tensorflow.contrib.slim import nets
slim = tf.contrib.slim

ResModel = 'resnet_v2_50'

def ResSR(inputs, is_training, IMGCHANNEL):
    with slim.arg_scope(nets.resnet_v2.resnet_arg_scope()):
        _, end_points = nets.resnet_v2.resnet_v2_50(inputs, num_classes=None, is_training= is_training, output_stride= 32)
    with tf.variable_scope('ResSR'):
        net = end_points[ResModel + '/block2']   #512 channels
        #net = slim.conv2d(net, IMGCHANNEL, [1,1], activation_fn = None, normalizer_fn= None)

        net = tf.layers.batch_normalization(net, momentum= 0.9, training= is_training, name='BN1')
        net = tf.layers.conv2d_transpose(net, filters= 128, kernel_size= 3, strides= [4,4],
                                         activation= tf.nn.relu, kernel_initializer= tf.contrib.layers.xavier_initializer(), name= 'Deconv1')
        net = tf.layers.batch_normalization(net, momentum=0.9, training=is_training, name='BN2')
        net = tf.layers.conv2d_transpose(net, filters=128, kernel_size=3, strides=[4, 4],
                                         activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Deconv2')
        net = tf.layers.conv2d(net, filters= 30, kernel_size= 1, strides= 1, padding= 'SAME',
                               activation= None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Last')

        return net


# def conv2d_transpose(k_size, in_put, out_shape, out_channel, c_stride, name= None):
#     shap = in_put.get_shape().as_list()
#     in_channel = shap[-1]
#     with tf.variable_scope(name):
#         pass
#
# def get_var_transpose(k_size, in_channel, out_channel, name = None):
#     weights = tf.get_variable('w', shape= [k_size, k_size, out_channel, in_channel],
#                               initializer= tf.contrib.layers.xavier_initializer())
#     bias = tf.get_variable