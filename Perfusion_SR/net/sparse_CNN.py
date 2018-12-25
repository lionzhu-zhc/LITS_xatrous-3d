#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2018/11/14 10:42
# @Author  : LionZhu
# this is a CNN using conv and relu to learn residual
# ref: View-interpolation of sparsely sampled sinogram using convolutional neural network

import tensorflow as tf

def build_sparseCNN(tensor_in, BN_FLAG, KEEPPROB, IMGCHANNEL):
    res = tensor_in
    shap= tf.shape(tensor_in)
    for i in range(5):
        if i != 4:
            out_channel = 32
            res = tf.layers.conv3d(res, filters= out_channel, kernel_size= 3, strides= [1,1,1], padding= 'same', activation= tf.nn.relu,
                                   kernel_initializer= tf.initializers.truncated_normal(), name= 'Layer_{}'.format(i))
        else:
            out_channel = 1
            res = tf.layers.conv3d(res, filters=out_channel, kernel_size=3, strides=[1,1,1], padding='same',
                                   activation= None,
                                   kernel_initializer=tf.initializers.truncated_normal(), name='Layer_{}'.format(i))

        # res = tf.layers.conv2d(res, filters= out_channel, kernel_size= 5, strides= 1, padding= 'same', activation= tf.nn.relu,
        #                        kernel_initializer= tf.initializers.truncated_normal(), name= 'Layer_{}'.format(i))



    # out_shape = tf.stack([shap[0], 2*shap[1], shap[2], shap[3], shap[4]])
    # res = conv3d_transpose_layer(3, res, out_shape= out_shape, in_channel= 32, out_channel= 1, c_stride= 2, name='Deconv')

    return res
    #res is the residual of img and annotation


def conv3d_transpose_layer(kernel_size, in_put, out_shape, in_channel, out_channel, c_stride, name):
    with tf.variable_scope(name):
        deconv_weights, deconv_bias = get_var_transpose(kernel_size, in_channel, out_channel, name)
        deconv = tf.nn.conv3d_transpose(in_put, deconv_weights, out_shape, [1, c_stride, c_stride, c_stride, 1],
                                        padding='SAME')
        deconv_addbias = tf.nn.bias_add(deconv, deconv_bias)
        return deconv_addbias

def get_var_transpose(kernel_size, in_channel, out_channel, name=None):
    weights = tf.get_variable('w', shape=[kernel_size, kernel_size, kernel_size, out_channel, in_channel],
                              initializer=tf.contrib.layers.xavier_initializer())

    bias = tf.get_variable('b', [out_channel],
                           initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))
    return weights, bias
