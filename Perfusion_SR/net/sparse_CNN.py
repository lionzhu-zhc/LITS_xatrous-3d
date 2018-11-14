#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2018/11/14 10:42
# @Author  : LionZhu
# this is a CNN using conv and relu to learn residual
# ref: View-interpolation of sparsely sampled sinogram using convolutional neural network

import tensorflow as tf

def build_sparseCNN(tensor_in, BN_FLAG, KEEPPROB, IMGCHANNEL):
    res = tensor_in
    for i in range(20):
        if i != 19:
            out_channel = 64
        else:
            out_channel = 30
        res = tf.layers.conv2d(res, filters= out_channel, kernel_size= 5, strides= 1, padding= 'same', activation= tf.nn.relu,
                               kernel_initializer= tf.initializers.truncated_normal(), name= 'Layer_{}'.format(i))

    return res
    #res is the residual of img and annotation