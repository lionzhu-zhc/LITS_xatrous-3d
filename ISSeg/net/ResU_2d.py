#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2018/10/25 9:10
# @Author  : LionZhu

# this network is similar with NetDeepLab_2d but with pretrained resnet in first layers

import tensorflow as tf
from tensorflow.contrib.slim import nets
slim = tf.contrib.slim

ResModel = 'resnet_v2_50'
Initializer = tf.contrib.layers.variance_scaling_initializer()

def resU_2d(tensor_in, BN_FLAG, CLASSNUM):
    with slim.arg_scope(nets.resnet_v2.resnet_arg_scope()):
        _, end_points = nets.resnet_v2.resnet_v2_50(tensor_in, num_classes=None, is_training=BN_FLAG, output_stride= 32)
        resnet = end_points[ResModel + '/block4']
    with tf.variable_scope('NewDeepLab'):
        with tf.variable_scope('ASPP'):
            with tf.variable_scope('Branch1'):
                br1 = tf.layers.conv2d(resnet, filters= 128, kernel_size= 1, padding= 'same', activation= None,
                                       kernel_initializer= Initializer, name='conv')
                br1 = tf.layers.batch_normalization(br1, momentum=0.9, training=BN_FLAG, name='BN')
                br1 = tf.nn.relu(br1)

            with tf.variable_scope('Branch2'):
                br2 = tf.layers.conv2d(resnet, filters= 128, kernel_size= 1, padding= 'same', activation= None,
                                       kernel_initializer= Initializer, name='conv')
                br2 = tf.layers.batch_normalization(br2, momentum=0.9, training=BN_FLAG, name='BN')
                br2 = tf.nn.relu(br2)
                br2 = tf.layers.conv2d(br2, filters= 128, kernel_size= 3, strides= 1, padding= 'same', dilation_rate= 2,
                                       activation= None, kernel_initializer= Initializer, name= 'conv2')
                br2 = tf.layers.batch_normalization(br2, momentum= 0.9, training= BN_FLAG, name='BN2')
                br2 = tf.nn.relu(br2)

            with tf.variable_scope('Branch3'):
                br3 = tf.layers.conv2d(resnet, filters= 128, kernel_size= 1, padding= 'same', activation= None,
                                       kernel_initializer= Initializer, name='conv')
                br3 = tf.layers.batch_normalization(br3, momentum= 0.9, training= BN_FLAG, name= 'BN')
                br3 = tf.nn.relu(br3)
                br3 = tf.layers.conv2d(br3, filters= 128, kernel_size= 3, strides= 1, padding= 'same', dilation_rate= 4,
                                       activation= None, kernel_initializer= Initializer, name= 'conv2')
                br3 = tf.layers.batch_normalization(br3, momentum= 0.9, training= BN_FLAG, name='BN2')
                br3 = tf.nn.relu(br3)

            with tf.variable_scope('Branch4'):
                br4 = tf.layers.conv2d(resnet, filters= 128, kernel_size= 1, padding= 'same', activation= None,
                                       kernel_initializer= Initializer, name='conv')
                br4 = tf.layers.batch_normalization(br4, momentum= 0.9, training= BN_FLAG, name= 'BN')
                br4 = tf.nn.relu(br4)
                br4 = tf.layers.conv2d(br4, filters= 128, kernel_size= 3, strides= 1, padding= 'same', dilation_rate= 8,
                                       activation= None, kernel_initializer= Initializer, name= 'conv2')
                br4 = tf.layers.batch_normalization(br4, momentum= 0.9, training= BN_FLAG, name='BN2')
                br4 = tf.nn.relu(br4)

        with tf.variable_scope('Concat'):
            concat = tf.concat([br1, br2, br3], axis=3)

        with tf.variable_scope('Conv_after_concat'):
            conv_after_concat = tf.layers.conv2d(concat, filters= 128, kernel_size= 3, strides= 1, padding= 'same',
                                                 activation= None, kernel_initializer= Initializer)
            conv_after_concat = tf.layers.batch_normalization(conv_after_concat, momentum= 0.9, training= BN_FLAG)
            conv_after_concat = tf.nn.relu(conv_after_concat)

        with tf.variable_scope('Deconv1'):
            deconv1_shape = resnet.get_shape().as_list()
            res = tf.layers.conv2d_transpose(conv_after_concat, filters= deconv1_shape[-1], kernel_size= 3, strides= 1,padding= 'same',
                                                activation= None, kernel_initializer= Initializer)
            res = tf.layers.batch_normalization(res, momentum=0.9, training=BN_FLAG)
            res = tf.nn.relu(res)
            fuse1 = tf.add(res, resnet)

        with tf.variable_scope('Deconv2'):
            deconv2_shape = end_points[ResModel+'/block3'].get_shape().as_list()
            res = tf.layers.conv2d_transpose(fuse1, filters= deconv2_shape[-1], kernel_size= 3, strides= 1,padding= 'same',
                                                activation= None, kernel_initializer= Initializer)
            res = tf.layers.batch_normalization(res, momentum=0.9, training=BN_FLAG)
            res = tf.nn.relu(res)
            fuse2 = tf.add(res, end_points[ResModel+'/block3'])

        with tf.variable_scope('Deconv3'):
            deconv3_shape = end_points[ResModel + '/block2'].get_shape().as_list()
            res = tf.layers.conv2d_transpose(fuse2, filters=deconv3_shape[-1], kernel_size=3, strides=2,
                                                padding='same',
                                                activation= None,
                                                kernel_initializer= Initializer)
            res = tf.layers.batch_normalization(res, momentum=0.9, training=BN_FLAG)
            res = tf.nn.relu(res)
            fuse3 = tf.add(res, end_points[ResModel + '/block2'])

        with tf.variable_scope('Deconv4'):
            deconv4_shape = end_points[ResModel + '/block1'].get_shape().as_list()
            res = tf.layers.conv2d_transpose(fuse3, filters=deconv4_shape[-1], kernel_size=3, strides=2,
                                                padding='same',
                                                activation= None,
                                                kernel_initializer= Initializer)
            res = tf.layers.batch_normalization(res, momentum=0.9, training=BN_FLAG)
            res = tf.nn.relu(res)
            fuse4 = tf.add(res, end_points[ResModel + '/block1'])

        with tf.variable_scope('Deconv5'):
            deconv5_shape = end_points[ResModel + '/conv1'].get_shape().as_list()
            res = tf.layers.conv2d_transpose(fuse4, filters=deconv5_shape[-1], kernel_size=3, strides=4,
                                                padding='same',
                                                activation= None,
                                                kernel_initializer= Initializer)
            res = tf.layers.batch_normalization(res, momentum=0.9, training=BN_FLAG)
            res = tf.nn.relu(res)
            fuse5 = tf.add(res, end_points[ResModel + '/conv1'])

        with tf.variable_scope('Last_layer'):
            logits = tf.layers.conv2d_transpose(fuse5, filters= 64, kernel_size=3, strides=2,
                                                padding='same',
                                                activation= None,
                                                kernel_initializer= Initializer)
            res = tf.layers.batch_normalization(logits, momentum=0.9, training=BN_FLAG)
            res = tf.nn.relu(res)
            logits = tf.layers.conv2d(res, filters= CLASSNUM, kernel_size= 3, strides= 1, padding= 'same', activation= None,
                                      kernel_initializer= Initializer)

        with tf.variable_scope('Argmax'):
            annotation_pred = tf.argmax(logits, axis=3, name='prediction')
        return logits, tf.expand_dims(annotation_pred, axis=3)










