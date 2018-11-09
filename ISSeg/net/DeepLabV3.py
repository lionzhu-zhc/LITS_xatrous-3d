#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2018/9/25 9:31
# @Author  : LionZhu
# this is the net based on DeepLab v3-2d

from net.resnet import resnet_v2, resnet_utils
import tensorflow as tf
import tensorflow.contrib.slim as slim

@slim.add_arg_scope
def atrous_spatial_pyramid_pooling(net, scope, depth = 256, reuse= None):
    '''
    aspp include a 1x1 conv and three 3x3 conv with rate[6,12,18] when output stride = 16
    :param net: the input, [BS, height, width, depth]
    :param scope: variable scope
    :param depth: filters num
    :param reuse:
    :return:
    '''
    with tf.variable_scope(scope, reuse= reuse):
        featuremap_size = tf.shape(net)

        #apply global average pooling
        # global pooling results: shape [BS, 1,1, channels], that is each feature map just has one pixel
        img_level_feature = tf.reduce_mean(net, [1,2], name= 'img_level_global_pool', keepdims= True)
        img_level_feature = slim.conv2d(img_level_feature, depth, [1,1], scope= 'img_level_conv1x1', activation_fn = None)
        img_level_feature = tf.image.resize_bilinear(img_level_feature, (featuremap_size[1], featuremap_size[2]))

        aspp_1 = slim.conv2d(net, depth, [1,1], scope= 'conv_1x1', activation_fn = None)
        aspp_2 = slim.conv2d(net, depth, [3,3], scope= 'conv_3x3_1', rate= 6, activation_fn = None)
        aspp_3 = slim.conv2d(net, depth, [3,3], scope= 'conv_3x3_2', rate= 12, activation_fn = None)
        aspp_4 = slim.conv2d(net, depth, [3,3], scope= 'conv_3x3_4', rate= 18, activation_fn = None)

        net = tf.concat([img_level_feature, aspp_1, aspp_2, aspp_3, aspp_4], axis= 3, name= 'concat')
        net = slim.conv2d(net, depth, [1,1], scope= 'conv1x1_output', activation_fn= None)
        return net


def deeplab_v3(inputs, args, is_training, reuse):
    with slim.arg_scope(resnet_utils.resnet_arg_scope(args.l2_regularizer, is_training, args.batch_norm_decay, args.batch_norm_epsilon)):
        #------------------here to choose the resnet layers------------------------------------------------------------------------------------------
        resnet = getattr(resnet_v2, args.resnet_model)
        _, end_points = resnet(inputs, args.number_of_class, is_training= is_training, global_pool= False, output_stride= args.output_stride, reuse= reuse)

        with tf.variable_scope('DeepLab_v3', reuse = reuse):
            net = end_points[args.resnet_model + '/block4']
            # get block4 feature output
            net = atrous_spatial_pyramid_pooling(net, scope='ASPP_layer', depth= 256, reuse= reuse)
            net = slim.conv2d(net, args.number_of_class, [1,1], activation_fn= None, normalizer_fn= None, scope = 'logits')
            size = tf.shape(inputs)[1:3]
            net = tf.image.resize_bilinear(net, size)
            return  net

