# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 21:06:09 2017

@author: hank
"""
import tensorflow as tf

def batch_norm(input_,depth,bn_train=True):
    return tf.contrib.layers.batch_norm(input_,
                                        decay=0.999, ############0.997
                                        epsilon=1e-3,
                                        is_training=bn_train,
                                        scope='BN')
def conv3d(input, kernel_shape,stride=[1,1,1,1,1]):
    # Create variable named "weights".
    weights = tf.get_variable("weights", kernel_shape,regularizer=tf.contrib.layers.l2_regularizer(1e-5), 
        initializer=tf.truncated_normal_initializer(stddev=0.01))###0.001
    conv = tf.nn.conv3d(input, weights,
        strides=stride, padding='VALID')
    return conv

#inputs batchsizei in_depth in_height in_width in_channels
def resnet3d(inputs,train_sign=True):
    with tf.variable_scope("C1"):
        c1_bias = tf.get_variable("c1_bias",64,initializer=tf.constant_initializer(0.0))
        c1_conv = conv3d(inputs,[5,5,5,1,64])
        c1_relu = tf.nn.relu(tf.nn.bias_add(c1_conv,c1_bias))
        #c1_conv = conv3d(inputs,[5,5,5,1,64])
        #c1_bn = batch_norm(c1_conv,bn_train=train_sign)
        #c1_relu = tf.nn.relu(c1_bn)
        
        
    with tf.variable_scope("C2"):
        c2_conv = conv3d(c1_relu,[3,3,1,64,64])
        c2_bn = batch_norm(c2_conv,64,bn_train=train_sign)
        c2_relu = tf.nn.relu(c2_bn)
        
    with tf.variable_scope("res1_C1"):
        res1_C1_conv = conv3d(c2_relu,[3,3,3,64,64])
        res1_C1_bn = batch_norm(res1_C1_conv,64,bn_train=train_sign)
        res1_C1_relu = tf.nn.relu(res1_C1_bn)
        
    with tf.variable_scope("res1_C2"):
        res1_C2 = conv3d(res1_C1_relu,[3,3,1,64,64])
        
    c3_sum = c2_relu[:,2:-2,2:-2,1:-1,:] + res1_C2
    
    with tf.variable_scope("C4"):
        c4_bn_1 = batch_norm(c3_sum,64,bn_train=train_sign)
        c4_relu_1 = tf.nn.relu(c4_bn_1)
        c4_conv = conv3d(c4_relu_1,[3,3,3,64,64])

        
    with tf.variable_scope("C5"):
        c5_relu = tf.nn.relu(c4_conv)
        c5_bn = batch_norm(c5_relu,64,bn_train=train_sign)
        c5_conv = conv3d(c5_bn,[3,3,1,64,64])
        #c5_relu = tf.nn.relu(c5_bn)
        #c5_conv = conv3d(c5_relu,[3,3,1,64,64])
        
    c5_sum = c3_sum[:,2:-2,2:-2,1:-1,:] + c5_conv
    
    with tf.variable_scope("C6"):
        c6_bn = batch_norm(c5_sum,64,bn_train=train_sign)
        c6_relu = tf.nn.relu(c6_bn)
        c6_conv = conv3d(c6_relu,[3,3,3,64,64])
        
    with tf.variable_scope("C7"):
        c7_bn = batch_norm(c6_conv,64,bn_train=train_sign)
        c7_relu = tf.nn.relu(c7_bn)
        c7_conv = conv3d(c7_relu,[3,3,1,64,64])
        
    c7_sum = c5_sum[:,2:-2,2:-2,1:-1,:] + c7_conv
    
    with tf.variable_scope("C8"):
        c8_bn = batch_norm(c7_sum,64,bn_train=train_sign)
        c8_relu = tf.nn.relu(c8_bn)
        c8_conv = conv3d(c8_relu,[3,3,3,64,64])
        
    with tf.variable_scope("C9"):
        c9_bn = batch_norm(c8_conv,64,bn_train=train_sign)
        c9_relu = tf.nn.relu(c9_bn)
        c9_conv = conv3d(c9_relu,[3,3,1,64,64])
        
    c9_sum = c7_sum[:,2:-2,2:-2,1:-1,:] + c9_conv
    c9_sum_bn = batch_norm(c9_sum,64,bn_train=train_sign)
    c9_sum_relu = tf.nn.relu(c9_sum_bn)
    
#    with tf.variable_scope("C10"):
#        c10_conv = conv3d(c9_sum_relu,[3,3,3,64,64])
#        c10_bn = batch_norm(c10_conv,bn_train=train_sign)
#        c10_relu = tf.nn.relu(c10_bn)
    
    c_last_bias = tf.get_variable("bias",1,initializer=tf.constant_initializer(0.0))
    #if train_sign == True:
    #    c9_sum_relu = tf.nn.dropout(c9_sum_relu,0.5)
    c_last_conv = conv3d(c9_sum_relu,[3,3,1,64,1])
    c_last = tf.nn.bias_add(c_last_conv,c_last_bias)
    
    return c_last

    