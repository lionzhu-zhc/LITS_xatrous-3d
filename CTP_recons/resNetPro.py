# -*- coding: utf-8 -*-
"""

2d resnet pro

180917

@author: Cifer Dog
"""

import tensorflow as tf
"""
def leaky_relu(x,alpha=0.2,max_value=None):

    #alpha: slope of negative section

    negative_part = tf.nn.relu(tf.negative(x))
    x = tf.nn.relu(x)
    if max_value is not None:
        x = tf.clip_by_value(x,tf.cast(0.,dtype=tf.float32),
                             tf.cast(max_value,dtype = tf.float32))
    x = tf.subtract(x,tf.multiply(tf.constant(alpha,dtype=tf.float32),negative_part))
    return x
"""
def conv2d(name,input, kernel_size,stride=[1,1,1,1],bn=False,relu=False,bias=False,is_train=True):

    weights = tf.get_variable(name+'_weights', shape=kernel_size,regularizer=tf.contrib.layers.l2_regularizer(1e-5), 
                              initializer=tf.truncated_normal_initializer(stddev=0.001),dtype=tf.float32)
    conv = tf.nn.conv2d(input, filter=weights, strides=stride, padding='SAME',name=name+'conv')
    C_out = conv

    if bn:
        C_out = tf.contrib.layers.batch_norm(C_out,decay=0.999,epsilon=1e-3,
                                             # param_initializers=tf.contrib.layers.xavier_initializer,
                                             is_training=is_train,scope=name+'_BN')
        
    if bias:
        bias_value = tf.get_variable(name+'_bias',kernel_size[3],
                                     initializer=tf.constant_initializer(0.0))
        C_out = tf.nn.bias_add(C_out,bias_value)  
        
    if relu:
        C_out = tf.nn.leaky_relu(C_out)

    return C_out

#inputs batchsizei in_depth in_height in_width in_channels
def resnet(input,train_sign=True):
    
    C1 = conv2d('Pro_C1',input,[5,5,30,16],bn=False,relu=True,bias=True,is_train=train_sign)
    C2 = conv2d('Pro_C2',C1,[3,3,16,32],bn=True,relu=True,bias=False,is_train=train_sign)
    C3 = conv2d('Pro_C3',C2,[3,3,32,64],bn=True,relu=True,bias=False,is_train=train_sign)
    C4 = conv2d('Pro_C4',C3,[3,3,64,128],bn=True,relu=True,bias=False,is_train=train_sign)
    
    #Res1_input = conv3d('Res1_input',C4,[3,3,1,128,128],bn=True,relu=True,bias=False,is_train=train_sign)
    Res1_input = C4
    C5 = conv2d('Pro_C5',C4,[3,3,128,128],bn=True,relu=True,bias=False,is_train=train_sign)
    C6 = conv2d('Pro_C6',C5,[3,3,128,128],bn=True,relu=True,bias=False,is_train=train_sign)
    Res1_weight = tf.get_variable('Res1_weight',1,initializer=tf.constant_initializer(1.0))
    Res1_output = tf.add(tf.multiply(Res1_input,Res1_weight),C6)
    Res1_output = tf.contrib.layers.batch_norm(Res1_output,is_training=train_sign,scope='Res1_BN')
    Res1_output = tf.nn.leaky_relu(Res1_output)
    
    #Res2_input = conv3d('Res2_input',Res1_output,[3,3,1,128,128],bn=True,relu=True,bias=False,is_train=train_sign)
    Res2_input = Res1_output
    C7 = conv2d('Pro_C7',Res1_output,[3,3,128,128],bn=True,relu=True,bias=False,is_train=train_sign)
    C8 = conv2d('Pro_C8',C7,[3,3,128,128],bn=True,relu=True,bias=False,is_train=train_sign)
    Res2_weight = tf.get_variable('Res2_weight',1,initializer=tf.constant_initializer(1.0))
    Res2_output = tf.add(tf.multiply(Res2_input,Res2_weight),C8)
    Res2_output = tf.contrib.layers.batch_norm(Res2_output,is_training=train_sign,scope='Res2_BN')
    Res2_output = tf.nn.leaky_relu(Res2_output)
    
    #Res3_input = conv2d('Res3_input',Res2_output,[3,3,1,128,128],bn=True,relu=True,bias=False,is_train=train_sign)
    Res3_input = Res2_output
    C9 = conv2d('Pro_C9',Res2_output,[3,3,128,128],bn=True,relu=True,bias=False,is_train=train_sign)
    C10 = conv2d('Pro_C10',C9,[3,3,128,128],bn=True,relu=True,bias=False,is_train=train_sign)
    Res3_weight = tf.get_variable('Res3_weight',1,initializer=tf.constant_initializer(1.0))
    Res3_output = tf.add(tf.multiply(Res3_input,Res3_weight),C10)
    Res3_output = tf.contrib.layers.batch_norm(Res3_output,is_training=train_sign,scope='Res3_BN')
    Res3_output = tf.nn.leaky_relu(Res3_output)
    
    C_last = conv2d('Pro_C_last',Res3_output,[3,3,128,1],bn=False,relu=False,bias=True,is_train=train_sign)
    
    return C_last

    