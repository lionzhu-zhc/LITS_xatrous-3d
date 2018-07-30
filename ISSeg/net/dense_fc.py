'''
this is network of Full Conv Densenet
Lionzhu-list 2018-07-27
'''

import tensorflow as tf

NUMPOOL = 4
FIRST_LAYER_FILTERS = 48

def build_dense_fc(tensor_in, BN_FLAG, BATCHSIZE, CLASSNUM, IMGCHANNEL):
    print(tensor_in.get_shape())

    first_layer = tf.nn.conv2d(tensor_in, filter= [3,3,IMGCHANNEL, FIRST_LAYER_FILTERS], strides= [1,1,1,1], padding= 'SAME', name= 'First_Conv')

    n_filters = FIRST_LAYER_FILTERS
    


def dense_block(in_put, )


def BN_Relu_Conv(in_put, in_channel, out_channel, keep_prob, BN_FLAG, kernel_size = 3, name = 'BN_Relu_Conv'):
    with tf.name_scope(name):
        res = tf.layers.batch_normalization(in_put, momentum= 0.9, training= BN_FLAG, name= 'BatchNorm')
        res = tf.nn.relu(res)
        res = conv2d_layer(kernel_size, res, in_channel, out_channel, keep_prob= keep_prob)
        return res

def down_layer(in_put, in_channel, out_channel, keep_prob, BN_FLAG, name = 'down_layer'):
    with tf.name_scope(name):
        res = BN_Relu_Conv(in_put, in_channel, out_channel, keep_prob, BN_FLAG, kernel_size=1)
        res = pool2d_layer(res, ksize = 2, stride= 2)

        return res

def up_layer(in_put, out_shape, in_channel, out_channel, kernel_size, stride, c_rate= 1, name= 'up_layer'):
    with tf.name_scope(name):
        deconv_weights, deconv_bias = get_var_transpose(kernel_size, in_channel, out_channel)
        deconv = tf.nn.conv2d_transpose(in_put, deconv_weights, out_shape, strides= [1, stride, stride, 1],
                                        padding = 'SAME', dialations= [1, c_rate, c_rate, 1])
        deconv = tf.nn.bias_add(deconv, deconv_bias)

        return deconv


def conv2d_layer(kernel_size, in_put, in_channel, out_channel, c_stride = 1, c_rate = 1, keep_prob = 0.8, name = None):
    with tf.name_scope(name= name):
        conv_weights, conv_bias = get_var(kernel_size, in_channel, out_channel)
        conv = tf.nn.conv2d(in_put, conv_weights, stride = [1, c_stride, c_stride, 1], 
                            padding= 'SAME', dialations= [1, c_rate, c_rate, 1])
        conv = tf.nn.bias_add(conv, conv_bias)
        conv = tf.nn.dropout(conv, keep_prob)

        return conv

def get_var(kernel_size, in_channel, out_channel):
    weights = tf.get_variable('w', shape=[kernel_size, kernel_size, in_channel, out_channel],
                              initializer= tf.contrib.layers.xavier_initializer())
    bias = tf.get_variable('b', [out_channel], initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))

    return weights, bias

def pool2d_layer(in_put, ksize, stride):
    res = tf.nn.max_pool(in_put, ksize= [1, ksize, ksize, 1], 
                        strides= [1, stride, stride, 1], padding= 'SAME', name = 'Max_Pool')

    return res

def get_var_transpose(kernel_size, in_channel, out_channel):
    weights = tf.get_variable('w', shape=[kernel_size, kernel_size, out_channel, in_channel],
                              initializer=tf.contrib.layers.xavier_initializer())

    bias = tf.get_variable('b', [out_channel],
                           initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))
    return weights, bias















