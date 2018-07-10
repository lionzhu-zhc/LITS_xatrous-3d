'''
2d unet
'''

import tensorflow as tf
from collections import OrderedDict
import numpy as np

Layer_num = 3
Feature_root = 16
Filter_size = 3
Pool_size = 2

def build_unet(tensor_in, BN_FLAG, BATCHSIZE, CLASSNUM, keep_porb = 0.85, in_channels= 1,):
    in_node = tensor_in

    weights = []
    biases = []
    convs = []
    pools = OrderedDict()
    deconv = OrderedDict()
    dw_h_convs = OrderedDict()
    up_h_convs = OrderedDict()

    in_size = 1000
    size = in_size

    #downwards layer-----------------------------------------------------------------------------------------------
    for layer in range(0, Layer_num):
        featuresNum = 2 ** layer * Feature_root
        stddev = np.sqrt(2 / (Filter_size**2 * featuresNum))
        if layer == 0:
            w1 = weight_variable([Filter_size, Filter_size, in_channels, featuresNum], stddev)
        else:
            w1 = weight_variable([Filter_size, Filter_size, featuresNum // 2, featuresNum], stddev)

        w2 = weight_variable([Filter_size, Filter_size, featuresNum, featuresNum], stddev)
        b1 = bias_variable([featuresNum])
        b2 = bias_variable([featuresNum])

        conv1 = conv2d(in_node, w1, keep_porb)
        conv1_relu = tf.nn.relu(tf.add(conv1, b1))
        conv2 = conv2d(conv1_relu, w2, keep_porb)
        dw_h_convs[layer] = tf.nn.relu(tf.add(conv2, b2))

        weights.append((w1, w2))                 # list pair [(w1, w2), (w1,w2)...]
        biases.append((b1,b2))
        convs.append((conv1, conv2))

        size = size - 4
        if layer < (Layer_num - 1):
            pools[layer] = max_pool(dw_h_convs[layer], Pool_size)
            in_node = pools[layer]
            size = size / 2
    in_node = dw_h_convs[Layer_num-1]

    #upwards layers----------------------------------------------------------------------------------------------------
    for layer in range(Layer_num-2, -1, -1):
        featuresNum = 2**(layer+1) * Feature_root
        stddev = np.sqrt(2 / (Filter_size ** 2 * featuresNum))

        wd = weight_variable_deconv([Pool_size, Pool_size, featuresNum//2, featuresNum], stddev)
        bd = bias_variable([featuresNum // 2])
        h_deconv = tf.nn.relu((deconv2d(in_node, wd, cstride= Pool_size) + bd))
        h_deconv_concat = corp_concat(dw_h_convs[layer], h_deconv)
        deconv[layer] = h_deconv_concat

        w1 = weight_variable([Filter_size, Filter_size, featuresNum, featuresNum//2], stddev)
        w2 = weight_variable([Filter_size, Filter_size, featuresNum//2, featuresNum//2], stddev)
        b1 = bias_variable([featuresNum//2])
        b2 = bias_variable([featuresNum//2])

        conv1 = conv2d(h_deconv_concat, w1, keep_porb)
        h_conv = tf.nn.relu(tf.add(conv1, b1))
        conv2 = conv2d(h_conv, w2, keep_porb)
        in_node = tf.nn.relu(tf.add(conv2, b2))
        up_h_convs[layer] = in_node

        weights.append((w1, w2))
        biases.append((b1, b2))
        convs.append((conv1, conv2))

        size = 2 * size
        size = size - 4

    #out map -----------------------------------------------------------------------------------------------------
    wei = weight_variable([1, 1, Feature_root, CLASSNUM], stddev)
    bias = bias_variable([CLASSNUM])
    conv = conv2d(in_node, wei, keepprob= 1.0)
    out_map = tf.nn.relu(tf.add(conv, bias))
    up_h_convs['out'] = out_map
    predicter = pixwise_softmax_2(out_map)
    prediction = tf.argmax(predicter, axis= 3)

    variables = []
    for w1, w2 in weights:
        variables.append(w1)
        variables.append(w2)

    for b1,b2 in biases:
        variables.append(b1)
        variables.append(b2)

    return out_map, prediction, variables, int(in_size - size)



def weight_variable(shape, stddev = 0.1):
    init = tf.truncated_normal(shape, stddev = stddev)
    return tf.Variable(init)

def weight_variable_deconv(shape, stddev= 0.1):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev))

def bias_variable(shape):
    init = tf.constant(0.1, shape= shape)
    return tf.Variable(init)

def conv2d(x, w, keepprob, cstride= 1):
    conv = tf.nn.conv2d(x,w, strides= [1,cstride,cstride,1], padding= 'SAME')
    return tf.nn.dropout(conv, keepprob)

def max_pool(x, poolsize = 2):
    pool = tf.nn.max_pool(x, ksize=[1, poolsize, poolsize, 1], strides=[1, poolsize, poolsize, 1], padding='SAME')
    return pool

def deconv2d(x, w, cstride= 1):
    x_shape = tf.shape(x)
    outshape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]//2])
    deconv = tf.nn.conv2d_transpose(x, w, outshape, strides=[1,cstride,cstride,1], padding= 'SAME')
    return deconv

def corp_concat(x1, x2):
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)

    #offset  for the top left corner of the corp
    offsets = [0, (x1_shape[1]-x2_shape[1])//2, (x1_shape[2]-x2_shape[2])//2, 0]
    size = [-1, x2_shape[1], x2_shape[2], -1]
    x1_corp = tf.slice(x1, offsets, size)
    return tf.concat([x1_corp, x2], axis= 3)

def pixwise_softmax_2(out_map):
    exp_map = tf.exp(out_map)
    exp_sum = tf.reduce_sum(exp_map, 3, keepdims= True)
    tensor_sum_exp = tf.tile(exp_sum, tf.stack([1, 1, 1, tf.shape(out_map)[3]]))
    return tf.div(exp_map, tensor_sum_exp)