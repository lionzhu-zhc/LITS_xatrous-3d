'''
20180404 LionZhu
--------------psp_atrous_net-------------------
the file for net structure
the ori net changed from pspnet
'''

import tensorflow as tf

print("beign")

def build_LITS_Xatrous_3d(tensor_in, BN_FLAG, BATCHSIZE, IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH, CLASSNUM):
    print(tensor_in.get_shape())

    with tf.variable_scope('st1_un1'):
        conv_res = conv_layer(kernel_size=3, in_put= tensor_in, in_channel= 1,
                              out_channel= 64, c_stride= 2, c_rate=3, name='Conv')
        bn_res =  tf.layers.batch_normalization(conv_res, momentum= 0.9, training= BN_FLAG, name= 'BN')
        s1u1_res = tf.nn.relu(bn_res)

    with tf.variable_scope('st1_un2_a1'):
        conv_res = conv_layer(kernel_size= 1, in_put= s1u1_res, in_channel= 64, out_channel= 32, name= 'Conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum= 0.9, training= BN_FLAG, name = 'BN')
        s1u2a1_res = tf.nn.relu(bn_res)

    with tf.variable_scope('st1_un2_a2'):
        conv_res = conv_layer(kernel_size= 3, in_put= s1u2a1_res, in_channel= 32, out_channel= 32, c_rate= 3, name= 'Conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum= 0.9, training= BN_FLAG, name = 'BN')
        s1u1a2_res = tf.nn.relu(bn_res)

    with tf.variable_scope('st1_un2_a3'):
        conv_res = conv_layer(kernel_size= 1, in_put= s1u1a2_res, in_channel= 32, out_channel= 64, name= 'Conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BN')
        s1u1a3_res = tf.nn.relu(bn_res)

    with tf.variable_scope('s1u2_sum'):
        s1u2_sum = tf.add(s1u1_res, s1u1a3_res)  # channel: 64

    with tf.variable_scope('st1_un3_a1'):
        conv_res = conv_layer(kernel_size= 1, in_put= s1u2_sum, in_channel=64, out_channel= 32, name= 'Conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BN')
        s1u3a1_res = tf.nn.relu(bn_res)

    with tf.variable_scope('st1_un3_a2'):
        conv_res = conv_layer(kernel_size= 3, in_put=s1u3a1_res, in_channel=32, out_channel= 32, c_rate= 3, name= 'Conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BN')
        s1u3a2_res = tf.nn.relu(bn_res)

    with tf.variable_scope('st1_un3_a3'):
        conv_res = conv_layer(kernel_size= 1, in_put= s1u3a2_res, in_channel= 32, out_channel= 64, name= 'Conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BN')
        s1u3a3_res = tf.nn.relu(bn_res)

    with tf.variable_scope('s1u3_sum'):
        s1u3_sum = tf.add(s1u2_sum, s1u3a3_res)  # channel: 64

    with tf.variable_scope('st1_un4_a1'):
        conv_res = conv_layer(kernel_size= 1, in_put= s1u3_sum, in_channel=64, out_channel= 32, name= 'Conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BN')
        s1u4a1_res = tf.nn.relu(bn_res)

    with tf.variable_scope('st1_un4_a2'):
        conv_res = conv_layer(kernel_size= 3, in_put= s1u4a1_res, in_channel=32, out_channel= 32, c_rate=3, name= 'Conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BN')
        s1u4a2_res = tf.nn.relu(bn_res)

    with tf.variable_scope('st1_un4_a3'):
        conv_res = conv_layer(kernel_size= 1, in_put= s1u4a2_res, in_channel= 32, out_channel= 64, name= 'Conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BN')
        s1u4a3_res = tf.nn.relu(bn_res)

    with tf.variable_scope('s1u4_sum'):
        s1u4_sum = tf.add(s1u3_sum, s1u4a3_res)  # channel: 64

    with tf.variable_scope('st2_un1_a1'):
        conv_res = conv_layer(kernel_size= 1, in_put= s1u4_sum, in_channel= 64, out_channel= 128, name= 'Conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BN')
        s2u1a1_res = tf.nn.relu(bn_res)

    with tf.variable_scope('st2_un1_a2'):
        conv_res = conv_layer(kernel_size= 3, in_put=s2u1a1_res, in_channel= 128, out_channel=128, c_stride= 2,c_rate= 3, name= 'Conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BN')
        s2u1a2_res = tf.nn.relu(bn_res)

    with tf.variable_scope('st2_un1_a3'):
        conv_res = conv_layer(kernel_size= 1, in_put= s2u1a2_res, in_channel= 128, out_channel= 256, name= 'Conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BN')
        s2u1a3_res = tf.nn.relu(bn_res)

    with tf.variable_scope('st2_un1_b1'):
        conv_res = conv_layer(kernel_size= 1, in_put= s1u4_sum, in_channel= 64, out_channel= 256, c_stride= 2, name= 'Conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BN')
        s2u1b1_res = tf.nn.relu(bn_res)

    with tf.variable_scope('s2u1_sum'):
        s2u1_sum = tf.add(s2u1b1_res, s2u1a3_res)  # channel: 256

    with tf.variable_scope('st2_un2_a1'):
        conv_res = conv_layer(kernel_size= 1, in_put=s2u1_sum, in_channel= 256, out_channel= 128, name= 'Conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BN')
        s2u2a1_res = tf.nn.relu(bn_res)

    with tf.variable_scope('st2_un2_a2'):
        conv_res = conv_layer(kernel_size=3, in_put=s2u2a1_res, in_channel=128, out_channel=128, c_rate= 3, name='Conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BatchNorm')
        s2u2a2_res = tf.nn.relu(bn_res)
    with tf.variable_scope('st2_un2_a3'):
        conv_res = conv_layer(kernel_size=1, in_put=s2u2a2_res, in_channel=128,
                                   out_channel=256, name='Conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BatchNorm')
        s2u2a3_res = tf.nn.relu(bn_res)
    with tf.variable_scope('s2u2_sum'):
        s2u2_sum = tf.add(s2u1_sum, s2u2a3_res)  # channel: 256

    with tf.variable_scope('st2_un3_a1'):
        conv_res = conv_layer(kernel_size=1, in_put=s2u2_sum, in_channel=256,
                                   out_channel=128, name='Conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BatchNorm')
        s2u3a1_res = tf.nn.relu(bn_res)
    with tf.variable_scope('st2_un3_a2'):
        conv_res = conv_layer(kernel_size=3, in_put=s2u3a1_res, in_channel=128,
                                   out_channel=128, c_rate= 3, name='Conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BatchNorm')
        s2u3a2_res = tf.nn.relu(bn_res)
    with tf.variable_scope('st2_un3_a3'):
        conv_res = conv_layer(kernel_size=1, in_put=s2u3a2_res, in_channel=128,
                                   out_channel=256, name='Conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BatchNorm')
        s2u3a3_res = tf.nn.relu(bn_res)
    with tf.variable_scope('s2u3_sum'):
        s2u3_sum = tf.add(s2u2_sum, s2u3a3_res)  # channel: 256

    with tf.variable_scope('st2_un4_a1'):
        conv_res = conv_layer(kernel_size=1, in_put=s2u3_sum, in_channel=256,
                                   out_channel=128, name='Conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BatchNorm')
        s2u4a1_res = tf.nn.relu(bn_res)
    with tf.variable_scope('st2_un4_a2'):
        conv_res = conv_layer(kernel_size=3, in_put=s2u4a1_res, in_channel=128,
                                   out_channel=128, c_rate= 3, name='Conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BatchNorm')
        s2u4a2_res = tf.nn.relu(bn_res)
    with tf.variable_scope('st2_un4_a3'):
        conv_res = conv_layer(kernel_size=1, in_put=s2u4a2_res, in_channel=128,
                                   out_channel=256, name='Conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BatchNorm')
        s2u4a3_res = tf.nn.relu(bn_res)
    with tf.variable_scope('s2u4_sum'):
        s2u4_sum = tf.add(s2u3_sum, s2u4a3_res)  # channel: 256

    with tf.variable_scope('st2_un5_a1'):
        conv_res = conv_layer(kernel_size=1, in_put=s2u4_sum, in_channel=256,
                                   out_channel=128,  name='Conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BatchNorm')
        s2u5a1_res = tf.nn.relu(bn_res)
    with tf.variable_scope('st2_un5_a2'):
        conv_res = conv_layer(kernel_size=3, in_put=s2u5a1_res, in_channel=128,
                                   out_channel=128, c_rate = 3, name='Conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BatchNorm')
        s2u5a2_res = tf.nn.relu(bn_res)
    with tf.variable_scope('st2_un5_a3'):
        conv_res = conv_layer(kernel_size=1, in_put=s2u5a2_res, in_channel=128,
                                   out_channel=256, name='Conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BatchNorm')
        s2u5a3_res = tf.nn.relu(bn_res)
    with tf.variable_scope('s2u5_sum'):
        s2u5_sum = tf.add(s2u4_sum, s2u5a3_res)  # channel: 256
        shape_s2u5 = tf.shape(s2u5_sum)

    with tf.variable_scope('s2u6_br1_pool'):
        s2u6br1_pool = max_pool3d(s2u5_sum, ksize=2, stride=2)
    with tf.variable_scope('s2u6_br1_conv'):
        conv_res = conv_layer(kernel_size=3, in_put=s2u6br1_pool, in_channel=256,
                                   out_channel=128, c_rate= 3, name='Conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BatchNorm')
        s2u6br1_conv = tf.nn.relu(bn_res)
    with tf.variable_scope('s2u6_br1_deconv'):
        deconv_res = conv3d_transpose_layer(kernel_size=2, in_put=s2u6br1_conv, out_shape=shape_s2u5,
                                                 in_channel=128, out_channel=256, c_stride=2, name='Deconv')
        bn_res = tf.layers.batch_normalization(deconv_res, momentum=0.9, training=BN_FLAG, name='BatchNorm')
        s2u6br1_deconv = tf.nn.relu(bn_res)

    with tf.variable_scope('s2u6_br2_pool'):
        s2u6br2_pool = max_pool3d(s2u5_sum, ksize=4, stride=4)
    with tf.variable_scope('s2u6_br2_conv'):
        conv_res = conv_layer(kernel_size=3, in_put=s2u6br2_pool, in_channel=256,
                                   out_channel=128, c_rate= 3, name='Conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BatchNorm')
        s2u6br2_conv = tf.nn.relu(bn_res)
    with tf.variable_scope('s2u6_br2_deconv'):
        deconv_res = conv3d_transpose_layer(kernel_size=4, in_put=s2u6br2_conv, out_shape=shape_s2u5,
                                                 in_channel=128, out_channel=256, c_stride=4, name='Deconv')
        bn_res = tf.layers.batch_normalization(deconv_res, momentum=0.9, training=BN_FLAG, name='BatchNorm')
        s2u6br2_deconv = tf.nn.relu(bn_res)

    with tf.variable_scope('s2u6_br3_pool'):
        s2u6br3_pool = max_pool3d(s2u5_sum, ksize=8, stride=8)
    with tf.variable_scope('s2u6_br3_conv'):
        conv_res = conv_layer(kernel_size=3, in_put=s2u6br3_pool, in_channel=256,
                                   out_channel=128, c_rate= 3, name='Conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BatchNorm')
        s2u6br3_conv = tf.nn.relu(bn_res)
    with tf.variable_scope('s2u6_br3_deconv'):
        deconv_res = conv3d_transpose_layer(kernel_size=8, in_put=s2u6br3_conv, out_shape=shape_s2u5,
                                                 in_channel=128, out_channel=256, c_stride=8, name='Deconv')
        bn_res = tf.layers.batch_normalization(deconv_res, momentum=0.9, training=BN_FLAG, name='BatchNorm')
        s2u6br3_deconv = tf.nn.relu(bn_res)

    with tf.variable_scope('s2u6_concat'):
        s2u6_concat = tf.concat([s2u5_sum, s2u6br1_deconv, s2u6br2_deconv, s2u6br3_deconv], axis=4, name='concat')
        # channel: 1024

    with tf.variable_scope('st3_un1'):
        conv_res = conv_layer(kernel_size=3, in_put=s2u6_concat, in_channel=1024,
                                   out_channel=128, c_rate= 3, name='conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BatchNorm')
        s3u1_res = tf.nn.relu(bn_res)  # channel 128

    with tf.variable_scope('st3_un1_a1'):
        conv_res = conv_layer(kernel_size=1, in_put=s3u1_res, in_channel=128,
                                   out_channel=64, name='conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BatchNorm')
        s3u1a1_res = tf.nn.relu(bn_res)
    with tf.variable_scope('st3_un1_a2'):
        conv_res = conv_layer(kernel_size=3, in_put=s3u1a1_res, in_channel=64,
                                   out_channel=64, c_rate= 3, name='conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BatchNorm')
        s3u1a2_res = tf.nn.relu(bn_res)
    with tf.variable_scope('st3_un1_a3'):
        conv_res = conv_layer(kernel_size=1, in_put=s3u1a2_res, in_channel=64,
                                   out_channel=128, name='conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BatchNorm')
        s3u1a3_res = tf.nn.relu(bn_res)
    with tf.variable_scope('st3_un1_b1'):
        conv_res = conv_layer(kernel_size=1, in_put=s3u1_res, in_channel=128,
                                   out_channel=128, name='conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BatchNorm')
        s3u1b1_res = tf.nn.relu(bn_res)
    with tf.variable_scope('s3u1_sum'):
        s3u1_sum = tf.add(s3u1a3_res, s3u1b1_res)  # channel: 128

    with tf.variable_scope('st3_un2_a1'):
        conv_res = conv_layer(kernel_size=1, in_put=s3u1_sum, in_channel=128,
                                   out_channel=64, name='conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BatchNorm')
        s3u2a1_res = tf.nn.relu(bn_res)
    with tf.variable_scope('st3_un2_a2'):
        conv_res = conv_layer(kernel_size=3, in_put=s3u2a1_res, in_channel=64,
                                   out_channel=64, c_rate= 3, name='conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BatchNorm')
        s3u2a2_res = tf.nn.relu(bn_res)
    with tf.variable_scope('st3_un2_a3'):
        conv_res = conv_layer(kernel_size=1, in_put=s3u2a2_res, in_channel=64,
                                   out_channel=128, name='conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BatchNorm')
        s3u2a3_res = tf.nn.relu(bn_res)
    with tf.variable_scope('s3u2_sum'):
        s3u2_sum = tf.add(s3u1_sum, s3u2a3_res)  # channel: 128

    with tf.variable_scope('st3_un3_a1'):
        conv_res = conv_layer(kernel_size=1, in_put=s3u2_sum, in_channel=128,
                                   out_channel=64, name='conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BatchNorm')
        s3u3a1_res = tf.nn.relu(bn_res)
    with tf.variable_scope('st3_un3_a2'):
        conv_res = conv_layer(kernel_size=3, in_put=s3u3a1_res, in_channel=64,
                                   out_channel=64, c_rate= 3, name='conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BatchNorm')
        s3u3a2_res = tf.nn.relu(bn_res)
    with tf.variable_scope('st3_un3_a3'):
        conv_res = conv_layer(kernel_size=1, in_put=s3u3a2_res, in_channel=64,
                                   out_channel=128, name='conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BatchNorm')
        s3u3a3_res = tf.nn.relu(bn_res)
    with tf.variable_scope('s3u3_sum'):
        s3u3_sum = tf.add(s3u2_sum, s3u3a3_res)  # channel: 128

    with tf.variable_scope('st3_un4_a1'):
        conv_res = conv_layer(kernel_size=1, in_put=s3u3_sum, in_channel=128,
                                   out_channel=64, name='conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BatchNorm')
        s3u4a1_res = tf.nn.relu(bn_res)
    with tf.variable_scope('st3_un4_a2'):
        conv_res = conv_layer(kernel_size=3, in_put=s3u4a1_res, in_channel=64,
                                   out_channel=64, c_rate= 3, name='conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BatchNorm')
        s3u4a2_res = tf.nn.relu(bn_res)
    with tf.variable_scope('st3_un4_a3'):
        conv_res = conv_layer(kernel_size=1, in_put=s3u4a2_res, in_channel=64,
                                   out_channel=128, name='conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BatchNorm')
        s3u4a3_res = tf.nn.relu(bn_res)
    with tf.variable_scope('s3u4_sum'):
        s3u4_sum = tf.add(s3u3_sum, s3u4a3_res)  # channel: 128

    with tf.variable_scope('s3_u5_deconv'):
        # outshape = tf.shape(tensor_in)
        deconv = conv3d_transpose_layer(kernel_size=4, in_put=s3u4_sum,
                                             out_shape=[BATCHSIZE, IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH, 64],
                                             in_channel=128, out_channel=64, c_stride=4, name='Deconv')
        bn_res = tf.layers.batch_normalization(deconv, momentum=0.9, training=BN_FLAG, name='BatchNorm')
        s3u5_deconv = tf.nn.relu(bn_res)

    with tf.variable_scope('s3_u6_conv'):
        s3u6_conv = conv_layer(kernel_size=1, in_put=s3u5_deconv, in_channel=64,
                                    out_channel=CLASSNUM, name='Conv')
    # shape [_, depth, height, width, channel]

    with tf.variable_scope('SoftMax'):
        pred_annot = tf.argmax(s3u6_conv, axis=4, name='prediction')

    return tf.expand_dims(pred_annot, dim=4), s3u6_conv

def conv_layer(kernel_size, in_put, in_channel, out_channel, c_stride = 1, c_rate = 1, name = None):
    with tf.variable_scope(name):
        conv_weights, conv_bias = get_var(kernel_size, in_channel, out_channel, name)
        conv = tf.nn.conv3d(in_put, conv_weights, strides= [1, c_stride, c_stride, c_stride, 1], padding= 'SAME', dilations= [1, 1, c_rate, c_rate, 1])
        conv_addbias = tf.nn.bias_add(conv, conv_bias)
        return conv_addbias

def max_pool3d(in_tensor, ksize, stride):
    maxpool_res = tf.nn.max_pool3d(in_tensor, ksize=[1, ksize, ksize, ksize, 1],
                                   strides=[1, stride, stride, stride, 1], padding='SAME', name='MaxPool')
    return maxpool_res

def conv3d_transpose_layer( kernel_size, in_put, out_shape, in_channel, out_channel, c_stride, name):
    with tf.variable_scope(name):
        deconv_weights, deconv_bias = get_var_transpose(kernel_size, in_channel, out_channel, name)
        deconv = tf.nn.conv3d_transpose(in_put, deconv_weights, out_shape, [1, c_stride, c_stride, c_stride, 1], padding= 'SAME')
        deconv_addbias = tf.nn.bias_add(deconv, deconv_bias)
        return  deconv_addbias

# xavier initializer---------------------------------------------------------------------------------------------------
# def get_var(kernel_size, in_channel, out_channel, name = None):
#     weights = tf.get_variable('w', shape=[kernel_size, kernel_size, kernel_size, in_channel, out_channel],
#                               initializer= tf.contrib.layers.xavier_initializer())
#     bias = tf.get_variable('b', [out_channel], initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))
#
#     return weights, bias
#
# def get_var_transpose(kernel_size, in_channel, out_channel, name = None):
#     weights = tf.get_variable('w', shape=[kernel_size, kernel_size, kernel_size, out_channel, in_channel],
#                               initializer=tf.contrib.layers.xavier_initializer())
#
#     bias = tf.get_variable('b', [out_channel],
#                            initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))
#     return weights, bias


# normal init-----------------------------------------------------------------------------------------------------------

def get_var(kernel_size, in_channel, out_channel, name = None):
    init_value = tf.truncated_normal([kernel_size, kernel_size, kernel_size, in_channel, out_channel], 0.0, 0.01)
    weights = tf.Variable(init_value, name= 'w')
    init_value = tf.truncated_normal([out_channel], 0.0, 0.01)
    bias = tf.Variable(init_value, name= 'b')
    return  weights, bias

def get_var_transpose(kernel_size, in_channel, out_channel, name = None):
    init_value = tf.truncated_normal([kernel_size, kernel_size, kernel_size, out_channel, in_channel], 0.0, 0.01)
    weights = tf.Variable(init_value, name= 'w')
    init_value = tf.truncated_normal([out_channel], 0.0, 0.01)
    bias = tf.Variable(init_value, name= 'b')
    return  weights, bias

