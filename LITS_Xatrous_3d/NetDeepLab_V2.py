'''
Lionzhu 0418
THis network is changed from deeplab_v3
modify channel num from ori py
'''

import tensorflow as tf

def LITS_DLab(tensor_in, BN_FLAG, BATCHSIZE, IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH, CLASSNUM):
    print(tensor_in.get_shape())

    #------------------- pre convolutions with 3 conv layer of 3x3 and pool----------------------------------
    with tf.variable_scope('PreConv_1'):
        conv_res = conv_layer(kernel_size=3, in_put=tensor_in, in_channel=1,
                              out_channel=32, c_stride=2, name='Conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BN')
        pre1_res = tf.nn.relu(bn_res)  # out channel = 32, downsample to 1/2

    with tf.variable_scope('PreConv_2'):
        conv_res = conv_layer(kernel_size=3, in_put=pre1_res, in_channel=32,
                              out_channel=32, c_stride=1, name='Conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BN')
        pre2_res = tf.nn.relu(bn_res)  # out channel = 32

    with tf.variable_scope('PreConv_3'):
        conv_res = conv_layer(kernel_size=3, in_put=pre2_res, in_channel=32,
                              out_channel=64, c_stride=1, name='Conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BN')
        pre3_res = tf.nn.relu(bn_res)  # out channel = 64



    #-------------------resnet conv block 1-----------------------------------------------------------------------
    with tf.variable_scope('Block1_1x1'):
        conv_res = conv_layer(kernel_size= 1, in_put= pre3_res, in_channel= 64,
                              out_channel= 32, c_stride=1, name= 'Conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BN')
        bk1_1_res = tf.nn.relu(bn_res)  # out channel = 32

    with tf.variable_scope('Block1_3x3'):
        conv_res = conv_layer(kernel_size= 3, in_put= bk1_1_res, in_channel= 32,
                              out_channel= 32, c_stride=1, name= 'Conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BN')
        bk1_3_res = tf.nn.relu(bn_res)  # out channel = 32

    with tf.variable_scope('Block1_1x1_T'):
        conv_res = conv_layer(kernel_size= 1, in_put= bk1_3_res, in_channel= 32,
                              out_channel= 64, c_stride=1, name= 'Conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BN')
        bk1_1T_res = tf.nn.relu(bn_res)  # out channel = 128

    with tf.variable_scope('Block1_BR'):
        conv_res = conv_layer(kernel_size= 1, in_put= pre3_res, in_channel= 64,
                              out_channel= 64, c_stride=1, name= 'Conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BN')
        bk1_br_res = tf.nn.relu(bn_res)  # out channel = 64

    with tf.variable_scope('Block1_SUM'):
        bk1_sum = tf.add(bk1_1T_res, bk1_br_res) # out channel = 64


    #-------------------resnet conv block 2--------------------------------------------------------------------
    with tf.variable_scope('Block2_1x1'):
        conv_res = conv_layer(kernel_size=1, in_put=bk1_sum, in_channel=64,
                              out_channel=32, c_stride=1, name='Conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BN')
        bk2_1_res = tf.nn.relu(bn_res)  # out channel = 32

    with tf.variable_scope('Block2_3x3'):
        conv_res = conv_layer(kernel_size= 3, in_put= bk2_1_res, in_channel= 32,
                              out_channel= 32, c_stride=1, name='Conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BN')
        bk2_3_res = tf.nn.relu(bn_res)  # out channel = 32

    with tf.variable_scope('Block2_1x1_T'):
        conv_res = conv_layer(kernel_size= 1, in_put= bk2_3_res, in_channel= 32,
                              out_channel= 64, c_stride=1, name= 'Conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BN')
        bk2_1T_res = tf.nn.relu(bn_res)  # out channel = 64

    with tf.variable_scope('Block2_SUM'):
        bk2_sum = tf.add(bk1_sum, bk2_1T_res) # out channel =64


    with tf.variable_scope('Pool_1'):
        pool_res = max_pool3d(bk2_sum, ksize= 2, stride= 2)  # out channel = 64 downsample to 1/4

    #-----------------resnet conv block 3--------------------------------------------------------------------
    with tf.variable_scope('Block3_1x1'):
        conv_res = conv_layer(kernel_size=1, in_put= pool_res, in_channel= 64,
                              out_channel= 32, c_stride=1, name='Conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BN')
        bk3_1_res = tf.nn.relu(bn_res)  # out channel = 32

    with tf.variable_scope('Block3_3x3'):
        conv_res = conv_layer(kernel_size=3, in_put=bk3_1_res, in_channel= 32,
                              out_channel= 32, c_stride=1, name='Conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BN')
        bk3_3_res = tf.nn.relu(bn_res)  # out channel = 32

    with tf.variable_scope('Block3_1_T'):
        conv_res = conv_layer(kernel_size=1, in_put= bk3_3_res, in_channel= 32,
                              out_channel= 128, c_stride=1, name='Conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BN')
        bk3_1T_res = tf.nn.relu(bn_res)  # out channel = 128

    with tf.variable_scope('Block3_BR'):
        conv_res = conv_layer(kernel_size=1, in_put= pool_res, in_channel= 64,
                              out_channel= 128, c_stride=1, name= 'Conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BN')
        bk3_br_res = tf.nn.relu(bn_res)  # out channel = 128

    with tf.variable_scope('Block3_SUM'):
        bk3_sum = tf.add(bk3_1T_res, bk3_br_res) # out channel = 128


    #----------------resnet conv block 4-------------------------------------------------------------------
    with tf.variable_scope('Block4_1x1'):
        conv_res = conv_layer(kernel_size=1, in_put=bk3_sum, in_channel=128,
                              out_channel=64, c_stride=1, name='Conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BN')
        bk4_1_res = tf.nn.relu(bn_res)  # out channel = 64

    with tf.variable_scope('Block4_3x3'):
        conv_res = conv_layer(kernel_size=3, in_put=bk4_1_res, in_channel=64,
                              out_channel=64, c_stride=1, c_rate= 2, name='Conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BN')
        bk4_3_res = tf.nn.relu(bn_res)  # out channel = 64

    with tf.variable_scope('Block4_1_T'):
        conv_res = conv_layer(kernel_size=1, in_put=bk4_3_res, in_channel=64,
                              out_channel=128, c_stride=1, name='Conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BN')
        bk4_1T_res = tf.nn.relu(bn_res)  # out channel = 512

    with tf.variable_scope('Block4_BR'):
        conv_res = conv_layer(kernel_size=1, in_put=bk3_sum, in_channel=128,
                              out_channel=128, c_stride=1, name='Conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BN')
        bk4_br_res = tf.nn.relu(bn_res)  # out channel = 128

    with tf.variable_scope('Block4_SUM'):
        bk4_sum = tf.add(bk4_1T_res, bk4_br_res)   # out channel = 128


    with tf.variable_scope('Pool_2'):
        pool2_res = max_pool3d(bk4_sum, ksize= 3, stride= 2)  # out channel = 128 downsample to 1/8


    #----------------------atrous spatial pyramid pooling----------------------------------------------
    with tf.variable_scope('ASPP_BR1'):
        conv_res = conv_layer(kernel_size=1, in_put= pool2_res, in_channel= 128,
                              out_channel= 64, c_stride= 1, name = 'Conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BN')
        aspp_br1_res = tf.nn.relu(bn_res)  # out channel = 64

    with tf.variable_scope('ASPP_BR2_1'):
        conv_res = conv_layer(kernel_size=1, in_put=pool2_res, in_channel=128,
                              out_channel=64, c_stride=1, name='Conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BN')
        aspp_br2_1_res = tf.nn.relu(bn_res)  # out channel = 64

    with tf.variable_scope('ASPP_BR2_3'):
        conv_res = conv_layer(kernel_size=3, in_put=aspp_br2_1_res, in_channel=64,
                              out_channel=64, c_stride=1, c_rate= 2,  name='Conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BN')
        aspp_br2_3_res = tf.nn.relu(bn_res)  # out channel = 64

    with tf.variable_scope('ASPP_BR3_1'):
        conv_res = conv_layer(kernel_size=1, in_put=pool2_res, in_channel=128,
                              out_channel=64, c_stride=1, name='Conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BN')
        aspp_br3_1_res = tf.nn.relu(bn_res)  # out channel = 64

    with tf.variable_scope('ASPP_BR3_3'):
        conv_res = conv_layer(kernel_size=3, in_put=aspp_br3_1_res, in_channel=64,
                              out_channel=64, c_stride=1, c_rate= 4,  name='Conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BN')
        aspp_br3_3_res = tf.nn.relu(bn_res)  # out channel = 64

    with tf.variable_scope('ASPP_BR4_1'):
        conv_res = conv_layer(kernel_size=1, in_put=pool2_res, in_channel=128,
                              out_channel=64, c_stride=1, name='Conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BN')
        aspp_br4_1_res = tf.nn.relu(bn_res)  # out channel = 64

    with tf.variable_scope('ASPP_BR4_3'):
        conv_res = conv_layer(kernel_size=3, in_put=aspp_br4_1_res, in_channel=64,
                              out_channel=64, c_stride=1, c_rate= 8,  name='Conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BN')
        aspp_br4_3_res = tf.nn.relu(bn_res)  # out channel = 64
    with tf.variable_scope('Concat'):
        concat_res = tf.concat([aspp_br1_res, aspp_br2_3_res, aspp_br3_3_res, aspp_br4_3_res],
                               axis= 4, name= 'Concat') # out channel 256

    #---------------------conv after ASPP----------------------------------------------------------------------
    with tf.variable_scope('Conv_after_Aspp'):
        conv_res = conv_layer(kernel_size= 3, in_put= concat_res, in_channel= 256,
                              out_channel= 128, c_stride=1, name= 'Conv')
        bn_res = tf.layers.batch_normalization(conv_res, momentum=0.9, training=BN_FLAG, name='BN')
        conv_aspp_res = tf.nn.relu(bn_res)  # out channel = 128


    #---------------------deconv block--------------------------------------------------------------------------
    shape_deconv1 = pool2_res.get_shape()
    with tf.variable_scope('Deconv1'):
        deconv = conv3d_transpose_layer(kernel_size= 3, in_put= conv_aspp_res, out_shape= tf.shape(pool2_res),
                                        in_channel= 128, out_channel= 128, c_stride=1, name= 'Deconv')
        bn_res = tf.layers.batch_normalization(deconv, momentum=0.9, training=BN_FLAG, name='BN')
        deconv1_res = tf.nn.relu(bn_res)
        fuse1_res = tf.add(deconv1_res, pool2_res)       # out channel 128

    with tf.variable_scope('Deconv2'):
        deconv = conv3d_transpose_layer(kernel_size=3, in_put= fuse1_res, out_shape=tf.shape(pool_res),
                                        in_channel= 128, out_channel= 64, c_stride=2, name= 'Deconv')
        bn_res = tf.layers.batch_normalization(deconv, momentum=0.9, training=BN_FLAG, name='BN')
        deconv2_res = tf.nn.relu(bn_res)
        fuse2_res = tf.add(deconv2_res, pool_res)    # out channel 64

    with tf.variable_scope('Deconv3'):
        tensor_in_shape = tf.shape(tensor_in)
        deconv3_shape = tf.stack([tensor_in_shape[0], tensor_in_shape[1], tensor_in_shape[2], tensor_in_shape[3], CLASSNUM])
        deconv = conv3d_transpose_layer(kernel_size= 4, in_put= fuse2_res, out_shape= deconv3_shape,
                                        in_channel= 64, out_channel= CLASSNUM, c_stride=4, name= 'BN')
        bn_res = tf.layers.batch_normalization(deconv, momentum=0.9, training=BN_FLAG, name='BN')
        deconv3_res = tf.nn.relu(bn_res)  # out channel 3, shape=[batchsize, depth, height, width, channel]

    with tf.variable_scope('Argmax'):
        annotation_pred = tf.argmax(deconv3_res, axis =4, name='prediction')

    return tf.expand_dims(annotation_pred, dim=4), deconv3_res
   # return tf.expand_dims(annotation_pred, dim=4), deconv3_res



def conv_layer(kernel_size, in_put, in_channel, out_channel, c_stride=1, c_rate=1, name=None):
    with tf.variable_scope(name):
        conv_weights, conv_bias = get_var(kernel_size, in_channel, out_channel, name)
        conv = tf.nn.conv3d(in_put, conv_weights, strides=[1, c_stride, c_stride, c_stride, 1], padding='SAME',
                            dilations=[1, 1, c_rate, c_rate, 1])
        conv_addbias = tf.nn.bias_add(conv, conv_bias)
        return conv_addbias

def get_var(kernel_size, in_channel, out_channel, name=None):
    weights = tf.get_variable('w', shape=[kernel_size, kernel_size, kernel_size, in_channel, out_channel],
                              initializer=tf.contrib.layers.xavier_initializer())
    bias = tf.get_variable('b', [out_channel], initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))

    return weights, bias

def max_pool3d(in_tensor, ksize, stride):
    maxpool_res = tf.nn.max_pool3d(in_tensor, ksize=[1, ksize, ksize, ksize, 1],
                                   strides=[1, stride, stride, stride, 1], padding='SAME', name='MaxPool')
    return maxpool_res

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
