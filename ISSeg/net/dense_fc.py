'''
this is network of Full Conv Densenet
Lionzhu-list 2018-07-27
'''

import tensorflow as tf

NUMPOOL = 2
FIRST_LAYER_FILTERS = 48
GROWTH_RATE = 12
LAYERS_PER_BLOCK = [5] * (2 * NUMPOOL + 1)

def build_dense_fc(tensor_in, BN_FLAG, BATCHSIZE, CLASSNUM, IMGCHANNEL, keep_prob):
    print(tensor_in.get_shape())
    # the first conv layer
    stack = BN_Relu_Conv(in_put= tensor_in, out_channel= FIRST_LAYER_FILTERS, 
                        keep_prob= 0.8, BN_FLAG= BN_FLAG, kernel_size= 3, name= 'First_Conv')

    n_filters = FIRST_LAYER_FILTERS

    skip_list= []

    #--------------------down scale layers--------------------------------------------------------
    for i in range(NUMPOOL):
        for j in range(LAYERS_PER_BLOCK[i]):
            # here ignore the bottle layer of the ori densenet
            l = bottleneck_layer(stack, keep_prob= 0.8, BN_FLAG= BN_FLAG, name= 'DownConv_{}_{}'.format(i,j))
            stack = tf.concat([stack, l], axis =3, name= 'Concat')
        skip_list.append(stack)
        stack = transition_layer(stack, keep_prob= 0.8, BN_FLAG= BN_FLAG, reduction= 0.5, name= 'Down_{}'.format(i))

    skip_list = skip_list[::-1]       # reverse the order of features

    block_to_up = []

    # the middle denseblock

    for j in range (LAYERS_PER_BLOCK[NUMPOOL]):
        l = bottleneck_layer(stack, keep_prob= 0.8, BN_FLAG= BN_FLAG, name='MConv_{}'.format(j))
        block_to_up.append(l)
        stack = tf.concat([stack, l], axis= 3, name= 'Concat')

    #-------------------up scale layers-----------------------------------------------------------
    for i in range(NUMPOOL):
        stack = up_layer(skip_list[i], block_to_up, kernel_size= 3, stride= 2, name= 'Up_{}'.format(i))

        block_to_up = []

        for j in range(LAYERS_PER_BLOCK[NUMPOOL + i + 1]):
            l =bottleneck_layer(stack, keep_prob= 0.8, BN_FLAG= BN_FLAG, name= 'UpConv_{}_{}'.format(i,j))
            block_to_up.append(l)
            stack = tf.concat([stack, l], axis= 3, name = 'Concat')

    #----------------------the last 1x1 conv layer-----------------------------------------------
    conv = BN_Relu_Conv(stack, out_channel= CLASSNUM, keep_prob= 0.8, BN_FLAG= BN_FLAG, kernel_size= 1, name= 'LastConv')

    #---------------------softmax layer-----------------------------------------------------------
    with tf.variable_scope('Argmax'):
        annot_pred = tf.argmax(conv, axis= -1, name=  'Prediction')

    return conv, tf.expand_dims(annot_pred, axis= 3)


def BN_Relu_Conv(in_put, out_channel, keep_prob, BN_FLAG, kernel_size = 3, name = 'BN_Relu_Conv'):
    with tf.variable_scope(name):
        shap = in_put.get_shape().as_list()
        in_channel = shap[-1]
        res = tf.layers.batch_normalization(in_put, momentum= 0.9, training= BN_FLAG, name= 'BatchNorm')
        res = tf.nn.relu(res)
        res = conv2d_layer(kernel_size, res, in_channel, out_channel, keep_prob= keep_prob, name= name)
        return res

def bottleneck_layer(in_put, keep_prob, BN_FLAG, name = 'bottleneck'):
    with tf.variable_scope(name):
        shap = in_put.get_shape().as_list()
        in_channel = shap[-1]
        res = BN_Relu_Conv(in_put, 4*GROWTH_RATE, keep_prob, BN_FLAG, kernel_size= 1, name= 'Kernel_1')
        res = BN_Relu_Conv(res, GROWTH_RATE, keep_prob, BN_FLAG, kernel_size= 3, name= 'Kernel_3')
        return res

def transition_layer(in_put, keep_prob, BN_FLAG, reduction= 0.5,name = 'down_layer'):
    with tf.variable_scope(name):
        shap = in_put.get_shape().as_list()
        in_channel = shap[-1]
        out_channel = round(in_channel * reduction)
        res = BN_Relu_Conv(in_put, out_channel, keep_prob, BN_FLAG, kernel_size=1, name= name)
        res = pool2d_layer(res, ksize = 2, stride= 2)
        return res

def up_layer(skip_conn, block_to_up, kernel_size, stride, name= 'up_layer'):
    '''
    perform upscale on block_to_up by factor 2 and concat it with skip_conn
    :param skip_conn:
    :param block_to_up: to upsample
    :return:
    '''

    l = tf.concat(block_to_up, axis= 3, name= 'Concat')
    shap = l.get_shape().as_list()
    in_channel = shap[3]
    out_channel = in_channel * 2
    out_shape = tf.convert_to_tensor(list([shap[0], 2*shap[1], 2*shap[2], out_channel]))
    with tf.variable_scope(name):
        deconv_weights, deconv_bias = get_var_transpose(kernel_size, in_channel, out_channel, name)
        l = tf.nn.conv2d_transpose(l, deconv_weights, out_shape, strides= [1, stride, stride, 1], padding= 'SAME')
        l = tf.nn.bias_add(l, deconv_bias)
        l = tf.concat([l, skip_conn], axis= 3, name= 'Concat')

    return l

def conv2d_layer(kernel_size, in_put, in_channel, out_channel, c_stride = 1, c_rate = 1, keep_prob = 0.8, name = None):
    with tf.variable_scope(name):
        conv_weights, conv_bias = get_var(kernel_size, in_channel, out_channel, name)
        conv = tf.nn.conv2d(in_put, conv_weights, strides = [1, c_stride, c_stride, 1],
                            padding= 'SAME', dilations= [1, c_rate, c_rate, 1])
        conv = tf.nn.bias_add(conv, conv_bias)
        conv = tf.nn.dropout(conv, keep_prob)

        return conv

def get_var(kernel_size, in_channel, out_channel, name= None):
    weights = tf.get_variable('w', shape=[kernel_size, kernel_size, in_channel, out_channel],
                              initializer= tf.contrib.layers.xavier_initializer())
    bias = tf.get_variable('b', [out_channel], initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))

    return weights, bias

def pool2d_layer(in_put, ksize, stride):
    res = tf.nn.max_pool(in_put, ksize= [1, ksize, ksize, 1], 
                        strides= [1, stride, stride, 1], padding= 'SAME', name = 'Max_Pool')

    return res

def get_var_transpose(kernel_size, in_channel, out_channel, name= None):
    weights = tf.get_variable('w', shape=[kernel_size, kernel_size, out_channel, in_channel],
                              initializer=tf.contrib.layers.xavier_initializer())

    bias = tf.get_variable('b', [out_channel],
                           initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))
    return weights, bias



#def bottleneck_layer(in_put, growth_rate, BN_FLAG, keep_prob= 0.8, name = None):
#     with tf.variable_scope(name):
#         shap = in_put.get_shape()
#         in_channel = shap[-1]
#         res = BN_Relu_Conv(in_put, growth_rate, growth_rate*4, keep_prob, BN_FLAG, kernel_size= 1)
#         res = tf.nn.dropout(res, keep_prob)
#         res = BN_Relu_Conv(res, growth_rate*4, growth_rate, keep_prob, BN_FLAG, kernel_size= 3)
#         res = tf.nn.dropout(res, keep_prob)
#         return res
#
# def dense_block(in_put, n_layers, growth_rate, BN_FLAG, keep_prob, name):
#     with tf.variable_scope(name):
#         layer_concat = list()
#         layer_concat.append(in_put)
#
#         res = bottleneck_layer(in_put, growth_rate, BN_FLAG, keep_prob, name= 'Bottle')
#         layer_concat.append(res)
#
#         for i in range (n_layers - 1):
#             res = tf.concat(layer_concat, axis= 3, name= 'concat')
#             res = bottleneck_layer(res, growth_rate, BN_FLAG, keep_prob, name= 'Bottle')
#             layer_concat.append(res)
#
#         res = tf.concat(layer_concat, axis= 3, name= 'concat')
#         return res

# def up_layer(in_put, out_shape, in_channel, out_channel, kernel_size, stride, c_rate= 1, name= 'up_layer'):
#     with tf.variable_scope(name):
#         deconv_weights, deconv_bias = get_var_transpose(kernel_size, in_channel, out_channel)
#         deconv = tf.nn.conv2d_transpose(in_put, deconv_weights, out_shape, strides= [1, stride, stride, 1],
#                                         padding = 'SAME')
#         deconv = tf.nn.bias_add(deconv, deconv_bias)
#
#         return deconv









