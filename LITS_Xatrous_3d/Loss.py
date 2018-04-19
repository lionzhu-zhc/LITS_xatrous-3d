'''
Lion Zhu
compute dif loss fun
'''

import tensorflow as tf
import utils

def dice(pred, ground_truth, weight_map = None):
    '''
    the classical dice coef
    :param pred:
    :param ground_truth:
    :param weight_map:
    :return:
    '''

    smooth = 1e-5

    pred = tf.cast(pred, tf.float32)
    if (len(pred.shape)) == (len(ground_truth.shape)):
        ground_truth = ground_truth[...,-1]   # discard the channel axis
    one_hot = utils.labels_to_onehot(ground_truth, class_num= tf.shape(pred)[-1])

    if weight_map is not None:
        n_classes = pred.shape[1].value  #depth??????
        weight_map_nclass = tf.reshape(tf.tile(weight_map, [n_classes]), pred.get_shape())   # weigth_map duplicate nclass times
        dice_numerator = 2.0 * tf.sparse_reduce_sum(weight_map_nclass * one_hot *pred, reduction_axes= [0])
        dice_denominator = tf.reduce_sum(pred * weight_map_nclass, reduction_indices=[0]) \
                           + tf.sparse_reduce_sum(weight_map_nclass *one_hot, reduction_axes= [0])

    else:
        dice_numerator = 2.0 * tf.sparse_reduce_sum(one_hot * pred, reduction_axes= [0])
        dice_denominator = tf.reduce_sum(pred, reduction_indices= [0]) + tf.sparse_reduce_sum(one_hot, reduction_axes= [0])

    dice_coe = dice_numerator / (dice_denominator + smooth)
    return 1-tf.reduce_mean(dice_coe)
