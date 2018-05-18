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
    one_hot = utils.to_onehot(ground_truth, class_num= 2)

    if weight_map is not None:
        n_classes = pred.shape[1].value  #depth??????
        weight_map_nclass = tf.reshape(tf.tile(weight_map, [n_classes]), pred.get_shape())   # weigth_map duplicate nclass times
        dice_numerator = 2.0 * tf.sparse_reduce_sum(weight_map_nclass * one_hot *pred, axis= [0])
        dice_denominator = tf.reduce_sum(pred * weight_map_nclass, axis=[0]) \
                           + tf.sparse_reduce_sum(weight_map_nclass *one_hot, axis= [0])

    else:
        dice_numerator = 2.0 * tf.sparse_reduce_sum(one_hot * pred, axis= [0])
        dice_denominator = tf.reduce_sum(pred, axis= [0]) + tf.sparse_reduce_sum(one_hot, axis= [0])

    dice_coe = dice_numerator / (dice_denominator + smooth)
    return 1-tf.reduce_mean(dice_coe)


def cross_entropy_loss(pred, ground_truth, class_weight = None):
    if class_weight is not None:
        current_weight= tf.gather(class_weight, tf.squeeze(ground_truth, axis=4))
        loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(
                logits=pred, labels=tf.squeeze(ground_truth, squeeze_dims=[4]), weights=current_weight))

    return  loss


def dice_sqaure(pred, ground_truth, weight_map = None):
    smooth = 1e-5

    if (len(pred.shape)) == (len(ground_truth.shape)):
        ground_truth = ground_truth[...,-1]   # discard the channel axis

    #one_hot = utils.labels_to_onehot(ground_truth, class_num= tf.shape(pred)[-1])
    one_hot = utils.to_onehot(ground_truth, class_num= tf.shape(pred)[-1])

    if weight_map is not None:
        print('weight_map not none')

    else:
        dice_numerator = 2.0 * tf.reduce_sum(one_hot * pred, axis= [0])

        dice_denominator = tf.reduce_sum(tf.square(pred), axis=[0]) + \
                           tf.reduce_sum(one_hot, axis=[0])

    dice_coe = dice_numerator / (dice_denominator + smooth)
    return 1- tf.reduce_mean(dice_coe)


def focal_loss(pred, ground_truth, gamma = 2.0, alpha = 0.25):

    if (len(pred.shape)) == (len(ground_truth.shape)):   # gt has the channel dim =1
        ground_truth = ground_truth[...,-1]   # discard the channel axis
    gt_onehot = utils.to_onehot(ground_truth, class_num= tf.shape(pred)[-1])

    sigmoid_p = tf.nn.sigmoid(pred)
    zeros = tf.zeros_like(sigmoid_p, dtype= sigmoid_p.dtype)

    #for positive pred, only need to consider front part loss, back part is 0
    # gt > zeros == z=1, so positive coe = z- p
    pos_p_sub = tf.where(gt_onehot > zeros, gt_onehot - sigmoid_p, zeros)

    #for negative pred, only need to consider back part loss, front part loss is 0
    # gt > zeros == z=1, so negative coe = 0
    neg_p_sub = tf.where(gt_onehot > zeros, zeros, sigmoid_p)

    entropy_cross = -alpha *(pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                    - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1 - sigmoid_p, 1e-8, 1.0))

    return tf.reduce_sum(entropy_cross)
