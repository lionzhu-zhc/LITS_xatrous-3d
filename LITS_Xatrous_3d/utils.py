import numpy as np
import random
import tensorflow as tf
import os
import scipy.misc as smc

def get_data_train(trainPath):
    dirs_train = os.listdir(trainPath + 'vol/')
    samples = random.choice(dirs_train)
    print(samples)
    vol_batch = np.load(trainPath + 'vol/' + samples)
    seg_batch = np.load(trainPath + 'seg/' + samples)
    vol_batch = np.transpose(vol_batch, [2, 0, 1])  # shape[depth, width, height]
    seg_batch = np.transpose(seg_batch, [2, 0, 1])
    vol_batch = np.expand_dims(vol_batch, axis=0)
    seg_batch = np.expand_dims(seg_batch, axis=0)
    vol_batch = np.expand_dims(vol_batch, axis=4)
    seg_batch = np.expand_dims(seg_batch, axis=4)
    vol_batch.astype(np.float32)
    seg_batch.astype(np.int32)

    return vol_batch, seg_batch



def get_data_test(testPath, tDir):

    vol_batch = np.load(testPath + 'vol/' +tDir)
    seg_batch = np.load(testPath + 'seg/' +tDir)
    vol_batch = np.transpose(vol_batch, [2, 0, 1])  # shape[depth, width, height]
    seg_batch = np.transpose(seg_batch, [2, 0, 1])
    vol_batch = np.expand_dims(vol_batch, axis=0)
    seg_batch = np.expand_dims(seg_batch, axis=0)
    vol_batch = np.expand_dims(vol_batch, axis=4)
    seg_batch = np.expand_dims(seg_batch, axis=4)
    vol_batch.astype(np.float32)
    seg_batch.astype(np.int32)
    return vol_batch,

def get_data_test_xianjin(testPath, tDir):
    vol_batch = np.load(testPath +tDir)
    vol_batch = np.transpose(vol_batch, [2, 0, 1])  # shape[depth, width, height]
    vol_batch = np.expand_dims(vol_batch, axis=0)
    vol_batch = np.expand_dims(vol_batch, axis=4)
    vol_batch.astype(np.float32)
    return vol_batch




def save_imgs(resultPath, name_pre, label_batch, pred_batch, IMAGE_DEPTH, IMAGE_WIDTH, IMAGE_HEIGHT):
    str_split = name_pre.split('-')
    if not(os.path.exists(resultPath + 'imgs/' + str_split[0])):
        os.makedirs(resultPath + 'imgs/' + str_split[0])

    file_index = int(str_split[-1]) * IMAGE_DEPTH

    for dept in range(IMAGE_DEPTH):
        label_img_mat = np.zeros((3, IMAGE_WIDTH, IMAGE_HEIGHT))
        pred_img_mat = np.zeros((3, IMAGE_WIDTH, IMAGE_HEIGHT))
        label_slice = label_batch[:, :, dept]
        pred_slice = pred_batch[:, :, dept]

        label_cord = np.where(label_slice == 0)
        label_img_mat[0, label_cord[0], label_cord[1]] = 128
        label_img_mat[1, label_cord[0], label_cord[1]] = 128
        label_img_mat[2, label_cord[0], label_cord[1]] = 128

        label_cord = np.where(label_slice == 1)
        label_img_mat[0, label_cord[0], label_cord[1]] = 255
        label_img_mat[1, label_cord[0], label_cord[1]] = 69
        label_img_mat[2, label_cord[0], label_cord[1]] = 0

        label_cord = np.where(label_slice == 2)
        label_img_mat[0, label_cord[0], label_cord[1]] = 64
        label_img_mat[1, label_cord[0], label_cord[1]] = 0
        label_img_mat[2, label_cord[0], label_cord[1]] = 128

        label_img_mat = np.transpose(label_img_mat, [1, 2, 0])

        pred_cord = np.where(pred_slice == 0)
        pred_img_mat[0, pred_cord[0], pred_cord[1]] = 128
        pred_img_mat[1, pred_cord[0], pred_cord[1]] = 128
        pred_img_mat[2, pred_cord[0], pred_cord[1]] = 128

        pred_cord = np.where(pred_slice == 1)
        pred_img_mat[0, pred_cord[0], pred_cord[1]] = 255
        pred_img_mat[1, pred_cord[0], pred_cord[1]] = 69
        pred_img_mat[2, pred_cord[0], pred_cord[1]] = 0

        pred_cord = np.where(pred_slice == 2)
        pred_img_mat[0, pred_cord[0], pred_cord[1]] = 64
        pred_img_mat[1, pred_cord[0], pred_cord[1]] = 0
        pred_img_mat[2, pred_cord[0], pred_cord[1]] = 128

        pred_img_mat = np.transpose(pred_img_mat, [1, 2, 0])


        smc.toimage(label_img_mat, cmin=0.0, cmax=255).save(
            resultPath + 'imgs/' + str_split[0] + '/%d-mask.png' % (file_index + dept))
        smc.toimage(pred_img_mat, cmin=0.0, cmax=255).save(
            resultPath + 'imgs/' + str_split[0] + '/%d-pred.png' % (file_index + dept))


def save_imgs_xianjin(resultPath, name_pre, pred_batch, IMAGE_DEPTH, IMAGE_WIDTH, IMAGE_HEIGHT):
    str_split = name_pre.split('-')
    if not(os.path.exists(resultPath + 'imgs/' + str_split[0])):
        os.makedirs(resultPath + 'imgs/' + str_split[0])

    file_index = int(str_split[-1]) * IMAGE_DEPTH

    for dept in range(IMAGE_DEPTH):
        pred_img_mat = np.zeros((3, IMAGE_WIDTH, IMAGE_HEIGHT))
        pred_slice = pred_batch[:, :, dept]

        pred_cord = np.where(pred_slice == 0)
        pred_img_mat[0, pred_cord[0], pred_cord[1]] = 128
        pred_img_mat[1, pred_cord[0], pred_cord[1]] = 128
        pred_img_mat[2, pred_cord[0], pred_cord[1]] = 128

        pred_cord = np.where(pred_slice == 1)
        pred_img_mat[0, pred_cord[0], pred_cord[1]] = 255
        pred_img_mat[1, pred_cord[0], pred_cord[1]] = 69
        pred_img_mat[2, pred_cord[0], pred_cord[1]] = 0

        pred_cord = np.where(pred_slice == 2)
        pred_img_mat[0, pred_cord[0], pred_cord[1]] = 64
        pred_img_mat[1, pred_cord[0], pred_cord[1]] = 0
        pred_img_mat[2, pred_cord[0], pred_cord[1]] = 128

        pred_img_mat = np.transpose(pred_img_mat, [1, 2, 0])

        smc.toimage(pred_img_mat, cmin=0.0, cmax=255).save(
            resultPath + 'imgs/' + str_split[0] + '/%d-pred.png' % (file_index + dept))


def save_npys(resultPath, name_pre, label_batch, pred_batch):
    np.save(resultPath + 'npys/' + name_pre + '-mask.npy', label_batch)
    np.save(resultPath + 'npys/' + name_pre + '-pred.npy', pred_batch)

def save_npys_xianjin(resultPath, name_pre, pred_batch):
    np.save(resultPath + 'npys/' + name_pre + '-pred.npy', pred_batch)


def labels_to_onehot(lables, class_num = 1):
    '''
    :param lables:  shape [batchsize, depth, height, width], 4D, no channel axis
    :param class_num:
    :return:
    '''

    if isinstance(class_num, tf.Tensor):
        class_num_tf = tf.to_int32(class_num)
    else:
        class_num_tf = tf.constant(class_num, tf.int32)
    in_shape = tf.shape(lables)
    out_shape = tf.concat([in_shape, tf.reshape(class_num_tf, (1,))], 0) # add a extra axis for classNum, 5D

    if class_num == 1:
        return tf.reshape(lables, out_shape)
    else:
        lables = tf.reshape(lables, (-1,)) # squeeze labels to one row x N cols vector [0,0,0,1,......]
        dense_shape = tf.stack([tf.shape(lables)[0], class_num_tf], 0)   # denshape [N cols , classNum]

        lables = tf.to_int64(lables)
        ids = tf.range(tf.to_int64(dense_shape[0]), dtype= tf.int64)  # ids is a 1xN vector as[0,1,2,3...., N-1]
        ids = tf.stack([ids, lables], axis= 1)  #ids is N x clsNum mat
        one_hot = tf.SparseTensor(indices= ids, values= tf.ones_like(lables, dtype= tf.float32), dense_shape = tf.to_int64(dense_shape))
        one_hot = tf.sparse_reshape(one_hot, out_shape)
        return tf.cast(one_hot, tf.float32)