import numpy as np
import random
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
    return vol_batch, seg_batch




def save_imgs(resultPath, name_pre, label_batch, pred_batch, IMAGE_DEPTH, IMAGE_WIDTH, IMAGE_HEIGHT):
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

        # change dir here ..........................................................................................
        smc.toimage(label_img_mat, cmin=0.0, cmax=255).save(
            resultPath + 'imgs/' + name_pre + '-%d-mask.png' % dept)
        smc.toimage(pred_img_mat, cmin=0.0, cmax=255).save(
            resultPath + 'imgs/' + name_pre + '-%d-pred.png' % dept)