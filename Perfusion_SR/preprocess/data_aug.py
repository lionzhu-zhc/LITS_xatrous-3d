#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2018/10/12 15:37
# @Author  : Lionzhu
# Augmentation of the downsampled mats from folder SuperResolution_15
# the shape is

import os
import numpy as np
import scipy.io as sio

ori_path = 'E:/Cerebral Infarction/SuperResolution/SR_15_30/'
label_path = 'E:/Cerebral Infarction/SuperResolution/Perfusion_mat/'
# 30x512x512
dst_path = 'E:/Cerebral Infarction/SuperResolution/exp_data2/'

ImgWidth = 512
ImgHeight = 512
PatchSize = 128
Step = 64
# 6*Step + Pachsize = 512

def ZeroOneNorm(x):
    min = np.min(x)
    max = np.max(x)
    x_norm = (x-min)/(max - min + 1e-5)
    return x_norm

def zscore(x):
    x_mean = np.mean(x)
    x_std = np.std(x) + 1e-5
    x_norm = (x - x_mean) / x_std
    return x_norm


def PrepareTraining(AUG):
    if AUG:
        patients = os.listdir(ori_path)
        for i in range(18):
            mats = os.listdir(os.path.join(ori_path, patients[i]))
            for j in range(4):
                name_pre = mats[j][:-4]
                ori_mat = sio.loadmat(os.path.join(ori_path, patients[i], mats[j]))
                vol_norm = ori_mat['Mat']
                # vol_norm = ZeroOneNorm(mat)
                ori_mat = sio.loadmat(os.path.join(label_path, patients[i], mats[j]))
                label = ori_mat['Mat']

                for n in range (31):
                    for m in range(31):
                        vol_patch = vol_norm[:, m*Step: (m*Step + PatchSize), n*Step: (n*Step + PatchSize)]   #30x32x32
                        label_patch = label[:, m*Step: (m*Step + PatchSize), n*Step: (n*Step + PatchSize)]
                        vol_patch = ZeroOneNorm(vol_patch)
                        label_patch = ZeroOneNorm(label_patch)
                        label_patch = label_patch[:, 6:26, 6:26]  # crop 30x20x20

                        np.save(dst_path + 'train/' + 'vol/' + patients[i] + '_ori' + '_' + name_pre + '_' + str(n*31+m) + '.npy', vol_patch.astype(np.float32))
                        np.save(dst_path + 'train/' + 'label/' + patients[i] + '_ori' + '_' + name_pre + '_' + str(n*31+m) + '.npy', label_patch.astype(np.float32))

                #-------- rot 90--------------
                vol_norm_rot90 = np.zeros_like(vol_norm)
                label_rot90 = np.zeros_like(label)
                for m in range(vol_norm.shape[0]):
                    vol_norm_rot90[m, ...] = np.rot90(vol_norm[m, ...])
                    label_rot90[m, ...] = np.rot90(label[m, ...])
                for n in range (31):
                    for m in range(31):
                        vol_patch = vol_norm_rot90[:, m*Step: (m*Step + PatchSize), n*Step: (n*Step + PatchSize)]   #30x32x32
                        label_patch = label_rot90[:, m*Step: (m*Step + PatchSize), n*Step: (n*Step + PatchSize)]
                        vol_patch = ZeroOneNorm(vol_patch)
                        label_patch = ZeroOneNorm(label_patch)
                        label_patch = label_patch[:, 6:26, 6:26]  # crop 30x20x20

                        np.save(dst_path + 'train/' + 'vol/' + patients[i] + '_rot' + '_' + name_pre + '_' + str(n*31+m) + '.npy', vol_patch.astype(np.float32))
                        np.save(dst_path + 'train/' + 'label/' + patients[i] + '_rot' + '_' + name_pre + '_' + str(n*31+m) + '.npy', label_patch.astype(np.float32))

                # -------- mirror--------------
                vol_norm_mir = np.zeros_like(vol_norm)
                label_mir = np.zeros_like(label)
                for m in range(vol_norm.shape[0]):
                    vol_norm_mir[m, ...] = np.fliplr(vol_norm[m, ...])
                    label_mir[m, ...] = np.fliplr(label[m, ...])
                for n in range(7):
                    for m in range(7):
                        vol_patch = vol_norm_mir[:, m*Step: (m*Step + PatchSize), n*Step: (n*Step + PatchSize)]  # 30x32x32
                        label_patch = label_mir[:, m*Step: (m*Step + PatchSize), n*Step: (n*Step + PatchSize)]
                        vol_patch = ZeroOneNorm(vol_patch)
                        label_patch = ZeroOneNorm(label_patch)
                        label_patch = label_patch[:, 6:26, 6:26]  # crop 30x20x20

                        np.save(dst_path + 'train/' + 'vol/' + patients[i] + '_mir' + '_' + name_pre + '_' + str(
                            n * 31 + m) + '.npy', vol_patch.astype(np.float32))
                        np.save(dst_path + 'train/' + 'label/' + patients[i] + '_mir' + '_' + name_pre + '_' + str(
                            n * 31 + m) + '.npy', label_patch.astype(np.float32))

    else:
        patients = os.listdir(ori_path)
        for i in range(18):
            mats = os.listdir(os.path.join(ori_path, patients[i]))
            for j in range(4):
                name_pre = mats[j][:-4]
                ori_mat = sio.loadmat(os.path.join(ori_path, patients[i], mats[j]))
                vol_norm = ori_mat['Mat']
                # vol_norm = ZeroOneNorm(mat)
                ori_mat = sio.loadmat(os.path.join(label_path, patients[i], mats[j]))
                label = ori_mat['Mat']

                for n in range(7):
                    for m in range(7):
                        vol_patch = vol_norm[:, m * Step: (m * Step + PatchSize),
                                    n * Step: (n * Step + PatchSize)]  # 30x32x32
                        label_patch = label[:, m * Step: (m * Step + PatchSize), n * Step: (n * Step + PatchSize)]
                        vol_patch = zscore(vol_patch)
                        #label_patch = ZeroOneNorm(label_patch)

                        #label_patch = label_patch[:, 6:26, 6:26]  # crop 30x20x20

                        np.save(dst_path + 'train_noaug_128/' + 'vol/' + patients[i] + '_ori' + '_' + name_pre + '_' + str(
                            n * 7 + m) + '.npy', vol_patch.astype(np.float32))
                        np.save(dst_path + 'train_noaug_128/' + 'label/' + patients[i] + '_ori' + '_' + name_pre + '_' + str(
                            n * 7 + m) + '.npy', label_patch.astype(np.float32))



def PrepareTest():
    patients = os.listdir(ori_path)
    for i in range(18,24):
        mats = os.listdir(os.path.join(ori_path, patients[i]))
        for j in range(4):
            name_pre = mats[j][:-4]
            ori_mat = sio.loadmat(os.path.join(ori_path, patients[i], mats[j]))
            mat = ori_mat['Mat']
            vol_norm = zscore(mat)
            shap = vol_norm.shape
            # vol_dst = np.zeros((shap[0], shap[1] + 12, shap[2] + 12))
            # vol_dst[:, 6:518, 6:518] = vol_norm
            ori_mat = sio.loadmat(os.path.join(label_path, patients[i], mats[j]))
            label = ori_mat['Mat']

            np.save(
                dst_path + 'test/' + 'vol/' + patients[i] + '_' + name_pre  + '.npy',
                vol_norm.astype(np.float32))
            np.save(
                dst_path + 'test/' + 'label/' + patients[i] + '_' + name_pre + '.npy',
                label.astype(np.float32))


if __name__ == '__main__':
    AUG_Flag = False
    PrepareTraining(AUG_Flag)
    # PrepareTest()
    print('ok')