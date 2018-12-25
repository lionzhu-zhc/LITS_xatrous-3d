#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2018/10/12 15:37
# @Author  : Lionzhu
# Augmentation of the downsampled mats from folder SuperResolution_15
# the shape is

import os
import numpy as np
import scipy.io as sio

ori_path = 'E:/Cerebral Infarction/SuperResolution/SR_30/'
label_path = 'E:/Cerebral Infarction/SuperResolution/Perf_mat/'
# nx512x512
#change dir here-------------------------------------------------------
dst_path = 'E:/Cerebral Infarction/SuperResolution/exp_data3/'

ImgWidth = 512
ImgHeight = 512
PatchSize = 32
Step = 16
# 30*Step + Pachsize = 512, n=30+1
n_patch = 31

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

def Norm(x):
    return (x / 1000)

def PrepareTraining(AUG):
    if AUG:
        patients = os.listdir(ori_path)
        for i in range(18):
            mats = os.listdir(os.path.join(ori_path, patients[i]))
            for j in range(4):
                name_pre = mats[j][:-4]
                ori_mat = sio.loadmat(os.path.join(ori_path, patients[i], mats[j]))
                vol_norm = ori_mat['Mat']
                # vol_norm = ZeroOneNorm(vol_norm)
                ori_mat = sio.loadmat(os.path.join(label_path, patients[i], mats[j]))
                label = ori_mat['Mat']

                for n in range (n_patch):
                    for m in range(n_patch):
                        vol_patch = vol_norm[:, m*Step: (m*Step + PatchSize), n*Step: (n*Step + PatchSize)]   #30xPSxPS
                        label_patch = label[:, m*Step: (m*Step + PatchSize), n*Step: (n*Step + PatchSize)]
                        vol_patch = Norm(vol_patch)
                        label_patch = Norm(label_patch)
                        # label_patch = label_patch[:, 6:26, 6:26]  # crop 30x20x20

                        np.save(dst_path + 'train/' + 'vol/' + patients[i] + '_ori' + '_' + name_pre + '_' + str(n*n_patch+m) + '.npy', vol_patch.astype(np.float32))
                        np.save(dst_path + 'train/' + 'label/' + patients[i] + '_ori' + '_' + name_pre + '_' + str(n*n_patch+m) + '.npy', label_patch.astype(np.float32))

                #-------- rot 90--------------
                vol_norm_rot90 = np.zeros_like(vol_norm)
                label_rot90 = np.zeros_like(label)
                for m in range(vol_norm.shape[0]):
                    vol_norm_rot90[m, ...] = np.rot90(vol_norm[m, ...])
                    label_rot90[m, ...] = np.rot90(label[m, ...])
                for n in range (n_patch):
                    for m in range(n_patch):
                        vol_patch = vol_norm_rot90[:, m*Step: (m*Step + PatchSize), n*Step: (n*Step + PatchSize)]   #30x32x32
                        label_patch = label_rot90[:, m*Step: (m*Step + PatchSize), n*Step: (n*Step + PatchSize)]
                        vol_patch = Norm(vol_patch)
                        label_patch = Norm(label_patch)
                        # label_patch = label_patch[:, 6:26, 6:26]  # crop 30x20x20

                        np.save(dst_path + 'train/' + 'vol/' + patients[i] + '_rot' + '_' + name_pre + '_' + str(n*n_patch+m) + '.npy', vol_patch.astype(np.float32))
                        np.save(dst_path + 'train/' + 'label/' + patients[i] + '_rot' + '_' + name_pre + '_' + str(n*n_patch+m) + '.npy', label_patch.astype(np.float32))

                # -------- mirror--------------
                vol_norm_mir = np.zeros_like(vol_norm)
                label_mir = np.zeros_like(label)
                for m in range(vol_norm.shape[0]):
                    vol_norm_mir[m, ...] = np.fliplr(vol_norm[m, ...])
                    label_mir[m, ...] = np.fliplr(label[m, ...])
                for n in range(n_patch):
                    for m in range(n_patch):
                        vol_patch = vol_norm_mir[:, m*Step: (m*Step + PatchSize), n*Step: (n*Step + PatchSize)]  # 30x32x32
                        label_patch = label_mir[:, m*Step: (m*Step + PatchSize), n*Step: (n*Step + PatchSize)]
                        vol_patch = Norm(vol_patch)
                        label_patch = Norm(label_patch)
                        # label_patch = label_patch[:, 6:26, 6:26]  # crop 30x20x20

                        np.save(dst_path + 'train/' + 'vol/' + patients[i] + '_mir' + '_' + name_pre + '_' + str(
                            n * n_patch + m) + '.npy', vol_patch.astype(np.float32))
                        np.save(dst_path + 'train/' + 'label/' + patients[i] + '_mir' + '_' + name_pre + '_' + str(
                            n * n_patch + m) + '.npy', label_patch.astype(np.float32))
    #-------------------------------------------------------------------------------------------------------------------------
    else:
        patients = os.listdir(ori_path)
        for i in range(18):
            mats = os.listdir(os.path.join(ori_path, patients[i]))
            for j in range(4):
                name_pre = mats[j][:-4]
                ori_mat = sio.loadmat(os.path.join(ori_path, patients[i], mats[j]))
                vol_norm_t = ori_mat['Mat']    # shape: [Depth, Width, Height]
                vol_norm = Norm(vol_norm_t)
                ori_mat = sio.loadmat(os.path.join(label_path, patients[i], mats[j]))
                label_t = ori_mat['Mat']
                label = Norm(label_t)

                for n in range(n_patch):
                    for m in range(n_patch):
                        vol_patch = vol_norm[:, m * Step: (m * Step + PatchSize),
                                    n * Step: (n * Step + PatchSize)]  # Nx32x32
                        label_patch = label[:, m * Step: (m * Step + PatchSize), n * Step: (n * Step + PatchSize)]
                        shap = vol_patch.shape
                        print(shap)

                        #label_patch = label_patch[:, 6:26, 6:26]  # crop 30x20x20

                        np.save(dst_path + 'train_noaug/' + 'vol/' + patients[i] + '_ori' + '_' + name_pre + '_' + str(
                            n * n_patch + m) + '.npy', vol_patch.astype(np.float32))
                        np.save(dst_path + 'train_noaug/' + 'label/' + patients[i] + '_ori' + '_' + name_pre + '_' + str(
                            n * n_patch + m) + '.npy', label_patch.astype(np.float32))



def PrepareTest():
    patients = os.listdir(ori_path)
    for i in range(18,24):
        mats = os.listdir(os.path.join(ori_path, patients[i]))
        for j in range(4):
            name_pre = mats[j][:-4]
            ori_mat = sio.loadmat(os.path.join(ori_path, patients[i], mats[j]))
            vol_norm_t = ori_mat['Mat']
            vol_norm = Norm(vol_norm_t)
            shap = vol_norm.shape
            print(shap)
            # vol_dst = np.zeros((shap[0], shap[1] + 12, shap[2] + 12))
            # vol_dst[:, 6:518, 6:518] = vol_norm
            ori_mat = sio.loadmat(os.path.join(label_path, patients[i], mats[j]))
            label_t = ori_mat['Mat']
            label = Norm(label_t)

            np.save(
                dst_path + 'test/' + 'vol/' + patients[i] + '_' + name_pre + '.npy',
                vol_norm.astype(np.float32))
            np.save(
                dst_path + 'test/' + 'label/' + patients[i] + '_' + name_pre + '.npy',
                label.astype(np.float32))


if __name__ == '__main__':
    AUG_Flag = False
    PrepareTraining(AUG_Flag)
    PrepareTest()
    print('ok')