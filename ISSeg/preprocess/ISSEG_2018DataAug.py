'''
read the mats from All folder
and prepare the training and test set
Lionzhu-list 20180726
180905 keep the CBF and CBV 2 channels
and discard the other channels
'''

import os
import numpy as np
import random
import scipy.io as sio
from PIL import Image
import cv2

dataPath = 'E:/ISSEG/Dataset/2018REGROUP/224'
cases = os.listdir(os.path.join(dataPath, 'vol_case'))
random.shuffle(cases)

def zscore(x):
    xshape = x.shape
    x_zs = np.zeros_like(x)
    for i in range (xshape[2]):
        sliceMean = np.mean(x[..., i])
        sliceStd = np.std(x[..., i]) + 1e-5
        slice = (x[..., i] - sliceMean) / sliceStd
        x_zs[..., i] = slice
    return x_zs

def ZeroOneNorm(x):
    x_shape = x.shape
    x_zs = np.zeros_like(x)
    for i in range(x_shape[2]):
        slice_min = np.min(x[...,i])
        slice_max = np.max(x[...,i])
        slice = (slice - slice_min) / (slice_max - slice_min + 1e-5)
        x_zs[..., i] = slice
    return x_zs


def aug(data, mode):
    shap = data.shape
    for m in mode:
        if m == 'mirror':
            mir = np.zeros_like(data)
            for i in range(shap[2]):
                mir[..., i] = np.fliplr(data[..., i])
        if m == 'rot90':
            rot90 = np.zeros_like(data)
            for i in range(shap[2]):
                rot90[..., i] = np.rot90(data[..., i])
        if m == 'rot-90':
            rot_90 = np.zeros_like(data)
            for i in range(shap[2]):
                rot_90[..., i] = np.rot90(data[..., i], 3)
        if m == 'trans':
            trans_1 = np.zeros_like(data)
            dx = random.randint(-15, 15)
            dy = random.randint(-15, 15)
            T_1 = np.float32([[1,0,dx], [0,1,dy]])
            trans_2 = np.zeros_like(data)
            dx = random.randint(-15, 15)
            dy = random.randint(-15, 15)
            T_2 = np.float32([[1, 0, -dx], [0, 1, -dy]])
            for i in range(shap[2]):
                trans_1[..., i] = cv2.warpAffine(data[..., i], T_1, (shap[0], shap[0]))
                trans_2[..., i] = cv2.warpAffine(data[..., i], T_2, (shap[0], shap[0]))
    return  mir, rot90, rot_90, trans_1, trans_2


def prepareTrain(augFlag):
    for i in range(70):
        print(cases[i])
        volPath = os.path.join(dataPath, 'vol_case', cases[i])
        segPath = os.path.join(dataPath, 'seg_case', cases[i])
        vols = os.listdir(volPath)
        segs = os.listdir(segPath)
        for f in vols:
            vol = sio.loadmat(os.path.join(volPath, f))
            volData = vol['vol']
            volData_zs = zscore(volData)
            volData_zs = volData_zs[..., 0:5]
            seg = sio.loadmat(os.path.join(segPath, f))
            segData = seg['seg']
            dstVolPath = os.path.join(dataPath, '5c', 'train', 'vol')
            if not os.path.exists(dstVolPath):
                os.makedirs(dstVolPath)
            dstSegPath = os.path.join(dataPath, '5c', 'train', 'seg')
            if not os.path.exists(dstSegPath):
                os.makedirs(dstSegPath)

            np.save(os.path.join(dstVolPath, cases[i]+'_'+f[:-4]+'_ori.npy'), volData_zs.astype(np.float32))
            np.save(os.path.join(dstSegPath, cases[i]+'_'+f[:-4]+'_ori.npy'), segData.astype(np.uint8))

            # if AUGMENTATION
            if augFlag:
                mode = ['mirror', 'rot90', 'rot-90', 'trans']
                v_mir, v_rot90, v_rot_90, v_trans1, v_trans2 = aug(volData_zs, mode)
                segData = np.expand_dims(segData, axis= 2)
                s_mir, s_rot90, s_rot_90, s_trans1, s_trans2 = aug(segData, mode)
                s_mir = np.squeeze(s_mir)
                s_rot90 = np.squeeze(s_rot90)
                s_rot_90 = np.squeeze(s_rot_90)
                s_trans1 = np.squeeze(s_trans1)
                s_trans2 = np.squeeze(s_trans2)
                # save aug volumes
                np.save(os.path.join(dstVolPath, cases[i]+'_'+f[:-4] + '_mir.npy'), v_mir.astype(np.float32))
                np.save(os.path.join(dstVolPath, cases[i]+'_'+f[:-4] + '_r90.npy'), v_rot90.astype(np.float32))
                np.save(os.path.join(dstVolPath, cases[i]+'_'+f[:-4] + '_rn90.npy'), v_rot_90.astype(np.float32))
                np.save(os.path.join(dstVolPath, cases[i]+'_'+f[:-4] + '_tran1.npy'), v_trans1.astype(np.float32))
                np.save(os.path.join(dstVolPath, cases[i]+'_'+f[:-4] + '_tran2.npy'), v_trans2.astype(np.float32))
                # save aug segs
                np.save(os.path.join(dstSegPath, cases[i]+'_'+f[:-4] + '_mir.npy'), s_mir.astype(np.uint8))
                np.save(os.path.join(dstSegPath, cases[i]+'_'+f[:-4] + '_r90.npy'), s_rot90.astype(np.uint8))
                np.save(os.path.join(dstSegPath, cases[i]+'_'+f[:-4] + '_rn90.npy'), s_rot_90.astype(np.uint8))
                np.save(os.path.join(dstSegPath, cases[i]+'_'+f[:-4] + '_tran1.npy'), s_trans1.astype(np.uint8))
                np.save(os.path.join(dstSegPath, cases[i]+'_'+f[:-4] + '_tran2.npy'), s_trans2.astype(np.uint8))


def prepareTest():
    for i in range(70, len(cases)):
        print(cases[i])
        volPath = os.path.join(dataPath, 'vol_case', cases[i])
        segPath = os.path.join(dataPath, 'seg_case', cases[i])
        vols = os.listdir(volPath)
        segs = os.listdir(segPath)
        for f in vols:
            vol = sio.loadmat(os.path.join(volPath, f))
            volData = vol['vol']
            volData_zs = zscore(volData)
            volData_zs = volData_zs[:, :, 0:5]
            seg = sio.loadmat(os.path.join(segPath, f))
            segData = seg['seg']
            dstVolPath = os.path.join(dataPath, '5c', 'test', 'vol')
            if not os.path.exists(dstVolPath):
                os.makedirs(dstVolPath)
            dstSegPath = os.path.join(dataPath, '5c', 'test', 'seg')
            if not os.path.exists(dstSegPath):
                os.makedirs(dstSegPath)

            np.save(os.path.join(dstVolPath, cases[i]+'_'+f[:-4]+'_ori.npy'), volData_zs.astype(np.float32))
            np.save(os.path.join(dstSegPath, cases[i]+'_'+f[:-4]+'_ori.npy'), segData.astype(np.uint8))

if __name__ == '__main__':
    augFlag = True
    print('prepare train......')
    prepareTrain(augFlag)
    print('prepare test......')
    prepareTest()

print('ok')
