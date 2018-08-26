'''
preprocess ISLES2018 dataset
80 case as training
24 case as test
the old version
'''

import numpy as np
import nibabel as nib
import os
import cv2
import scipy.io as sio
import random

IMGWIDTH = 256
IMGHEIGHT = 256

datasetPath = 'E:/MRI Brain Seg/ISLES2018/TRAINING'
dstPath = 'E:/MRI Brain Seg/ISLES2018/process_set'

if not os.path.exists(dstPath):
    os.makedirs(dstPath)

cases = os.listdir(datasetPath)
case_num = len(cases)

#-----------------------respacing-----------------------------------------------
def respacing():
    ##########  cv.resize cannot resize 3d data

    for i in range(case_num):
        dir_case = os.path.join(datasetPath, cases[i])
        folders_case = os.listdir(dir_case)

        # extract dwi data
        dwi_name = [fold_name for fold_name in folders_case if ('DWI' in fold_name)]
        dwi_path = os.path.join(dir_case, dwi_name[0])
        files = os.listdir(dwi_path)
        dwi_nii = [f for f in files if ('.nii' in f)]
        vol = nib.load(os.path.join(dwi_path, dwi_nii[0]))
        vol_data = vol.get_data()
        pixdim = vol.header.get_zooms()
        # extract OT data
        ot_name = [fold_name for fold_name in folders_case if ('OT' in fold_name)]
        ot_path = os.path.join(dir_case, ot_name[0])
        files = os.listdir(ot_path)
        ot_nii = [f for f in files if ('.nii' in f)]
        seg = nib.load(os.path.join(ot_path, ot_nii[0]))
        seg_data = seg.get_data()
        pixdim2 = seg.header.get_zooms()

        oriShape = vol_data.shape
        dstDepth = pixdim[2] * oriShape[2]
        assert oriShape[0] == IMGWIDTH

        #dst_vol = cv2.resize(vol_data, (IMGWIDTH, IMGHEIGHT, dstDepth))
        #dst_seg = cv2.resize(seg_data, (IMGWIDTH, IMGHEIGHT, dstDepth))

        dstVolPath = os.path.join(dstPath, 'mat', 'vol')
        dstSegPath = os.path.join(dstPath, 'mat', 'seg')
        dstVolPath = os.path.join(dstVolPath, '%d.mat' % i)
        dstSegPath = os.path.join(dstSegPath, '%d.mat' % i)
        sio.savemat(dstVolPath, {'vol': vol_data})
        sio.savemat(dstSegPath, {'seg': seg_data})



#------------------------for training set----------------------------------------
def zscore(x):
    x_mean = np.mean(x)
    x_std = np.std(x)
    x_std = x_std + 1e-5
    x_zs = (x - x_mean) / (x_std)
    #x_zs = x -x_mean
    return x_zs

def gen_train_2d(volmats):
    vol_num = len(volmats)
    for i in range(80):
        pre = volmats[i]
        pre = pre[:-4]
        segname = os.path.join(dstPath, 'mat/seg', volmats[i])
        vol = sio.loadmat(os.path.join(volmat_path, volmats[i]))
        vol_data = vol['vol']   # 256x265xn, float32
        vol_zs = zscore(vol_data)
        seg = sio.loadmat(segname)
        seg_data = seg['seg']   # 265x265xn, uint8

        for j in range(vol_zs.shape[2]):
            vol_np_path = os.path.join(dstPath, 'npy2d/train/vol')
            if not os.path.exists(vol_np_path):
                os.mkdir(vol_np_path)
            seg_np_path = os.path.join(dstPath, 'npy2d/train/seg')
            if not os.path.exists(seg_np_path):
                os.mkdir(seg_np_path)

            np.save(os.path.join(vol_np_path, pre + '_' + str(j) + '.npy'), vol_zs[..., j].astype(np.float32))
            np.save(os.path.join(seg_np_path, pre + '_' + str(j) + '.npy'), seg_data[..., j].astype(np.uint8))


def gen_test_2d(volmats):
    vol_num = len(volmats)
    for i in range(80, vol_num):
        pre = volmats[i]
        pre = pre[:-4]
        segname = os.path.join(dstPath, 'mat/seg', volmats[i])
        vol = sio.loadmat(os.path.join(volmat_path, volmats[i]))
        vol_data = vol['vol']  # 256x265xn, float32
        vol_zs = zscore(vol_data)
        seg = sio.loadmat(segname)
        seg_data = seg['seg']  # 265x265xn, uint8

        for j in range(vol_zs.shape[2]):
            vol_np_path = os.path.join(dstPath, 'npy2d/test/vol')
            if not os.path.exists(vol_np_path):
                os.mkdir(vol_np_path)
            seg_np_path = os.path.join(dstPath, 'npy2d/test/seg')
            if not os.path.exists(seg_np_path):
                os.mkdir(seg_np_path)

            np.save(os.path.join(vol_np_path, pre + '_' + str(j) + '.npy'), vol_zs[..., j].astype(np.float32))
            np.save(os.path.join(seg_np_path, pre + '_' + str(j) + '.npy'), seg_data[..., j].astype(np.uint8))


if __name__ == '__main__':
    #respacing()
    volmat_path = os.path.join(dstPath, 'mat/vol')
    volmats = os.listdir(volmat_path)
    random.shuffle(volmats)
    gen_train_2d(volmats)
    gen_test_2d(volmats)

print('ok')



