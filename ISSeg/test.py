'''
read the mats from All folder
and prepare the training and test set
Lionzhu-list 20180726
'''

import os
import numpy as np
import random
import scipy.io as sio
from PIL import Image
import cv2
import nibabel as nib

dataPath = 'E:/ISSEG/Dataset/2018REGROUP/all'
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

def aug(data, mode):
    shap = data.shape
    dx = dy = 30
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
            T_1 = np.float32([[1,0,dx], [0,1,dy]])
            trans_2 = np.zeros_like(data)
            T_2 = np.float32([[1, 0, -dx], [0, 1, -dy]])
            for i in range(shap[2]):
                trans_1[..., i] = cv2.warpAffine(data[..., i], T_1, (shap[0], shap[0]))
                trans_2[..., i] = cv2.warpAffine(data[..., i], T_2, (shap[0], shap[0]))
    return  mir, rot90, rot_90, trans_1, trans_2


def prepareTrain(augFlag):
    for i in range(3):
        print(cases[i])
        volPath = os.path.join(dataPath, 'vol_case', cases[i])
        segPath = os.path.join(dataPath, 'seg_case', cases[i])
        vols = os.listdir(volPath)
        segs = os.listdir(segPath)
        for f in vols:
            vol = sio.loadmat(os.path.join(volPath, f))
            volData = vol['vol']
            volData_zs = (volData)
            seg = sio.loadmat(os.path.join(segPath, f))
            segData = seg['seg']
            dstVolPath = os.path.join(dataPath, 'train', 'vol')
            if not os.path.exists(dstVolPath):
                os.mkdir(dstVolPath)
            dstSegPath = os.path.join(dataPath, 'train', 'seg')
            if not os.path.exists(dstSegPath):
                os.mkdir(dstSegPath)


            img = nib.Nifti1Image(volData_zs, np.eye(4))
            segg = nib.Nifti1Image(segData, np.eye(4))
            nib.save(img, os.path.join(dstVolPath, cases[i]+'_'+f[:-4]+'_ori.nii'))
            nib.save(segg, os.path.join(dstSegPath, cases[i] + '_' + f[:-4] + '_ori.nii'))

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

                img_mir = nib.Nifti1Image(v_mir, np.eye(4) )
                seg_mir = nib.Nifti1Image(s_mir, np.eye(4))
                img_trans = nib.Nifti1Image(v_trans1, np.eye(4))
                seg_trans = nib.Nifti1Image(s_trans1, np.eye(4))
                nib.save(img_mir, os.path.join(dstVolPath, cases[i] + '_' + f[:-4] + '_mir.nii'))
                nib.save(seg_mir, os.path.join(dstSegPath, cases[i] + '_' + f[:-4] + '_mir.nii'))
                nib.save(img_trans, os.path.join(dstVolPath, cases[i] + '_' + f[:-4] + '_trans.nii'))
                nib.save(seg_trans, os.path.join(dstSegPath, cases[i] + '_' + f[:-4] + '_trans.nii'))




if __name__ == '__main__':
    augFlag = True
    print('prepare train......')
    prepareTrain(augFlag)


print('ok')
