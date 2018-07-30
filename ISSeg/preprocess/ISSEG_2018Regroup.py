'''
regroup 2018 data
to save each slice as s single file
file strcuction: [NCCT; CBF; CBV; MTT; Tmax], channel  = 5
The OT save as seg labels. 2-D
LionZhu-list 20180725
'''

import numpy as np
import nibabel as nib
import os
import cv2
import scipy.io as sio
import random

oriDataset = 'E:/ISSEG/ISLES2018_Training/TRAINING'
dstDataset = 'E:/ISSEG/Dataset/2018REGROUP/all'

cases = os.listdir(oriDataset)
caseNum = len(cases)

for i in range(caseNum):
    casePath = os.path.join(oriDataset, cases[i])
    caseFolders = os.listdir(casePath)
    for folderName in caseFolders:
        lastFolder = os.listdir(os.path.join(casePath, folderName))
        niiName = [nii for nii in lastFolder if ('.nii' in nii)]
        if 'CT' in folderName:
            Vol = nib.load(os.path.join(casePath, folderName, niiName[0]))
            ctData = Vol.get_data()   # int32
        if 'CBV' in folderName:
            Vol = nib.load(os.path.join(casePath, folderName, niiName[0]))
            cbvData = Vol.get_data()  # int32
        if 'CBF' in folderName:
            Vol = nib.load(os.path.join(casePath, folderName, niiName[0]))
            cbfData = Vol.get_data()  # int32
        if 'MTT' in folderName:
            Vol = nib.load(os.path.join(casePath, folderName, niiName[0]))
            mttData = Vol.get_data()  # float64
        if 'Tmax' in folderName:
            Vol = nib.load(os.path.join(casePath, folderName, niiName[0]))
            tmaxData = Vol.get_data() # float64
        if 'OT' in folderName:
            Vol = nib.load(os.path.join(casePath, folderName, niiName[0]))
            otData = Vol.get_data()  # uint8

    ctShape = ctData.shape
    cbvShape = cbvData.shape
    cbfShape = cbfData.shape
    mttShape = mttData.shape
    tmaxShape = tmaxData.shape
    otShape = otData.shape

    assert ctShape == cbfShape == cbvShape == mttShape == tmaxShape == otShape

    for j in range(ctShape[2]):
        vol = np.zeros((ctShape[0], ctShape[1], 5))   # float64
        vol[..., 0] = ctData[..., j]
        vol[..., 1] = cbvData[..., j]
        vol[..., 2] = cbfData[..., j]
        vol[..., 3] = mttData[..., j]
        vol[..., 4] = tmaxData[..., j]
        otMap = otData[..., j]
        seg = otMap
        dstVolPath = os.path.join(dstDataset, 'vol', cases[i])
        dstSegPath = os.path.join(dstDataset, 'seg', cases[i])
        if not os.path.exists(dstVolPath):
            os.mkdir(dstVolPath)
        if not os.path.exists(dstSegPath):
            os.mkdir(dstSegPath)

        dstVolPath = os.path.join(dstVolPath, ('%d.mat')%j)
        dstSegPath = os.path.join(dstSegPath, ('%d.mat')%j)
        sio.savemat(dstVolPath, {'vol': vol})
        sio.savemat(dstSegPath, {'seg': seg})



print('ok')

