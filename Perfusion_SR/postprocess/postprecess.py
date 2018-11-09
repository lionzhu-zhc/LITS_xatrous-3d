#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2018/10/16 14:51
# @Author  : ***

# read the pred npys and the corresponding PRE POST AIFxy to save as mats for ctp recons

import os
import numpy as np
import scipy.io as sio

path = 'D:/DLexp/SuperResolution_Rst/exp4/'
ori_path = path + 'npys/'
dst_path = path + 'mats/'
ref_path = 'E:/Cerebral Infarction/SuperResolution/perfusion_mat/'

npys = os.listdir(ori_path)
npy_num = len(npys)

for i in range(0, npy_num, 2):
    pred = np.squeeze(np.load(ori_path + npys[i+1]))
    pred = np.transpose(pred, [2,0,1])
    splits = npys[i+1].split('_')
    patient = splits[0]
    loc = splits[1][:-9]

    ori_mat = sio.loadmat(ref_path + patient + '/' +loc + '.mat')
    AIFx = ori_mat['AIFx'].flatten()
    AIFy = ori_mat['AIFy'].flatten()
    PRE = ori_mat['PRE'].flatten()
    POST = ori_mat['POST'].flatten()
    VOFx = ori_mat['VOFx'].flatten()
    VOFy = ori_mat['VOFy'].flatten()

    if not os.path.exists(dst_path+patient):
        os.makedirs(dst_path+patient)

    sio.savemat(dst_path+patient+'/' +loc + '.mat',
                {'Mat':pred, 'AIFx': AIFx, 'AIFy':AIFy, 'VOFx':VOFx, 'VOFy':VOFy, 'PRE':PRE, 'POST':POST})

print('ok')

