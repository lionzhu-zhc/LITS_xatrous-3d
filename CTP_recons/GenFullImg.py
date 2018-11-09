'''
restore the ori 512 image from 128 block
0918 Lionzhu
'''

import os
import numpy as np
import scipy.io as sio

kBlockWidth = 128
path = 'D:/CTPRecons_Rst/exp4/'
ori_path = path + 'npys/'
dst_path = path +'mats/cbf/'

if not os.path.exists(dst_path):
    os.makedirs(dst_path)

dirs = os.listdir(ori_path)

for i in range(24):
    pred = np.zeros((512,512), dtype= np.float32)
    mask = np.zeros((512,512), dtype= np.float32)

    # to save 512 pred results mat
    # for j in range(2):
    #     name = dirs[i * 2 + j]
    #     npy = np.load(ori_path + name)
    #     name_splits = name.split('_')
    #     last_splits = name_splits[-1]
    #     if 'pred' in last_splits:
    #         pred = npy
    #     if 'mask' in last_splits:
    #         mask = npy
    # #-------------------

    for j in range (32):
        name = dirs[i*32 + j]
        npy = np.load(ori_path + name)
        name_splits = name.split('_')
        last_splits = name_splits[-1]
        if 'pred' in last_splits:
            order = last_splits[-9:len(last_splits)]
            order = int(last_splits.rstrip(order))

            row = order // 4
            col = order % 4
            pred[int(row*kBlockWidth): int((row+1)*kBlockWidth), int(col*kBlockWidth): int((col+1)*kBlockWidth)] = npy
        if 'mask' in last_splits:
            order = last_splits[-9:len(last_splits)]
            order = int(last_splits.rstrip(order))

            row = order // 4
            col = order % 4
            mask[int(row * kBlockWidth): int((row + 1) * kBlockWidth), int(col * kBlockWidth): int((col + 1) * kBlockWidth)] = npy

    sio.savemat(dst_path+ name_splits[0] +'_'+ name_splits[1] + '_pred.mat', {'ctp':pred})
    sio.savemat(dst_path + name_splits[0] +'_'+ name_splits[1] + '_mask.mat', {'ctp': mask})
    print(dst_path + name_splits[0] +'_'+ name_splits[1])



print('ok')