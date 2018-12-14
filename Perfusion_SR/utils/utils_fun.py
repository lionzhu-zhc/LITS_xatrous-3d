#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2018/10/13 16:34
# @Author  : ***

import numpy as np
import os
import random

def get_data_train_2d(trainPath, batchsize):
    vol_batch = []
    seg_batch = []
    for i in range(1, batchsize + 1):
        if i == 1:
            vol_batch, seg_batch = get_batch_train_2d(trainPath)
        else:
            vol_batch_tmp, seg_batch_tmp = get_batch_train_2d(trainPath)
            vol_batch = np.concatenate((vol_batch, vol_batch_tmp), axis=0)
            seg_batch = np.concatenate((seg_batch, seg_batch_tmp), axis=0)
    return vol_batch, seg_batch    #shape [BS,depth, height, width]

def get_batch_train_2d(trainPath):
    dirs_train = os.listdir(trainPath + 'vol/')
    samples = random.choice(dirs_train)
    #print(samples)
    vol_batch = np.load(trainPath + 'vol/' + samples)
    seg_batch = np.load(trainPath + 'label/' + samples)
    # vol_batch = np.transpose(vol_batch, [1,2,0])
    # seg_batch = np.transpose(seg_batch, [1,2,0])
    vol_batch = np.expand_dims(vol_batch, axis=0)
    seg_batch = np.expand_dims(seg_batch, axis=0)
    vol_batch.astype(np.float32)
    seg_batch.astype(np.float32)
    return vol_batch, seg_batch

def get_data_test_2d(testPath, tDir, batchsize= 1):
    vol_batch = []
    seg_batch = []
    for i in range(1, batchsize+1):
        if i == 1:
            vol_batch, seg_batch = get_batch_test_2d(testPath, tDir)
        else:
            vol_batch_tmp, seg_batch_tmp = get_batch_test_2d(testPath, tDir)
            vol_batch = np.concatenate((vol_batch, vol_batch_tmp), axis = 0)
            seg_batch = np.concatenate((seg_batch, seg_batch_tmp), axis = 0)
    return vol_batch, seg_batch  # vol_shape [BDWH]

def get_batch_test_2d(testPath, tDir):
    vol_batch = np.load(testPath + 'vol/' + tDir)
    seg_batch = np.load(testPath + 'label/' + tDir)
    # vol_batch = np.transpose(vol_batch, [1, 2, 0])
    # seg_batch = np.transpose(seg_batch, [1, 2, 0])
    vol_batch = np.expand_dims(vol_batch, axis = 0)
    seg_batch = np.expand_dims(seg_batch, axis = 0)
    vol_batch.astype(np.float32)
    seg_batch.astype(np.float32)
    return vol_batch, seg_batch


def SaveNpys(resultPath, name_pre, label_batch, pred_batch):
    np.save(resultPath + 'npys/' + name_pre + '-mask.npy', label_batch)
    np.save(resultPath + 'npys/' + name_pre + '-pred.npy', pred_batch)