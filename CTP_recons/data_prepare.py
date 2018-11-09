'''
read from perfusion_recons folder to gen npys for DL reconstruction
LionZhu
20180915
'''

import os
import numpy as np
import scipy.io as sio

IMGWidth = 512
IMGHeight = 512
ori_path = 'E:/Cerebral Infarction/perfusion_30c/perfusion_recons/'
dst_path = 'E:/Cerebral Infarction/perfusion_30c/perfusion_npy/'
datas = os.listdir(ori_path + 'dat')

def zscore(x):
    x_mean = np.mean(x)
    x_std = np.std(x)
    x_zs = (x-x_mean) / (x_std + 1e-5)
    return x_zs

def PrepareTrain(aug_flag):
    for ind in range(18*4):
        print(ind)
        name = datas[ind]
        name_pre = name[:-4]
        data_mat = sio.loadmat(ori_path+ 'dat/' +name)
        dat = data_mat['dat']
        dat_zs = zscore(dat)
        cbf_mat = sio.loadmat(ori_path + 'lab/' + 'cbf/' +name)
        cbf = cbf_mat['lab']
        cbv_mat = sio.loadmat(ori_path + 'lab/' + 'cbv/' + name)
        cbv = cbv_mat['lab']
        mtt_mat = sio.loadmat(ori_path + 'lab/' + 'mtt/' + name)
        mtt = mtt_mat['lab']
        ttp_mat = sio.loadmat(ori_path + 'lab/' + 'ttp/' + name)
        ttp = ttp_mat['lab']
        tmax_mat = sio.loadmat(ori_path + 'lab/' + 'tmax/' + name)
        tmax = tmax_mat['lab']
        if aug_flag:
            data_save = dat_zs
            cbf_save = cbf
            cbv_save = cbv
            mtt_save = mtt
            ttp_save = ttp
            tmax_save = tmax
            np.save(dst_path + 'train/' + 'data/' + name_pre + '_0' + '.npy',
                    data_save.astype(np.float32))
            np.save(dst_path + 'train/' + 'label/' + 'cbf/' + name_pre + '_0' + '.npy',
                    cbf_save.astype(np.float32))
            np.save(dst_path + 'train/' + 'label/' + 'cbv/' + name_pre + '_0' + '.npy',
                    cbv_save.astype(np.float32))
            np.save(dst_path + 'train/' + 'label/' + 'mtt/' + name_pre + '_0' + '.npy',
                    mtt_save.astype(np.float32))
            np.save(dst_path + 'train/' + 'label/' + 'ttp/' + name_pre + '_0' + '.npy',
                    ttp_save.astype(np.float32))
            np.save(dst_path + 'train/' + 'label/' + 'tmax/' + name_pre + '_0' + '.npy',
                    tmax_save.astype(np.float32))
        else:
            for i in range(4):
                for j in range(4):
                    data_save = dat_zs[:, int((IMGWidth/4)*i) : int((IMGWidth/4)*(i+1)), int((IMGWidth/4)*j) : int((IMGWidth/4)*(j+1))]
                    # data_save shape[3,512,512]
                    cbf_save = cbf[int((IMGWidth/4)*i) : int((IMGWidth/4)*(i+1)), int((IMGWidth/4)*j) : int((IMGWidth/4)*(j+1))]
                    cbv_save = cbv[int((IMGWidth/4)*i) : int((IMGWidth/4)*(i+1)), int((IMGWidth/4)*j) : int((IMGWidth/4)*(j+1))]
                    mtt_save = mtt[int((IMGWidth/4)*i) : int((IMGWidth/4)*(i+1)), int((IMGWidth/4)*j) : int((IMGWidth/4)*(j+1))]
                    ttp_save = ttp[int((IMGWidth/4)*i) : int((IMGWidth/4)*(i+1)), int((IMGWidth/4)*j) : int((IMGWidth/4)*(j+1))]
                    tmax_save = tmax[int((IMGWidth/4)*i) : int((IMGWidth/4)*(i+1)), int((IMGWidth/4)*j) : int((IMGWidth/4)*(j+1))]

                    np.save(dst_path+'train/'+'data/'+ name_pre +'_' +str(i*4+j) +'.npy', data_save.astype(np.float32))
                    np.save(dst_path+'train/'+'label/'+'cbf/' +name_pre +'_' +str(i*4+j) +'.npy', cbf_save.astype(np.float32))
                    np.save(dst_path+'train/'+'label/'+'cbv/' +name_pre +'_' +str(i*4+j) +'.npy', cbv_save.astype(np.float32))
                    np.save(dst_path+'train/'+'label/'+'mtt/' +name_pre +'_' +str(i*4+j) +'.npy', mtt_save.astype(np.float32))
                    np.save(dst_path+'train/'+'label/'+'ttp/' +name_pre +'_' +str(i*4+j) +'.npy', ttp_save.astype(np.float32))
                    np.save(dst_path+'train/'+'label/'+'tmax/' +name_pre +'_' +str(i*4+j) +'.npy', tmax_save.astype(np.float32))

def PrepareTest(aug_flag):
    for ind in range((18*4) , (24*4)):
        print(ind)
        name = datas[ind]
        name_pre = name[:-4]
        data_mat = sio.loadmat(ori_path+ 'dat/' +name)
        dat = data_mat['dat']
        dat_zs = zscore(dat)
        cbf_mat = sio.loadmat(ori_path + 'lab/' + 'cbf/' +name)
        cbf = cbf_mat['lab']
        cbv_mat = sio.loadmat(ori_path + 'lab/' + 'cbv/' + name)
        cbv = cbv_mat['lab']
        mtt_mat = sio.loadmat(ori_path + 'lab/' + 'mtt/' + name)
        mtt = mtt_mat['lab']
        ttp_mat = sio.loadmat(ori_path + 'lab/' + 'ttp/' + name)
        ttp = ttp_mat['lab']
        tmax_mat = sio.loadmat(ori_path + 'lab/' + 'tmax/' + name)
        tmax = tmax_mat['lab']
        if aug_flag:
            data_save = dat_zs
            cbf_save = cbf
            cbv_save = cbv
            mtt_save = mtt
            ttp_save = ttp
            tmax_save = tmax
            np.save(dst_path + 'test/' + 'data/' + name_pre + '_0' + '.npy',
                    data_save.astype(np.float32))
            np.save(dst_path + 'test/' + 'label/' + 'cbf/' + name_pre + '_0' + '.npy',
                    cbf_save.astype(np.float32))
            np.save(dst_path + 'test/' + 'label/' + 'cbv/' + name_pre + '_0' + '.npy',
                    cbv_save.astype(np.float32))
            np.save(dst_path + 'test/' + 'label/' + 'mtt/' + name_pre + '_0' + '.npy',
                    mtt_save.astype(np.float32))
            np.save(dst_path + 'test/' + 'label/' + 'ttp/' + name_pre + '_0' + '.npy',
                    ttp_save.astype(np.float32))
            np.save(dst_path + 'test/' + 'label/' + 'tmax/' + name_pre + '_0' + '.npy',
                    tmax_save.astype(np.float32))
        else:
            for i in range(4):
                for j in range(4):
                    data_save = dat_zs[:, int((IMGWidth/4)*i) : int((IMGWidth/4)*(i+1)), int((IMGWidth/4)*j) : int((IMGWidth/4)*(j+1))]
                    cbf_save = cbf[int((IMGWidth/4)*i) : int((IMGWidth/4)*(i+1)), int((IMGWidth/4)*j) : int((IMGWidth/4)*(j+1))]
                    cbv_save = cbv[int((IMGWidth/4)*i) : int((IMGWidth/4)*(i+1)), int((IMGWidth/4)*j) : int((IMGWidth/4)*(j+1))]
                    mtt_save = mtt[int((IMGWidth/4)*i) : int((IMGWidth/4)*(i+1)), int((IMGWidth/4)*j) : int((IMGWidth/4)*(j+1))]
                    ttp_save = ttp[int((IMGWidth/4)*i) : int((IMGWidth/4)*(i+1)), int((IMGWidth/4)*j) : int((IMGWidth/4)*(j+1))]
                    tmax_save = tmax[int((IMGWidth/4)*i) : int((IMGWidth/4)*(i+1)), int((IMGWidth/4)*j) : int((IMGWidth/4)*(j+1))]

                    np.save(dst_path+'test/'+'data/'+ name_pre +'_' +str(i*4+j) +'.npy', data_save.astype(np.float32))
                    np.save(dst_path+'test/'+'label/'+'cbf/' +name_pre +'_' +str(i*4+j) +'.npy', cbf_save.astype(np.float32))
                    np.save(dst_path+'test/'+'label/'+'cbv/' +name_pre +'_' +str(i*4+j) +'.npy', cbv_save.astype(np.float32))
                    np.save(dst_path+'test/'+'label/'+'mtt/' +name_pre +'_' +str(i*4+j) +'.npy', mtt_save.astype(np.float32))
                    np.save(dst_path+'test/'+'label/'+'ttp/' +name_pre +'_' +str(i*4+j) +'.npy', ttp_save.astype(np.float32))
                    np.save(dst_path+'test/'+'label/'+'tmax/' +name_pre +'_' +str(i*4+j) +'.npy', tmax_save.astype(np.float32))

if __name__ == '__main__':
    aug_flag = False
    # True is no cropping
    print('prepare training set...')
    PrepareTrain(aug_flag)
    print('prepare testing set...')
    PrepareTest(aug_flag)
    print('ok')