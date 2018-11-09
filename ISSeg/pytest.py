
#------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------for sun-------------------------------------------------------------------------------------------------------
# from pyexcel_xlsx import get_data
# from pyexcel_xlsx import save_data
# from collections import OrderedDict
#
# THISYEAR = 2018
# THISMONTH = 8
# THISDAY = 15
# path = 'D:/中医院上半年整理.xlsx'
# dstpath = 'D:/中医院上半年整理-xiugai.xlsx'
#
#
# xls_data = get_data(path)
# sheet_1 = xls_data['Sheet1']  # type:list
#
# length = len(sheet_1)
# name_col = sheet_1[0].index('姓名')
# ID_col = sheet_1[0].index('身份证号')
# birth_col = sheet_1[0].index('出生日期')
# gender_col = sheet_1[0].index('性别')
# age_col = sheet_1[0].index('年龄')
#
# for i in range(1, length):
#     info = sheet_1[i]
#     ID = info[ID_col]
#     ID = ''.join(ID.split())
#     if ID != '' and len(ID) == 18:
#         birth_year = int(ID[6:10])
#         birth_mon = int(ID[10:12])
#         birth_day = int(ID[12:14])
#         geder_mark = int(ID[16])
#         age = THISYEAR - birth_year
#         if birth_mon < THISMONTH:
#             age = str(age)
#         elif birth_mon == THISMONTH and birth_day < THISDAY:
#             age =  str(age)
#         else:
#             age = str(age-1)
#         birth_date = str(birth_year) + '-' + str(birth_mon) + '-' + str(birth_day)
#         info[birth_col] = birth_date
#         info[age_col] = age
#         if (geder_mark % 2 == 0):
#             info[gender_col] = '女'
#         else:
#             info[gender_col] = '男'
#     else:
#         print("身份证号不正确:", (i+1), info[name_col])
# save_data(dstpath, xls_data)
#
# print('ok')

#------------------------------------------------------------------------------------------------------------------------------------------

import os
import numpy as np
import scipy.misc as smc
import scipy.io as sio
import tensorflow as tf

# ori_path = 'D:/CTPRecons_Rst/exp/npys/'
# dirs = os.listdir(ori_path)

# for i in range(6):
#     data = np.load(ori_path+dirs[i])
#
#     img = np.zeros((128,128,3))
#     cord = np.where(data == 1)
#     img[cord[0], cord[1], 0] = 64
#     img[cord[0], cord[1], 1] = 0
#     img[cord[0], cord[1], 2] = 128
#     smc.toimage(img, cmin= 0, cmax= 255).save(ori_path + dirs[i][:-4] + '.jpg')


#-----------------------------------------------------------------------------------------------------------------------------------
# import nibabel as nib
# import numpy as np
# from PIL import Image
#
# nii = nib.load('E:/ISSEG/ISLES2018_Training/TRAINING/case_5/SMIR.Brain.XX.O.OT.345594/SMIR.Brain.XX.O.OT.345594.nii')
# data = nii.get_data()
# seg = data[...,7]
# im = Image.fromarray(np.uint8(seg * 255))
# im.show()
# im.save('aa.png')
# seg = np.rot90(seg, 3)
# seg = np.flipud(seg)
# im = Image.fromarray(np.uint8(seg * 255))
# im.show()
# im.save('bb.png')
# print('ok')

# a = np.array([[1,2,3, 4],[5,6,7,8], [9,10,11,12], [13,14, 15,16]])
# np.transpose(a, (1,0))
# c = []
# d = np.zeros((4,4))
#
# for i in range(2):
#     for j in range(2):
#         b = a[i*2:(i+1)*2, j*2:(j+1)*2]
#         c.append(b)
# for i in range(4):
#     row = i // 2
#     col = i%2
#     d[row*2:(row+1)*2, col*2:(col+1)*2] = c[i]
#

# a=np.load('E:/trial/yaoxianpu_64_5-mask.npy')
# b=np.load('D:/CTPRecons_Rst/exp2/npys/yaoxianpu_64_0-pred.npy')
# c = np.load('E:/Cerebral Infarction/perfusion_3c/perfusion_npy/train/data/liyuan_-0.75_0.npy')
#
#
# dirs = os.listdir('E:/trial/')
# pred= np.zeros((512,512), dtype= np.float32)
# kBlockWidth = 128
# for i in range(1,len(dirs)):
#     name = dirs[i]
#     npy = np.load('E:/trial/' + name)
#     name_splits = name.split('_')
#     last_splits = name_splits[-1]
#     if 'mask' in last_splits:
#         order = last_splits[-9:len(last_splits)]
#         order = int(last_splits.rstrip(order))
#         row = order // 4
#         col = order % 4
#         pred[int(row * kBlockWidth): int((row + 1) * kBlockWidth),
#         int(col * kBlockWidth): int((col + 1) * kBlockWidth)] = npy
#
# sio.savemat('E:/trial/'+ name_splits[0] +'_'+ name_splits[1] + '_pred.mat', {'ctp':pred})
#
#

#--------read params from ckpt
from tensorflow.python.tools.inspect_checkpoint import  print_tensors_in_checkpoint_file
from tensorflow.python import pywrap_tensorflow
print_tensors_in_checkpoint_file('D://resnet_v2_50/resnet_v2_50.ckpt', tensor_name= None, all_tensors= False, all_tensor_names= True)
# print_tensors_in_checkpoint_file('D:/DLexp/IESLES_Rst/CT_128_VFMT/exp12/ckpt/modle-10500', tensor_name= None, all_tensors= False, all_tensor_names= True)






print('ok')