
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

ori_path = 'E:/ISSEG/Dataset/2018REGROUP/128/1/'
dirs = os.listdir(ori_path)

for i in range(6):
    data = np.load(ori_path+dirs[i])

    img = np.zeros((128,128,3))
    cord = np.where(data == 1)
    img[cord[0], cord[1], 0] = 64
    img[cord[0], cord[1], 1] = 0
    img[cord[0], cord[1], 2] = 128
    smc.toimage(img, cmin= 0, cmax= 255).save(ori_path + dirs[i][:-4] + '.jpg')







