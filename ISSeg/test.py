
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


import tensorflow as tf

x1 = tf.constant(1.0, shape=[1, 3, 3, 1])

x2 = tf.constant(1.0, shape=[1, 6, 6, 3])

x3 = tf.constant(1.0, shape=[1, 5, 5, 3])

kernel = tf.constant(1.0, shape=[3, 3, 3, 1])

y1 = tf.nn.conv2d_transpose(x1, kernel, output_shape=[1, 6, 6, 3],
                            strides=[1, 2, 2, 1], padding="SAME")

y2 = tf.nn.conv2d(x3, kernel, strides=[1, 2, 2, 1], padding="SAME")

y3 = tf.nn.conv2d_transpose(y2, kernel, output_shape=[1, 5, 5, 3],
                            strides=[1, 2, 2, 1], padding="SAME")

y4 = tf.nn.conv2d(x2, kernel, strides=[1, 2, 2, 1], padding="SAME")

'''
Wrong!!This is impossible
y5 = tf.nn.conv2d_transpose(x1,kernel,output_shape=[1,10,10,3],strides=[1,2,2,1],padding="SAME")
'''
sess = tf.Session()
tf.global_variables_initializer().run(session=sess)
x1_decov, x3_cov, y2_decov, x2_cov = sess.run([y1, y2, y3, y4])
print(x1_decov.shape)
print(x3_cov.shape)
print(y2_decov.shape)
print(x2_cov.shape)





