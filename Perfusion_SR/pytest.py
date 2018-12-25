#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2018/10/13 14:29
# @Author  : ***

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

from tensorflow.python.platform import gfile
import tensorflow as tf
import os
from tensorflow.python.framework import graph_util
#
# path = 'D:/12.pb'
#
# def save_mode_pb(pb_file_path):
#     x = tf.placeholder(tf.int32, name='x')
#     y = tf.placeholder(tf.int32, name='y')
#     b = tf.Variable(1, name='b')
#     xy = tf.multiply(x, y)
#
#     op = tf.add(xy, b, name='op_to_store')
#
#     sess = tf.Session()
#     sess.run(tf.global_variables_initializer())
#
#     path = os.path.dirname(os.path.abspath(pb_file_path))
#     if os.path.isdir(path) is False:
#         os.makedirs(path)
#
#     constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['op_to_store'])
#     with tf.gfile.FastGFile(pb_file_path, mode='wb') as f:
#         f.write(constant_graph.SerializeToString())
#
#     # test
#     feed_dict = {x: 2, y: 3}
#     print(sess.run(op, feed_dict))
#
# def restore_mode_pb(pb_file_path):
#     sess = tf.Session()
#     with gfile.FastGFile(pb_file_path, 'rb') as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read())
#         sess.graph.as_default()
#         tf.import_graph_def(graph_def, name='')
#
#     print(sess.run('b:0'))
#
#     input_x = sess.graph.get_tensor_by_name('x:0')
#     input_y = sess.graph.get_tensor_by_name('y:0')
#
#     op = sess.graph.get_tensor_by_name('op_to_store:0')
#
#     ret = sess.run(op, {input_x: 5, input_y: 5})
#     print(ret)
#
#
#
# print_tensors_in_checkpoint_file('D://resnet_v2_50/resnet_v2_50.ckpt', tensor_name= None, all_tensors= False, all_tensor_names= True)
#
# if __name__ == '__main__':
#     #save_mode_pb(path)
#     restore_mode_pb(path)

# a = np.squeeze(np.load('D:/DLexp/SuperResolution_Rst/exp4/npys/zhaokuaile_37.25-pred.npy')) * 200
# b = np.squeeze(np.load('D:/DLexp/SuperResolution_Rst/exp2/npys/zhaokuaile_37.25-mask.npy')) * 200
# c = np.squeeze(a - b)
# min = np.min(c)
# max = np.max(c)
# line1 = a[:, 161, 281] - 1024
# # plt.plot(line1)
# line2 = b[:, 161, 281] -1024
# plt.plot(line2)
#
# mat = sio.loadmat('E:/Cerebral Infarction/SuperResolution/SR_15_30/zhaokuaile/37.25.mat')
# matt = mat['Mat']
# line3 = matt[:, 161, 281] - 1024
# plt.plot(line3)
# plt.show()

# a = np.squeeze(np.load('E:\Cerebral Infarction\SuperResolution\exp_data3/test/label/zhangyongjun_44.75.npy'))
# b = a[10,:,:]
# b =b*1000
# b = (b-1024).astype(np.int16)
# plt.imshow(b,  cmap ='gray')
# plt.show()
# print('ok')

patient = 'yaoxiaofa/41.mat'
ori_path = 'D:\DLexp\SuperResolution_Rst\exp14\mats/' + patient
sr_path = 'E:\Cerebral Infarction\SuperResolution\SR_15_30/' +  patient
ref_path = 'D:\DLexp\SuperResolution_Rst\exp14\mats_ref/' + patient

mat = sio.loadmat(ori_path)
target = mat['Mat'] /1000

mat = sio.loadmat(sr_path)
img = mat['Mat'] /1000

mat = sio.loadmat(ref_path)
annot = mat['Mat'] / 1000

residual = img - annot
pred_res = target - img

dif = residual - pred_res
square = np.square(dif)
loss = np.sqrt(np.mean(square))

print('ok')