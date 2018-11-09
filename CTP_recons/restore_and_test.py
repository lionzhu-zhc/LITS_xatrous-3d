'''
restore from ckpt foler and test
0920 2018 LionZhu
'''

import resNetPro
import tensorflow as tf
import os
import math
import utils
import numpy as np

path = 'E:/Cerebral Infarction/perfusion_npy/'
testPath = path + 'test/'
resultPath = 'D://CTPRecons_Rst/exp2/'

BS = 12

data = tf.placeholder(tf.float32, [None,None,None,3])
ctp = tf.placeholder(tf.float32, [None,None,None,1])
pred = resNetPro.resnet(data, False)

sess = tf.Session()
model_file = tf.train.latest_checkpoint(resultPath + 'ckpt/')
saver = tf.train.Saver()
saver.restore(sess, model_file)

test_dirs = os.listdir(testPath + '/data/')
test_num = len(test_dirs)
test_times = math.ceil(test_num / BS)
for i in range(int(test_times)):
    if i != (test_times - 1):
        tDir = test_dirs[i * BS: (i + 1) * BS]
        img_batch, ctp_batch = utils.GetTestData_2d(testPath, tDir, BS)
    if i == (test_times - 1):
        tDir = test_dirs[(test_num - BS): test_num]
        img_batch, ctp_batch = utils.GetTestData_2d(testPath, tDir, BS)

    test_feed = {data: img_batch}
    test_pred_ctp = sess.run(pred, feed_dict= test_feed)

    for j in range(BS):
        label_batch = np.squeeze(ctp_batch[j, ...])
        pred_batch = np.squeeze(test_pred_ctp[j, ...])
        label_tosave = label_batch
        pred_tosave = pred_batch

        namePre = tDir[j]
        namePre = namePre[:-4]
        print("test_itr:", namePre)
        utils.SaveNpys(resultPath, namePre, label_tosave, pred_tosave)