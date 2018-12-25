#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2018/10/13 15:53
# @Author  : LionZHu

# train a SRCNN and test for the restore of perfusion Mat

import tensorflow as tf
from net import SRCNN, LossPy, ResSR, sparse_CNN
from utils import utils_fun
import datetime
import os
import numpy as np
slim = tf.contrib.slim

path = 'E:/Cerebral Infarction/SuperResolution/exp_data2/'
# trainPath = path + '1/'
trainPath = path + 'train_noaug/'
testPath = path + 'test/'
#change dir here ..............................................................
resultPath = 'D:/DLexp/SuperResolution_Rst/exp16/'
pretrain_path = 'D://resnet_v2_50/resnet_v2_50.ckpt'

IMAGE_CHANNEL = 30

LEARNING_RATE = 1e-3
EPOCH = 4
ITER_PER_EPOCH = 2000
DECAY_INTERVAL = ITER_PER_EPOCH * EPOCH // 10
MAX_ITERATION = ITER_PER_EPOCH * EPOCH
SAVE_CKPT_INTERVAL = ITER_PER_EPOCH * EPOCH // 2
TRAIN_BATCHSIZE = 32

def training(lr, loss_val, va_list):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    optimizer = tf.train.AdamOptimizer(lr)
    with tf.control_dependencies(update_ops):
        grads = optimizer.compute_gradients(loss_val, var_list=va_list)
        return optimizer.apply_gradients(grads)

def srcnn_run():
    with tf.name_scope('inputs'):
        image = tf.placeholder(tf.float32, shape=[None, None, None, None, 1], name='image')           # shape BDHWC
        annotation = tf.placeholder(tf.float32, shape=[None, None, None, None, 1], name='annotation')  # shape BDHWC
    bn_flag = tf.placeholder(tf.bool)
    keep_prob = tf.placeholder(tf.float32)
    residual = tf.subtract(image, annotation)

    # pred = SRCNN.build_srcnn(tensor_in= image, BN_FLAG= bn_flag, KEEPPROB= keep_prob, IMGCHANNEL= IMAGE_CHANNEL)
    # pred = ResSR.ResSR(inputs= image, is_training= bn_flag, IMGCHANNEL= IMAGE_CHANNEL)
    pred_res = sparse_CNN.build_sparseCNN(tensor_in= image, BN_FLAG= bn_flag,  KEEPPROB= keep_prob, IMGCHANNEL= IMAGE_CHANNEL)
    target = tf.subtract(image, pred_res)


    with tf.variable_scope('loss'):
        # loss_reduce = tf.losses.mean_squared_error(annotation, target)
        dif = tf.subtract(residual, pred_res)
        square = tf.square(dif)
        loss_reduce = tf.reduce_mean(square)

        tf.summary.scalar('loss', loss_reduce)

    with tf.variable_scope('trainOP'):
        LRate = tf.placeholder(tf.float32)
        trainable_vars = tf.trainable_variables()
        train_op = training(LRate, loss_reduce, trainable_vars)
        tf.summary.scalar('lr', LRate)

    with tf.variable_scope('fcnx') as scope:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        #----get the vars to restore-------------------------------
        # checkpoint_exclude_scopes = 'ResSR, resnet_v2_50/conv1'
        # exclusions = None
        # if checkpoint_exclude_scopes:
        #     exclusions = [
        #         scope.strip() for scope in checkpoint_exclude_scopes.split(',')]
        # variables_to_restore = []
        # for var in slim.get_model_variables():
        #     excluded = False
        #     for exclusion in exclusions:
        #         if var.op.name.startswith(exclusion):
        #             excluded = True
        #     if not excluded:
        #         variables_to_restore.append(var)
        #-----------------------------------------------------------

        sess = tf.Session(config=config)
        print('Begin training:{}'.format(datetime.datetime.now()))
        merge_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(resultPath + '/log/train', sess.graph)
        valid_writer = tf.summary.FileWriter(resultPath + '/log/valid')
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        # restorer = tf.train.Saver(var_list=variables_to_restore)
        # try:
        #     restorer.restore(sess, pretrain_path)
        #     print('restore pretrained model ok')
        # except FileNotFoundError:
        #     print('Not found pretrained model ckpt')

        global LEARNING_RATE

        for itr in range(MAX_ITERATION):
            vol_batch, label_batch = utils_fun.get_data_train_2d(trainPath, batchsize=TRAIN_BATCHSIZE)
            vol_batch = np.expand_dims(vol_batch, axis=4)
            label_batch = np.expand_dims(label_batch, axis=4)
            valid_vol_batch, valid_seg_batch = utils_fun.get_data_train_2d(testPath, batchsize= 1)
            valid_vol_batch = np.expand_dims(valid_vol_batch, axis=4)
            valid_seg_batch = np.expand_dims(valid_seg_batch, axis=4)
            vol_shape = vol_batch.shape
            print('vol_shape: ', vol_shape)

            # ----------------------changed learning rate --------------------------------------------------------------------
            if (itr + 1) % DECAY_INTERVAL == 0:
                LEARNING_RATE = LEARNING_RATE * 0.90
                print('learning_rate:', LEARNING_RATE)

            # -----------------------------------------training training training training-------------------------------------
            feed = {LRate: LEARNING_RATE, image: vol_batch, annotation: label_batch, bn_flag: True, keep_prob: 0.8}
            sess.run(train_op, feed_dict= feed)
            train_loss_print, summary_str = sess.run([loss_reduce, merge_op], feed_dict=feed)
            print(itr, 'loss:', train_loss_print)
            train_writer.add_summary(summary_str, itr)

            valid_feed = {LRate: LEARNING_RATE, image: valid_vol_batch, annotation: valid_seg_batch, bn_flag: False, keep_prob: 1}
            valid_loss_print, summary_str = sess.run([loss_reduce, merge_op], feed_dict=valid_feed)
            valid_writer.add_summary(summary_str, itr)
            print('valid_loss:', valid_loss_print)

            if (itr + 1) % SAVE_CKPT_INTERVAL == 0:
                saver.save(sess, resultPath + 'ckpt/modle', global_step= (itr+1))

            # -------------------------------------Test Test Test Test--------------------------------------------------------------
            if itr == (MAX_ITERATION - 1):
                test_dirs = os.listdir(testPath + '/vol/')
                test_num = len(test_dirs)
                for i in range(test_num):
                    t_dir = test_dirs[i]
                    vol_batch, label_batch = utils_fun.get_batch_test_2d(testPath, t_dir)
                    vol_batch = np.expand_dims(vol_batch, axis=4)
                    label_batch = np.expand_dims(label_batch, axis=4)
                    test_feed = {image: vol_batch, annotation: label_batch, bn_flag: False, keep_prob: 1}
                    test_pred = sess.run(target, feed_dict=test_feed)
                    label_tosave = np.squeeze(label_batch, axis=4)
                    pred_tosave = np.squeeze(test_pred, axis=4)    # shape [1,30, 512, 512]
                    name_pre = t_dir[:-4]
                    print("test_itr:", name_pre)
                    utils_fun.SaveNpys(resultPath, name_pre, label_tosave, pred_tosave)


if __name__ == '__main__':
    print("Begin...")
    srcnn_run()
    print("Finished!")