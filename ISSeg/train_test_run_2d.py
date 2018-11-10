'''

'''


import net.UNet_2D as UNet_2D
import net.NetDeepLab_2d as NetDeepLab_2d
import net.dense_fc as dense_fc
import net.PSP_Atrous_2d as PSP_Atrous
import net.ResU_2d as resu_2d
import utils.utils_fun as utils
import net.LossPy as LossPy
import os
import tensorflow as tf
import numpy as np
import datetime
import math
slim = tf.contrib.slim

#---------------------paths--------------------------------------------------
path = 'E:/ISSEG/Dataset/2018REGROUP/128/5c/'
trainPath = path + 'train/'
testPath = path + 'test/'
#change dir here ............................................................
resultPath = 'D:/DLexp/IESLES_Rst/CT_128/exp23/'
# resultPath = 'D:/DLexp/IESLES_Rst/CT_128/111/12/exp27/'
pretrain_path = 'D://resnet_v2_50/resnet_v2_50.ckpt'
#---------------------paths--------------------------------------------------

#-----------------Img paras----------------------------------------------
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_CHANNEL = 5
CLASSNUM = 2
#-----------------Img paras----------------------------------------------

#------------training paras-------------------------------------------
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCH = 150
ITER_PER_EPOCH = 40
DECAY_INTERVAL = ITER_PER_EPOCH * EPOCH // 15
MAX_ITERATION = ITER_PER_EPOCH * EPOCH
SAVE_CKPT_INTERVAL = ITER_PER_EPOCH * EPOCH // 2
TRAIN_BATCHSIZE = 64

ValidFlag = True
TestFlag = True
#------------training paras-------------------------------------------


def early_training(lr, loss_val, va_list):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    optimizer = tf.train.AdamOptimizer(lr)
    with tf.control_dependencies(update_ops):
        grads = optimizer.compute_gradients(loss_val, var_list=va_list)
        return optimizer.apply_gradients(grads)

def late_training(lr, loss_val, va_list):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    optimizer = tf.train.GradientDescentOptimizer(lr)
    with tf.control_dependencies(update_ops):
        grads = optimizer.compute_gradients(loss_val, var_list=va_list)
        return optimizer.apply_gradients(grads)

def FCNX_run():
    with tf.name_scope('inputs'):
        annotation = tf.placeholder(tf.int32, shape=[TRAIN_BATCHSIZE, IMAGE_WIDTH, IMAGE_HEIGHT, 1], name='annotation')   # shape BHWC
        image = tf.placeholder(tf.float32, shape=[TRAIN_BATCHSIZE, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL], name='image')   # shape BHWC

    bn_flag = tf.placeholder(tf.bool)
    keep_prob = tf.placeholder(tf.float32)
    train_batchsize = tf.placeholder(tf.int32)

    # ----------------------choose net model here------------------------------------------------------------------------------------------------
    #logits, pred_annot, _,_ = UNet_2D.build_unet(tensor_in= image, BN_FLAG= bn_flag, BATCHSIZE= train_batchsize,
                                                # CLASSNUM= CLASSNUM, keep_porb= keep_prob, in_channels= IMAGE_CHANNEL)

    # logits, pred_annot = NetDeepLab_2d.LITS_DLab(tensor_in = image, BN_FLAG = bn_flag, BATCHSIZE = train_batchsize,
    #                                              CLASSNUM = CLASSNUM, KEEPPROB = keep_prob, IMGCHANNEL = IMAGE_CHANNEL)

    # logits, pred_annot = PSP_Atrous.build_net(tensor_in=image, BN_FLAG=bn_flag, BATCHSIZE=train_batchsize,CLASSNUM=CLASSNUM,
    #                                           KEEPPROB=keep_prob, IMGCHANNEL=IMAGE_CHANNEL)

    # logits, pred_annot = dense_fc.build_dense_fc(tensor_in= image, BN_FLAG= bn_flag, BATCHSIZE= train_batchsize,
    #                                              CLASSNUM= CLASSNUM, keep_prob= keep_prob)

    logits, pred_annot = resu_2d.resU_2d(tensor_in= image, BN_FLAG= bn_flag, CLASSNUM= CLASSNUM)

    # ----------------------choose net model here------------------------------------------------------------------------------------------------

    with tf.variable_scope('loss'):
        class_weight = tf.constant([0.15,1])
        loss_reduce = LossPy.cross_entropy_loss(pred= logits, ground_truth= annotation, class_weight= class_weight)
        # valid_loss = LossPy.cross_entropy_loss(pred= logits, ground_truth= annotation, class_weight= class_weight)
        # l2_loss = [WEIGHT_DECAY * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'w' in v.name]
        # loss_reduce = tf.reduce_mean(loss) + tf.add_n(l2_loss)

        #loss_reduce = LossPy.focal_loss(pred= logits, ground_truth= annotation)

        # loss_reduce = LossPy.dice_sqaure(pred = logits, ground_truth= annotation)

        tf.summary.scalar('loss', loss_reduce)
        # tf.summary.scalar('valid_loss', valid_loss)

    with tf.variable_scope('valid_IOU'):
        iou = tf.placeholder(tf.float32)
        tf.summary.scalar('IOU', iou)

    with tf.variable_scope('trainOP'):
        LRate = tf.placeholder(tf.float32)
        trainable_vars = tf.trainable_variables()
        train_op = early_training(LRate, loss_reduce, trainable_vars)
        tf.summary.scalar('lr', LRate)

    with tf.variable_scope('fcnx') as scope:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # ----------get ckpt to restore--------------------------
        ckpt_exclude_scopes = 'NewDeepLab, resnet_v2_50/conv1'
        exclusions = None
        if ckpt_exclude_scopes:
            exclusions = [scope.strip() for scope in ckpt_exclude_scopes.split(',')]
        vars_to_restore = []
        for var in slim.get_model_variables():
            excluded = False
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    excluded = True
            if not excluded:
                vars_to_restore.append(var)

        sess = tf.Session(config = config)
        print('Begin training:{}'.format(datetime.datetime.now()))

        merge_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(resultPath + '/log/train', sess.graph)
        valid_writer = tf.summary.FileWriter(resultPath + '/log/valid')
        sess.run(tf.global_variables_initializer())
        scope.reuse_variables()
        saver = tf.train.Saver()

        restorer = tf.train.Saver(var_list= vars_to_restore)
        try:
            restorer.restore(sess, pretrain_path)
            print('restore ok')
        except FileNotFoundError:
            print('Not found pretrained model ckpt')

        global LEARNING_RATE
        meanIOU = 0.001
        for itr in range(MAX_ITERATION):
            vol_batch, seg_batch = utils.get_data_train_2d(trainPath, batchsize= TRAIN_BATCHSIZE)
            valid_vol_batch, valid_seg_batch = utils.get_data_train_2d(testPath, batchsize=TRAIN_BATCHSIZE)
            vol_shape = vol_batch.shape
            print(vol_shape)

            #----------------------changed learning rate --------------------------------------------------------------------
            if (itr + 1) % DECAY_INTERVAL == 0:
                LEARNING_RATE = LEARNING_RATE * 0.90
                print('learning_rate:',LEARNING_RATE)

            # ---------------------validation with IOU each 10 epoch------------------------------------------------------
            if (itr + 1)% (ITER_PER_EPOCH * 10) == 0 and ValidFlag:
                test_dirs = os.listdir(testPath + '/vol/')
                one_pred_or_label = one_label_and_pred = 0
                test_num = len(test_dirs)
                test_times = math.ceil(test_num / TRAIN_BATCHSIZE)
                for i in range(test_times):
                    if i != (test_times - 1):
                        tDir = test_dirs[i * TRAIN_BATCHSIZE: (i + 1) * TRAIN_BATCHSIZE]
                        vol_batch, seg_batch = utils.get_data_test_2d(testPath, tDir, TRAIN_BATCHSIZE)
                    if i == (test_times - 1):
                        tDir = test_dirs[(test_num - TRAIN_BATCHSIZE): test_num]
                        vol_batch, seg_batch = utils.get_data_test_2d(testPath, tDir, TRAIN_BATCHSIZE)
                    test_feed = {image: vol_batch, annotation: seg_batch, bn_flag: False, keep_prob: 1,
                                 train_batchsize: TRAIN_BATCHSIZE}
                    test_pred_annotation = sess.run(pred_annot, feed_dict=test_feed)
                    for j in range(TRAIN_BATCHSIZE):
                        label_batch = np.squeeze(seg_batch[j, ...]).astype(np.uint8)
                        pred_batch = np.squeeze(test_pred_annotation[j, ...]).astype(np.uint8)
                        label_bool = (label_batch == 1)
                        pred_bool = (pred_batch == 1)
                        union = np.logical_or(label_bool, pred_bool)
                        intersection = np.logical_and(label_bool, pred_bool)
                        one_pred_or_label = one_pred_or_label + np.count_nonzero(union)
                        one_label_and_pred = one_label_and_pred + np.count_nonzero(intersection)
                meanIOU = one_label_and_pred / (one_pred_or_label + 1e-4)
                print('valid meanIOU', meanIOU)

            #-----------------------------------------training training training training------------------------------------------------------------------
            feed = {LRate: LEARNING_RATE, iou: meanIOU, image: vol_batch, annotation: seg_batch, bn_flag: True, keep_prob: 1, train_batchsize: TRAIN_BATCHSIZE}
            sess.run(train_op, feed_dict= feed)
            # train_loss_print, summary_str = sess.run([loss_reduce, merge_op], feed_dict=feed)
            train_loss_print, summary_str= sess.run([loss_reduce, merge_op], feed_dict=feed)
            train_writer.add_summary(summary_str, itr)
            valid_feed = {LRate: LEARNING_RATE, iou: meanIOU, image: valid_vol_batch, annotation: valid_seg_batch, bn_flag: False, keep_prob: 1, train_batchsize: TRAIN_BATCHSIZE}
            valid_loss_print, summary_str = sess.run([loss_reduce, merge_op], feed_dict= valid_feed)
            valid_writer.add_summary(summary_str, itr)
            print(itr, vol_batch.shape)
            print('loss:', train_loss_print)
            print('valid_loss:', valid_loss_print)

            if (itr + 1) % SAVE_CKPT_INTERVAL == 0:
                saver.save(sess, resultPath + 'ckpt/modle', global_step= (itr+1))

            #-------------------------------------Test Test Test Test-------------------------------------------------------------------------------
            if itr == (MAX_ITERATION - 1) and TestFlag:
                print('End training:{}'.format(datetime.datetime.now()))
                test_dirs = os.listdir(testPath + '/vol/')
                test_num = len(test_dirs)
                test_times = math.ceil(test_num / TRAIN_BATCHSIZE)
                for i in range(test_times):
                    if i != (test_times - 1):
                        tDir = test_dirs[i * TRAIN_BATCHSIZE : (i+1) * TRAIN_BATCHSIZE]
                        vol_batch, seg_batch = utils.get_data_test_2d(testPath, tDir, TRAIN_BATCHSIZE)
                    if i == (test_times - 1):
                        tDir = test_dirs[(test_num-TRAIN_BATCHSIZE) : test_num]
                        vol_batch, seg_batch = utils.get_data_test_2d(testPath, tDir, TRAIN_BATCHSIZE)
                    test_feed = {image: vol_batch, annotation: seg_batch,bn_flag: False, keep_prob:1, train_batchsize: TRAIN_BATCHSIZE}
                    test_pred_annotation = sess.run(pred_annot, feed_dict=test_feed)
                    for j in range(TRAIN_BATCHSIZE):
                        label_batch = np.squeeze(seg_batch[j,...])
                        pred_batch = np.squeeze(test_pred_annotation[j,...])
                        label_tosave = np.rot90(label_batch, 1).astype(np.uint8)
                        pred_tosave = np.rot90(pred_batch, 1).astype(np.uint8)
                        label_tosave = np.fliplr(label_tosave)
                        pred_tosave = np.fliplr(pred_tosave)  

                        namePre = tDir[j]
                        namePre = namePre[:-4]
                        print("test_itr:", namePre)
                        utils.save_imgs_IELES_2d(resultPath, namePre, label_tosave, pred_tosave)
                        utils.save_npys(resultPath, namePre, label_tosave, pred_tosave)

        train_writer.close()
        valid_writer.close()



if __name__ == '__main__':
    print("Begin...")
    FCNX_run()
    print("Finished!")
