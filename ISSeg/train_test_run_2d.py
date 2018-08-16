'''

'''


import net.UNet_2D as UNet_2D
import net.NetDeepLab_2d as NetDeepLab_2d
import net.dense_fc as dense_fc
import utils.utils_fun as utils
import net.LossPy as LossPy
import os
import tensorflow as tf
import numpy as np
import datetime


trainPath = 'E:/ISSEG/Dataset/2018REGROUP/all/train/'
testPath = 'E:/ISSEG/Dataset/2018REGROUP/all/test/'

#change dir here ..............................................................
resultPath = 'D:/IESLES_Rst/CT_256/exp6/'

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
IMAGE_CHANNEL = 5
#IMAGE_DEPTH = 24


LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_ITERATION = 10
ITER_PER_EPOCH = 2100
STEPINTERVAL = 50000
CLASSNUM = 2
TRAIN_BATCHSIZE = 1


def training(lr, loss_val, va_list):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    optimizer = tf.train.AdamOptimizer(lr)
    with tf.control_dependencies(update_ops):
        grads = optimizer.compute_gradients(loss_val, var_list=va_list)
        return optimizer.apply_gradients(grads)

def FCNX_run():
    with tf.name_scope('inputs'):
        annotation = tf.placeholder(tf.int32, shape=[TRAIN_BATCHSIZE, IMAGE_WIDTH, IMAGE_HEIGHT, 1], name='annotation')   # shape BHWC
        image = tf.placeholder(tf.float32, shape=[TRAIN_BATCHSIZE, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL], name='image')           # shape BHWC

    bn_flag = tf.placeholder(tf.bool)
    keep_prob = tf.placeholder(tf.float32)
    train_batchsize = tf.placeholder(tf.int32)

    #logits, pred_annot, _,_ = UNet_2D.build_unet(tensor_in= image, BN_FLAG= bn_flag, BATCHSIZE= train_batchsize,
                                                # CLASSNUM= CLASSNUM, keep_porb= keep_prob, in_channels= IMAGE_CHANNEL)

    # logits, pred_annot = NetDeepLab_2d.LITS_DLab(tensor_in = image, BN_FLAG = bn_flag, BATCHSIZE = train_batchsize,
    #                                              CLASSNUM = CLASSNUM, KEEPPROB = keep_prob, IMGCHANNEL = IMAGE_CHANNEL)

    logits, pred_annot = dense_fc.build_dense_fc(tensor_in= image, BN_FLAG= bn_flag, BATCHSIZE= train_batchsize,
                                                 CLASSNUM= CLASSNUM,  IMGCHANNEL= IMAGE_CHANNEL,keep_prob= keep_prob)

    with tf.variable_scope('loss'):
        class_weight = tf.constant([0.15,1])
        #loss_reduce = LossPy.cross_entropy_loss(pred= logits, ground_truth= annotation, class_weight= class_weight)
        # l2_loss = [WEIGHT_DECAY * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'w' in v.name]
        # loss_reduce = tf.reduce_mean(loss) + tf.add_n(l2_loss)

        loss_reduce = LossPy.focal_loss(pred= logits, ground_truth= annotation)

        #loss_reduce = LossPy.dice_sqaure(pred = logits, ground_truth= annotation)
        tf.summary.scalar('loss', loss_reduce)

    with tf.variable_scope('valid_IOU'):
        iou = tf.placeholder(tf.float32)
        tf.summary.scalar('IOU', iou)

    with tf.variable_scope('trainOP'):
        LRate = tf.placeholder(tf.float32)
        trainable_vars = tf.trainable_variables()
        train_op = training(LRate, loss_reduce, trainable_vars)
        tf.summary.scalar('lr', LRate)

    with tf.variable_scope('fcnx') as scope:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        sess = tf.Session(config = config)
        print('Begin training:{}'.format(datetime.datetime.now()))

        merge_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(resultPath + '/log', sess.graph)
        sess.run(tf.global_variables_initializer())
        scope.reuse_variables()
        saver = tf.train.Saver()

        global LEARNING_RATE
        meanIOU = 0.001
        for itr in range(MAX_ITERATION):
            vol_batch, seg_batch = utils.get_data_train_2d(trainPath, batchsize= TRAIN_BATCHSIZE)
            vol_shape = vol_batch.shape
            print(vol_shape)

            #----------------------changed------------------------------------------------------
            if (itr + 1) % 10000 == 0:
                LEARNING_RATE = LEARNING_RATE * 0.90
                print('learning_rate:',LEARNING_RATE)
            # -------------------------validation with IOU each 10 epoch------------------------------------------------------
            if itr % (ITER_PER_EPOCH * 5) == 0:
                test_dirs = os.listdir(testPath + '/vol/')
                one_pred_or_label = one_label_and_pred = 0

                for t_dir in test_dirs:
                    vol_batch, seg_batch = utils.get_data_test_2d(testPath, t_dir)
                    test_feed = {image: vol_batch, annotation: seg_batch, bn_flag: False, keep_prob: 1,
                                 train_batchsize: 1}
                    test_pred_annotation = sess.run(pred_annot, feed_dict=test_feed)
                    label_batch = np.squeeze(seg_batch).astype(np.uint8)
                    pred_batch = np.squeeze(test_pred_annotation).astype(np.uint8)
                    label_bool = (label_batch == 1)
                    pred_bool = (pred_batch == 1)
                    union = np.logical_or(label_bool, pred_bool)
                    intersection = np.logical_and(label_bool, pred_bool)
                    one_pred_or_label = one_pred_or_label + np.count_nonzero(union)
                    one_label_and_pred = one_label_and_pred + np.count_nonzero(intersection)

                meanIOU = one_label_and_pred / (one_pred_or_label + 1e-4)
                print(meanIOU)
            #------------------------------------------------------------------------------------------------------------------------------

            feed = {LRate: LEARNING_RATE, iou: meanIOU, image: vol_batch, annotation: seg_batch, bn_flag: True, keep_prob: 0.8, train_batchsize: TRAIN_BATCHSIZE}
            sess.run(train_op, feed_dict= feed)
            train_loss_print, summary_str = sess.run([loss_reduce, merge_op], feed_dict=feed)
            print(itr, vol_batch.shape)
            print('loss:', train_loss_print)
            train_writer.add_summary(summary_str, itr)

            if (itr + 1) % STEPINTERVAL == 0:
                saver.save(sess, resultPath + 'ckpt/modle', global_step= (itr+1))
#-------------------------------------Test Test Test Test-------------------------------------------------------------------------------
            if itr == (MAX_ITERATION - 1):
                print('End training:{}'.format(datetime.datetime.now()))
                test_dirs = os.listdir(testPath + '/vol/')
                for tDir in test_dirs:
                    vol_batch, seg_batch = utils.get_data_test_2d(testPath, tDir)
                    vol_shape = vol_batch.shape
                    print(vol_shape)

                    test_feed = {image: vol_batch, annotation: seg_batch,bn_flag: False, keep_prob:1, train_batchsize: 1}
                    test_pred_annotation = sess.run([pred_annot], feed_dict=test_feed)
                    label_batch = np.squeeze(seg_batch)
                    pred_batch = np.squeeze(test_pred_annotation)
                    label_tosave = np.rot90(label_batch, 1).astype(np.uint8)
                    pred_tosave = np.rot90(pred_batch, 1).astype(np.uint8)
                    label_tosave = np.fliplr(label_tosave)
                    pred_tosave = np.fliplr(pred_tosave)

                    namePre = tDir[:-4]
                    print("test_itr:", namePre)
                    utils.save_imgs_IELES_2d(resultPath, namePre, label_tosave, pred_tosave)
                    utils.save_npys(resultPath, namePre, label_tosave, pred_tosave)




if __name__ == '__main__':
    print("Begin...")
    FCNX_run()
    print("Finished!")
