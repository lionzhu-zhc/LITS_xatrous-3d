'''
Lionzhu 0407
training and test lianxin data, use Network.py pspnet

0414
lianxin, use NetDeepLab deeplab+FCN
'''

import NetDeepLab_V2
import utils
import LossPy
import os
import tensorflow as tf
import numpy as np
import datetime


trainPath = 'E:/Lianxin_40/LxData_600_cut_280/train_npy_cutslice/'
testPath = 'E:/Lianxin_40/LxData_600_cut_280/test_npy/'

#change dir here ..............................................................
resultPath = 'D:/LITS_Rst/FCNDEEPLAB_lx280/exp7/'

IMAGE_WIDTH = 280
IMAGE_HEIGHT = 280
IMAGE_DEPTH = 24

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
MAX_ITERATION = 15000
CLASSNUM = 2


def training(lr, loss_val, va_list):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    optimizer = tf.train.AdamOptimizer(lr)
    with tf.control_dependencies(update_ops):
        grads = optimizer.compute_gradients(loss_val, var_list=va_list)
        return optimizer.apply_gradients(grads)

def FCNX_run():
    with tf.name_scope('inputs'):
        annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH, 1], name='annotation')
        image = tf.placeholder(tf.float32, shape=[None, IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH, 1], name='image')

    bn_flag = tf.placeholder(tf.bool)
    train_batchsize = tf.placeholder(tf.int32)
    pred_annot, logits = NetDeepLab_V2.LITS_DLab(tensor_in= image, BN_FLAG= bn_flag, BATCHSIZE= train_batchsize,
                                            IMAGE_DEPTH= IMAGE_DEPTH, IMAGE_HEIGHT = IMAGE_HEIGHT, IMAGE_WIDTH= IMAGE_WIDTH, CLASSNUM= CLASSNUM)


    with tf.name_scope('loss'):
        # class_weight = tf.constant([0.15, 1, 25])
        #loss = LossPy.cross_entropy_loss(pred= logits, ground_truth= annotation, class_weight= class_weight)
        #
        # l2_loss = [WEIGHT_DECAY * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'w' in v.name]
        # loss_reduce = tf.reduce_mean(loss) + tf.add_n(l2_loss)
        loss_reduce = LossPy.dice_sqaure(pred = logits, ground_truth= annotation)
        tf.summary.scalar('loss', loss_reduce)

    with tf.name_scope('trainOP'):
        LRate = tf.placeholder(tf.float32)
        trainable_vars = tf.trainable_variables()
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        #     train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss_reduce)
        train_op = training(LRate, loss_reduce, trainable_vars)

    with tf.variable_scope('fcnx') as scope:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        sess = tf.Session(config = config)
        print('Begin training:{}'.format(datetime.datetime.now()))

        merge_op = tf.summary.merge_all()
        writer = tf.summary.FileWriter(resultPath + '/log', sess.graph)
        sess.run(tf.global_variables_initializer())
        scope.reuse_variables()
        saver = tf.train.Saver()

        for itr in range(MAX_ITERATION):
            vol_batch, seg_batch = utils.get_data_train(trainPath)
            # vol_batch_2, seg_batch_2 = utils.get_data_train(trainPath)
            # vol_batch = np.concatenate((vol_batch, vol_batch_2), axis= 0)
            # seg_batch = np.concatenate((seg_batch, seg_batch_2), axis= 0)

            global LEARNING_RATE
            if (itr + 1) % 1000 == 0:
                LEARNING_RATE = LEARNING_RATE * 0.90
                print(LEARNING_RATE)

            # lr = LEARNING_RATE * math.pow((1 - itr/ MAX_ITERATION), 0.9)
            # print(LEARNING_RATE)
            #--------------------------------------------train_batchsize here--------------------------------------------------------
            feed = {LRate: LEARNING_RATE, image: vol_batch, annotation: seg_batch, bn_flag: True, train_batchsize: 1}
            sess.run(train_op, feed_dict= feed)
            train_loss_print, summary_str = sess.run([loss_reduce, merge_op], feed_dict=feed)
            print(itr, vol_batch.shape)
            print('loss:', train_loss_print)
            writer.add_summary(summary_str, itr)

            if (itr + 1) % 5000 == 0:
                saver.save(sess, resultPath + 'ckpt/modle', global_step= (itr+1) )

#-------------------------------------Test Test Test Test-------------------------------------------------------------------------------
            if itr == (MAX_ITERATION - 1):
                print('End training:{}'.format(datetime.datetime.now()))
                test_dirs = os.listdir(testPath + '/vol/')
                for tDir in test_dirs:
                    vol_batch, seg_batch = utils.get_data_test(testPath, tDir)
                    test_feed = {image: vol_batch, annotation: seg_batch,bn_flag: False, train_batchsize: 1}
                    test_pred_annotation = sess.run([pred_annot], feed_dict=test_feed)
                    label_batch = np.squeeze(seg_batch)
                    pred_batch = np.squeeze(test_pred_annotation)
                    label_tosave = np.transpose(label_batch, (2, 1, 0))
                    pred_tosave = np.transpose(pred_batch, (2, 1, 0))

                    namePre = tDir[:-4]
                    print("test_itr:", namePre)
                    utils.save_imgs(resultPath, namePre, label_tosave, pred_tosave, IMAGE_DEPTH, IMAGE_WIDTH, IMAGE_HEIGHT)
                    utils.save_npys(resultPath, namePre, label_tosave, pred_tosave)




if __name__ == '__main__':
    print("Begin...")
    FCNX_run()
    print("Finished!")
