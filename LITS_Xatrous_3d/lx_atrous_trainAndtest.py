'''
Lionzhu 0407
training and test lianxin data, use Network.py pspnet

0414
lianxin, use NetDeepLab deeplab+FCN
'''

import tensorflow as tf
import NetDeepLab
import utils
import math
import os
import numpy as np

trainPath = 'E:/Lianxin_LITS/lxData_rs_600_cut_280/train_cutslice_npy/'
testPath = 'E:/Lianxin_LITS/lxData_rs_600_cut_280/test_npy/'

#change dir here .........................................
resultPath = 'D:/LITS_Rst/FCNDEEPLAB_lx280/exp1/'

IMAGE_WIDTH = 280
IMAGE_HEIGHT = 280
IMAGE_DEPTH = 24

LEARNING_RATE = 1e-4
MAX_ITERATION = 30000
CLASSNUM = 3


def training(lr, loss_val, va_list):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    optimizer = tf.train.AdamOptimizer(lr)
    print ('lr:',lr)
    with tf.control_dependencies(update_ops):
        grads = optimizer.compute_gradients(loss_val, var_list=va_list)
        return optimizer.apply_gradients(grads)



def FCNX_run():
    with tf.name_scope('inputs'):
        annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH, 1], name='annotation')
        image = tf.placeholder(tf.float32, shape=[None, IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH, 1], name='image')

    bn_flag = tf.placeholder(tf.bool)
    train_batchsize = tf.placeholder(tf.int32)
    pred_annot, logits = NetDeepLab.LITS_DLab(tensor_in= image, BN_FLAG= bn_flag, BATCHSIZE= train_batchsize,
                                            IMAGE_DEPTH= IMAGE_DEPTH, IMAGE_HEIGHT = IMAGE_HEIGHT, IMAGE_WIDTH= IMAGE_WIDTH, CLASSNUM= CLASSNUM)

    with tf.name_scope('loss'):
        class_weight = tf.constant([0.15, 1, 25])
        current_weight = tf.gather(class_weight, tf.squeeze(annotation, axis=4))
        loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(
            logits=logits, labels=tf.squeeze(annotation, squeeze_dims=[4]), weights=current_weight))
        tf.summary.scalar('loss', loss)

    with tf.name_scope('trainOP'):
        LRate = tf.placeholder(tf.float32)
        trainable_vars = tf.trainable_variables()
        train_op = training(LRate, loss, trainable_vars)

    with tf.variable_scope('fcnx') as scope:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        sess = tf.Session(config = config)
        merge_op = tf.summary.merge_all()
        writer = tf.summary.FileWriter(resultPath + '/log', sess.graph)
        sess.run(tf.global_variables_initializer())
        scope.reuse_variables()
        saver = tf.train.Saver()

        for itr in range(MAX_ITERATION):
            vol_batch, seg_batch = utils.get_data_train(trainPath)

            global LEARNING_RATE
            if (itr + 1) % 1000 == 0:
                LEARNING_RATE = LEARNING_RATE * 0.90
                print(LEARNING_RATE)

            # lr = LEARNING_RATE * math.pow((1 - itr/ MAX_ITERATION), 0.9)
            # print(LEARNING_RATE)

            feed = {LRate: LEARNING_RATE, image: vol_batch, annotation: seg_batch, bn_flag: True, train_batchsize: 1}
            sess.run(train_op, feed_dict= feed)
            train_loss_print, summary_str = sess.run([loss, merge_op], feed_dict=feed)
            print(itr, vol_batch.shape)
            print('loss:', train_loss_print)
            writer.add_summary(summary_str, itr)

            if (itr + 1) % 10000 == 0:
                saver.save(sess, resultPath + 'ckpt/modle', global_step= (itr+1) )

##############################Test Test Test Test#########################################################################################
            if itr == (MAX_ITERATION - 1):
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
