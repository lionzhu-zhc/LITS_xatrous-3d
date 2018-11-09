'''
train ctp reconstruction net and test
010916 Lionzhu
'''

import numpy as np
import tensorflow as tf
import resNetPro
import utils
import os
import math

path = 'E:/Cerebral Infarction/perfusion_30c/perfusion_npy/'
trainPath = path + 'train/'
testPath = path + 'test/'
resultPath = 'D:/CTPRecons_Rst/exp4/'

IMGChannel = 3
Epoch = 20
IterPerEpoch = 100
MaxIter = Epoch * IterPerEpoch
# MaxIter =5
SAVE_CKPT_INTERVAL = IterPerEpoch * Epoch
BS = 12

init_lr = 1e-3

def training(lr, loss_val, va_list):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    optimizer = tf.train.AdamOptimizer(lr)
    with tf.control_dependencies(update_ops):
        grads = optimizer.compute_gradients(loss_val, var_list= va_list)
        return optimizer.apply_gradients(grads)


def run():
    with tf.name_scope('Input'):
        data = tf.placeholder(tf.float32, [None,None,None,30])
        ctp = tf.placeholder(tf.float32, [None,None,None,1])
        train_placeholder = tf.placeholder(tf.bool, shape=[])

    lr = tf.placeholder(tf.float32, shape= [])

    annot = resNetPro.resnet(data, train_placeholder)
    l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    loss = tf.reduce_mean(tf.square(annot - ctp)) + l2_loss

    with tf.name_scope('trainOP'):
        train_vars = tf.trainable_variables()
        train_op = training(lr, loss, train_vars)
    # updata_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # train = tf.group(tf.train.AdamOptimizer(lr).minimize(loss), *updata_op)
    init_global = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess= tf.Session(config = config)

    tf.summary.scalar('Loss', loss)
    tf.summary.scalar('LR', lr)
    merge_op = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(resultPath + '/log', sess.graph)

    saver = tf.train.Saver()
    sess.run(init_global)
    sess.run(tf.local_variables_initializer())
    global init_lr

    for itr in range(MaxIter):
        img_batch, ctp_batch = utils.GetTrainData_2d(trainPath, BS= BS)
        print('img_batch shape:', img_batch.shape)

        if (itr + 1) % (IterPerEpoch * 10) == 0:
            init_lr = init_lr * 0.90
            print('learning_rate:', init_lr)

        feed = {lr: init_lr, data: img_batch, ctp: ctp_batch, train_placeholder: True}
        sess.run(train_op, feed_dict= feed)
        loss_print, summary_str, anot = sess.run([loss, merge_op, annot], feed_dict= feed)
        print('loss:', itr, loss_print)
        train_writer.add_summary(summary_str, itr)

        if (itr + 1) % SAVE_CKPT_INTERVAL == 0:
            saver.save(sess, resultPath + 'ckpt/modle', global_step=(itr + 1))

        #-----------------------------test test------------------------------------------------------
        if itr == (MaxIter - 1):
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

                test_feed = {lr: init_lr, data: img_batch, ctp: ctp_batch, train_placeholder: False}
                test_pred_ctp = sess.run(annot, feed_dict= test_feed)

                for j in range(BS):
                    label_batch = np.squeeze(ctp_batch[j, ...])
                    pred_batch = np.squeeze(test_pred_ctp[j, ...])
                    # label_tosave = np.rot90(label_batch, 1).astype(np.float32)
                    # pred_tosave = np.rot90(pred_batch, 1).astype(np.float32)
                    # label_tosave = np.fliplr(label_tosave)
                    # pred_tosave = np.fliplr(pred_tosave)

                    label_tosave = label_batch
                    pred_tosave = pred_batch

                    namePre = tDir[j]
                    namePre = namePre[:-4]
                    print("test_itr:", namePre)
                    utils.SaveNpys(resultPath, namePre, label_tosave, pred_tosave)

if __name__ == '__main__':
    run()