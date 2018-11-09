#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2018/9/27 16:35
# @Author  : LionZhu
# the train file for deeplab_v3

import argparse
import tensorflow as tf
import numpy as np
from net import DeepLabV3 as deeplab
import os
import tensorflow.contrib.slim as slim
import net.LossPy as LossPy
import utils.utils_fun as utils
import math
from preprocess import training

parser = argparse.ArgumentParser()

env_arg = parser.add_argument_group('Training params')
env_arg.add_argument('--batch_norm_epsilon', type= float, default= 1e-5, help= 'batch normlization epsilon')
env_arg.add_argument('--batch_norm_decay', type= float, default= 0.997, help= 'batch normlization epsilon')
env_arg.add_argument("--number_of_class", type=int, default=2, help="Number of classes to be predicted.")
env_arg.add_argument("--l2_regularizer", type=float, default=0.0001, help="l2 regularizer parameter.")
env_arg.add_argument('--starting_learning_rate', type=float, default=1e-5, help="initial learning rate.")
env_arg.add_argument("--output_stride", type=int, default=16, help="Spatial Pyramid Pooling rates")
env_arg.add_argument("--gpu_id", type=int, default=0, help="Id of the GPU to be used")
env_arg.add_argument("--crop_size", type=int, default=513, help="Image Cropsize.")
env_arg.add_argument("--resnet_model", default="resnet_v2_50",
                    choices=["resnet_v2_50", "resnet_v2_101", "resnet_v2_152", "resnet_v2_200"],
                    help="Resnet model to use as feature extractor. Choose one of: resnet_v2_50 or resnet_v2_101")
env_arg.add_argument("--current_best_val_loss", type=int, default=99999, help="Best validation loss value.")
env_arg.add_argument("--accumulated_validation_miou", type=int, default=0, help="Accumulated validation intersection over union.")
env_arg.add_argument("--batch_size", type=int, default=12, help="Batch size for network train.")
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

path = 'E:/ISSEG/Dataset/2018REGROUP/128/4d_VFMT/'
trainPath = path + 'train/'
testPath = path + 'test/'
resnet_ckpt_path = 'D:/TensorflowDemo/resnet_deeplab_ckpt/'
#change dir here ..............................................................
resultPath = 'D:/DLexp/IESLES_Rst/CT_128_VFMT/exp13/'

IMG_CHANNEL = 4
ITER_PER_EPOCH = 200
DECAY_INTERVAL = ITER_PER_EPOCH * 10
MAX_ITERATION = ITER_PER_EPOCH * 100
SAVE_CKPT_INTERVAL = ITER_PER_EPOCH * 50


class_label = [v for v in range((args.number_of_class+1))]
# class_label[-1] = 255 #unknow

is_training_pl = tf.placeholder(tf.bool, shape= [])
image = tf.placeholder(tf.float32, shape= [None, None, None, IMG_CHANNEL])
annotation= tf.placeholder(tf.int32, shape= [None, None, None])


logits = tf.cond(is_training_pl, true_fn= lambda : deeplab.deeplab_v3(image, args, is_training= True, reuse= False),
                                 false_fn= lambda : deeplab.deeplab_v3(image, args, is_training= False, reuse= True))

labels_batch_tf, logits_batch_tf = training.get_valid_logits_and_labels(annotation_batch_tensor= annotation,
                                                                        logits_batch_tensor= logits,
                                                                        class_labels= class_label)

loss = LossPy.cross_entropy_loss(pred= logits_batch_tf, ground_truth= labels_batch_tf)
prediction = tf.argmax(logits, axis= -1)
tf.summary.scalar('loss', loss)

with tf.variable_scope('optimizer_vars'):
    global_step = tf.Variable(0, trainable= False)
    optimizer = tf.train.AdamOptimizer(learning_rate= args.starting_learning_rate)
    train_step = slim.learning.create_train_op(loss, optimizer, global_step= global_step)

vars_to_restore = slim.get_variables_to_restore(exclude= [args.resnet_model + '/logits', args.resnet_model + '/conv1', 'optimizer_vars', 'DeepLab_v3/ASPP_layer', 'DeepLab_v3/logits'])

miou, updata_op = tf.contrib.metrics.streaming_mean_iou(tf.argmax(logits_batch_tf, axis= 1),
                                                        tf.argmax(labels_batch_tf, axis= 1), num_classes= args.number_of_class)
with tf.name_scope('miou'):
    tf.summary.scalar('mIou', miou)

merge_op = tf.summary.merge_all()
restorer = tf.train.Saver(vars_to_restore)
saver = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config= config) as sess:
    train_writer = tf.summary.FileWriter(resultPath + '/log', sess.graph)

    #create a saver
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    try:
        restorer.restore(sess, resnet_ckpt_path+ 'model.ckpt')
        print('successfully restore pre-trained')
    except FileNotFoundError:
        print('not found restore ckpt')


    for itr in range(MAX_ITERATION):
        if (itr + 1) % SAVE_CKPT_INTERVAL == 0:
            saver.save(sess, resultPath + 'ckpt/modle', global_step=(global_step_np + 1))
        # -------------validation steps----------------------------------------------------------------------------
        if (itr + 1) % (ITER_PER_EPOCH * 10) == 0:
            test_dirs = os.listdir(testPath + '/vol/')
            test_num = len(test_dirs)
            test_times = math.ceil(test_num / args.batch_size)
            valid_miou_avg = 0
            for i in range(test_times):
                if i != (test_times - 1):
                    tDir = test_dirs[i * args.batch_size: (i + 1) * args.batch_size]
                    vol_batch, seg_batch = utils.get_data_test_2d(testPath, tDir, args.batch_size)
                if i == (test_times - 1):
                    tDir = test_dirs[(test_num - args.batch_size): test_num]
                    vol_batch, seg_batch = utils.get_data_test_2d(testPath, tDir, args.batch_size)
                seg_batch = seg_batch.squeeze()
                test_feed = {image: vol_batch, annotation: seg_batch, is_training_pl: False}
                valid_miou,global_step_np, summary_str = sess.run([miou, global_step, merge_op], feed_dict= test_feed)
                valid_miou_avg += valid_miou
            valid_miou_avg /= test_times

            train_writer.add_summary(summary_str, global_step_np)
        else:
            # --------------train steps--------------------------------------------------------------------------------
            vol_batch, seg_batch = utils.get_data_train_2d(trainPath, batchsize=args.batch_size)
            seg_batch = seg_batch.squeeze()
            feed = {image: vol_batch, annotation: seg_batch, is_training_pl: True}
            _, global_step_np, train_loss, summary_str = sess.run([train_step, global_step, loss, merge_op],
                                                                  feed_dict=feed)
            print('iter:', itr)
            print('train_loss', train_loss)
            train_writer.add_summary(summary_str, global_step_np)


    train_writer.close()

