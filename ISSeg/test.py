#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2018/10/8 19:11
# @Author  : Lionzhu

# the deeplab_v3 test py

import net.DeepLabV3 as DeepLabV3
import tensorflow as tf
import os
import math
import utils.utils_fun as utils
import numpy as np
import argparse
from preprocess import training

path = 'E:/ISSEG/Dataset/2018REGROUP/128/4d_VFMT/'
testPath = path + 'test/'
resultPath = 'D://DLexp/IESLES_Rst/CT_128_VFMT/exp13/'

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

class_label = [v for v in range((args.number_of_class+1))]
image = tf.placeholder(tf.float32, shape= [None, None, None, 4])
annotation= tf.placeholder(tf.int32, shape= [None, None, None])

logits = DeepLabV3.deeplab_v3(image, args, is_training=False, reuse=False)
labels_batch_tf, logits_batch_tf = training.get_valid_logits_and_labels(annotation_batch_tensor= annotation,
                                                                        logits_batch_tensor= logits,
                                                                        class_labels= class_label)

prediction = tf.argmax(logits, axis= -1)

with tf.Session() as sess:
    model_file = tf.train.latest_checkpoint(resultPath +'ckpt/')
    saver = tf.train.Saver()
    saver.restore(sess, model_file)

    test_dirs = os.listdir(testPath + '/seg')
    test_num = len(test_dirs)
    test_times = math.ceil(test_num / args.batch_size)

    for i in range(int(test_times)):
        if i != (test_times - 1):
            tDir = test_dirs[i*args.batch_size : (i+1)*args.batch_size]
            vol_batch, seg_batch = utils.get_data_test_2d(testPath, tDir, args.batch_size)
        if i == (test_times - 1):
            tDir = test_dirs[(test_num - args.batch_size) : test_num]
            vol_batch, seg_batch = utils.get_data_test_2d(testPath, tDir, args.batch_size)

        test_feed = {image: vol_batch}
        test_pred = sess.run(prediction, feed_dict= test_feed)

        for j in range (args.batch_size):
            label_batch = np.squeeze(seg_batch[j, ...])
            pred_batch = np.squeeze(test_pred[j, ...])
            label_tosave = np.rot90(label_batch, 1).astype(np.uint8)
            pred_tosave = np.rot90(pred_batch, 1).astype(np.uint8)
            label_tosave = np.fliplr(label_tosave)
            pred_tosave = np.fliplr(pred_tosave)

            namePre = tDir[j]
            namePre = namePre[:-4]
            print("test_itr:", namePre)
            utils.save_imgs_IELES_2d(resultPath, namePre, label_tosave, pred_tosave)
            utils.save_npys(resultPath, namePre, label_tosave, pred_tosave)