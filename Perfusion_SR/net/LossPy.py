#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2018/10/13 15:50
# @Author  : ***

import tensorflow as tf

def EuclideanLoss(pred, label):
    return tf.reduce_mean(tf.square(pred-label))
