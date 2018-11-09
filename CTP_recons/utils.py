# -*- encoding:utf-8 -*-
import tensorflow as tf
import numpy as np
import pydicom
import os
from PIL import Image
import random

Height = 512
Width = 512
Depth = 200
CropHeight = 44
CropWidth = 44
CropDepth = 24
BatchSize=30
Mean=513#513.7824
var=1#486.0686

def _crop(low_pro,high_pro,writer):
    for x in range(0,Height,CropHeight-6):
        for y in range(0,Width,CropWidth-6):
            for z in range(0,Depth,CropDepth-6):
                if x+CropHeight>Height:
                    x = Height-CropHeight
                if y+CropWidth>Width:
                    y = Width - CropWidth
                if z+CropDepth>Depth:
                    z = Depth - CropDepth
                low_patch = low_pro[x:x+CropHeight,y:y+CropWidth,z:z+CropDepth]
                high_patch = high_pro[x:x+CropHeight,y:y+CropWidth,z:z+CropDepth]
                low_patch_raw = low_patch.tostring()
                high_patch_raw = high_patch.tostring()
                example = tf.train.Example(features = tf.train.Features(feature = {
                    'low_pro': tf.train.Feature(bytes_list = tf.train.BytesList(value = [low_patch_raw])),
                    'high_pro': tf.train.Feature(bytes_list = tf.train.BytesList(value = [high_patch_raw]))
                }))
                writer.write(example.SerializeToString())

def make_tf_records(low_patient_list,high_patient_list,records_path,):
    writer = tf.python_io.TFRecordWriter(records_path)
    for low_patient,high_patient in zip(low_patient_list,high_patient_list):
        low_data=np.zeros([Height,Width,Depth],np.float32)
        high_data=np.zeros([Height,Width,Depth],np.float32)
        mask=Image.open(os.path.join(high_patient,'Mask.tif'))
        mask=np.array(mask,dtype=np.float32)
        mask=mask/255
        for i in range(Depth):
            lowfile = pydicom.read_file(os.path.join(low_patient,str(i+1)+'.dcm'))
            highfile = pydicom.read_file(os.path.join(high_patient,str(i+1)+'.dcm'))
            low=lowfile.pixel_array.astype(np.float32)
            high=highfile.pixel_array.astype(np.float32)
            low=low*mask
            high=high*mask
            low_data[:,:,i]=low
            high_data[:,:,i]=high
        low_pro=(low_data-Mean)/var
        high_pro=(high_data-Mean)/var
        high_pro=low_pro-high_pro
        _crop(low_pro,high_pro,writer)
    writer.close()
    return

def mk_tensor_from_tfrecords(records_path):
    queue = tf.train.string_input_producer([records_path])
    reader = tf.TFRecordReader()
    _,serializedexample = reader.read(queue)
    features = tf.parse_single_example(serializedexample,features={'low_pro':tf.FixedLenFeature([],tf.string),
                                                                   'high_pro':tf.FixedLenFeature([],tf.string)})
    low_pro_patch = tf.decode_raw(features['low_pro'],tf.float32)
    high_pro_patch = tf.decode_raw(features['high_pro'],tf.float32)
    low_pro_patch = tf.reshape(low_pro_patch,[CropHeight,CropWidth,CropDepth,1])
    high_pro_patch = tf.reshape(high_pro_patch,[CropHeight,CropWidth,CropDepth,1])
    low_pro_batch,high_pro_batch = tf.train.shuffle_batch([low_pro_patch,high_pro_patch],
                                                          batch_size=BatchSize,
                                                          capacity=10000,
                                                          min_after_dequeue=3000+3*BatchSize,
                                                          num_threads=4,
                                                          allow_smaller_final_batch=True)
    return low_pro_batch,high_pro_batch


def GetTrainData_2d(train_path, BS):
    img_batch = []
    ctp_batch = []
    for i in range(1, BS+1):
        if (i == 1):
            img_batch, ctp_batch = GetTrainBatch_2d(train_path)
        else:
            img_batch_temp, ctp_batch_temp = GetTrainBatch_2d(train_path)
            img_batch = np.concatenate((img_batch, img_batch_temp), axis= 0)
            ctp_batch = np.concatenate((ctp_batch, ctp_batch_temp), axis= 0)
    return  img_batch, ctp_batch


def GetTrainBatch_2d(train_path):
    train_dirs = os.listdir(train_path + 'data/')
    samples = random.choice(train_dirs)

    img_batch = np.load(train_path + 'data/' + samples)
    img_batch = np.transpose(img_batch, (1,2,0))
    ctp_batch = np.load(train_path + 'label/' + 'cbf/' + samples)
    img_batch = np.expand_dims(img_batch, axis= 0)
    ctp_batch = np.expand_dims(ctp_batch, axis= 0)
    ctp_batch = np.expand_dims(ctp_batch, axis= 3)
    return  img_batch, ctp_batch

def GetTestData_2d(testPath, tDir, batchsize):
    img_batch = []
    ctp_batch = []
    for i in range(1, batchsize+1):
        if i == 1:
            img_batch, ctp_batch = GetTestBatch_2d(testPath, tDir, i-1)
        else:
            img_batch_tmp, ctp_batch_tmp = GetTestBatch_2d(testPath, tDir, i-1)
            img_batch = np.concatenate((img_batch, img_batch_tmp), axis = 0)
            ctp_batch = np.concatenate((ctp_batch, ctp_batch_tmp), axis = 0)
    return img_batch, ctp_batch  # vol_shape [BWHC],[n,128,128,x]

def GetTestBatch_2d(testPath, tDir, ind):
    img_batch = np.load(testPath + 'data/' + tDir[ind])
    img_batch = np.transpose(img_batch, (1, 2, 0))
    ctp_batch = np.load(testPath + 'label/' + 'cbf/' +tDir[ind])
    img_batch = np.expand_dims(img_batch, axis = 0)
    ctp_batch = np.expand_dims(ctp_batch, axis = 0)
    ctp_batch = np.expand_dims(ctp_batch, axis = 3)
    return img_batch, ctp_batch

def SaveNpys(resultPath, name_pre, label_batch, pred_batch):
    np.save(resultPath + 'npys/' + name_pre + '-mask.npy', label_batch)
    np.save(resultPath + 'npys/' + name_pre + '-pred.npy', pred_batch)