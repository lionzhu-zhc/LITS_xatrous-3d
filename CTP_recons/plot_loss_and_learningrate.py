import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os
import utils
import resNet3d



LAST_EPOCH = 0
print('prepare the data')
tfrecords_path = './trainingdata/train.tfrecords'

patient_low='E:\PycharmProjects\proj\\trainingdata_FBP_pro\\'
patient_high='E:\PycharmProjects\proj\\trainingdata_FBP_high\\'
train_patient = ['L067','L096','L109','L143','L192','L286','L291','L310','L333']
test_patient = ['L506']

patient_low_list=[os.path.join(patient_low,var) for var in train_patient]
patient_high_list = [os.path.join(patient_high,var) for var in train_patient]

if not os.path.exists(tfrecords_path):
    utils.make_tf_records(patient_low_list,patient_high_list,tfrecords_path)

TotalNum=utils.Height*utils.Width*utils.Depth/utils.CropHeight/utils.CropWidth/utils.CropDepth*9

EpochBatch=TotalNum//utils.BatchSize

print('prepare the data finish')

print('construct resNetPro3d network')

# construct the network
low_pro = tf.placeholder(tf.float32, [None,None,None,None,1])
high_pro = tf.placeholder(tf.float32, [None,None,None,None,1])
train_placeholder = tf.placeholder(tf.bool, shape=[])

noise_pro = resNet3d.resnet3d(low_pro, train_placeholder)
recon_pro = low_pro[:,12:-12,12:-12,6:-6,:]-noise_pro
reference_pro = low_pro[:,12:-12,12:-12,6:-6,:]-high_pro[:,12:-12,12:-12,6:-6,:]

loss = tf.reduce_mean(tf.square(noise_pro-high_pro[:,12:-12,12:-12,6:-6,:]))+\
    tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

learning_rate = tf.placeholder(tf.float32,shape=[])




train = tf.group(tf.train.AdamOptimizer(learning_rate).minimize(loss),*update_op)
init_global = tf.global_variables_initializer()
sess = tf.Session()
Saver = tf.train.Saver()

# make up the input pipeline
low_pro_batch,high_pro_batch = utils.mk_tensor_from_tfrecords(tfrecords_path)
sess.run(init_global)
sess.run(tf.local_variables_initializer())



print('construct network finish')

print('begin train')
# begin input pipeline queue
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess,coord=coord)
learning_rate_list=[1e-6*var for var in range(1,1001,2)]
count_for_summary=0
loss_batch=[]
for epoch in range(500):
    LearnRate = learning_rate_list[epoch]
    batch = 0
    low_pro_batch_,high_pro_batch_=sess.run([low_pro_batch,high_pro_batch])
    sess.run(train,feed_dict={low_pro:low_pro_batch_,high_pro:high_pro_batch_,train_placeholder:True,
                              learning_rate:LearnRate})
    loss_value= sess.run(loss,feed_dict={low_pro:low_pro_batch_,
                                                                          high_pro:high_pro_batch_,
                                                                          train_placeholder:True,
                                                                          learning_rate:LearnRate
                                                                          })
    loss_batch.append(loss_value)

coord.request_stop()
coord.join()
plt.figure()
plt.plot([1e-5*var for var in range(1,1001,2)],loss_batch)
plt.show()






