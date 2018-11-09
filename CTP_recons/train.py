# -*- encoding : utf-8 -*-
import tensorflow as tf
import resNetPro
import os
import utils



LAST_EPOCH = 0
print('prepare the data')
tfrecords_path = './trainingdata/train.tfrecords'

patient_low='E:\PycharmProjects\proj\\trainingdata_FBP_pro\\'
patient_high='E:\PycharmProjects\proj\\trainingdata_TV_high\\'
train_patient = ['L067','L096','L109','L143','L192','L286','L291','L310','L333']
test_patient = ['L506']

patient_low_list=[os.path.join(patient_low,var) for var in train_patient]
patient_high_list = [os.path.join(patient_high,var) for var in train_patient]

if not os.path.exists(tfrecords_path):
    utils.make_tf_records(patient_low_list,patient_high_list,tfrecords_path)

TotalNum=utils.Height*utils.Width*utils.Depth/(utils.CropHeight-6)/(utils.CropWidth-6)/(utils.CropDepth-6)*9

EpochBatch=TotalNum//utils.BatchSize

print('prepare the data finish')

print('construct resNetPro3d network')

# construct the network
low_pro = tf.placeholder(tf.float32, [None,None,None,None,1])
high_pro = tf.placeholder(tf.float32, [None,None,None,None,1])
train_placeholder = tf.placeholder(tf.bool, shape=[])

noise_pro = resNetPro.resnet3d(low_pro, train_placeholder)
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

# make up the records
Writer = tf.summary.FileWriter(logdir='./logdir')
train_loss_scar = tf.summary.scalar('train_loss',loss)
low_pro_img = tf.summary.image('low_pro',tf.reshape(low_pro[0:3,12:-12,12:-12,6,:],[3,20,20,1]))
high_pro_img = tf.summary.image('high_pro',tf.reshape(reference_pro[0:3,:,:,6,:],[3,20,20,1]))
recon_pro_img = tf.summary.image('recon_pro',tf.reshape(recon_pro[0:3,:,:,6,:],[3,20,20,1]))
learning_rate_scar = tf.summary.scalar('learning_rate',learning_rate)
train_summary = tf.summary.merge([train_loss_scar,low_pro_img,high_pro_img,recon_pro_img,learning_rate_scar])

# make up the input pipeline
low_pro_batch,high_pro_batch = utils.mk_tensor_from_tfrecords(tfrecords_path)
sess.run(init_global)
sess.run(tf.local_variables_initializer())

if LAST_EPOCH != 0:
    Saver.restore(sess,'./model/-{0}'.format(LAST_EPOCH))

print('construct network finish')

print('begin train')
# begin input pipeline queue
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess,coord=coord)
init_learning_rate=1e-3
count_for_summary=0

for epoch in range(50):
    LearnRate = init_learning_rate*(0.5**(epoch//3))
    batch = 0
    while batch < EpochBatch:
        batch += 1
        low_pro_batch_,high_pro_batch_=sess.run([low_pro_batch,high_pro_batch])
        sess.run(train,feed_dict={low_pro:low_pro_batch_,high_pro:high_pro_batch_,train_placeholder:True,
                                  learning_rate:LearnRate})
        if batch % 10 == 0:
            loss_value,train_summary_ = sess.run([loss, train_summary],feed_dict={low_pro:low_pro_batch_,
                                                                                  high_pro:high_pro_batch_,
                                                                                  train_placeholder:True,
                                                                                  learning_rate:LearnRate
                                                                                  })
            Writer.add_summary(train_summary_,count_for_summary)
            count_for_summary += 1
            print('train epoch {0} batch {1} loss {2}'.format(epoch,batch,loss_value))
    Saver.save(sess,'./model/-{0}'.format(epoch))
coord.request_stop()
coord.join()
Writer.close()





