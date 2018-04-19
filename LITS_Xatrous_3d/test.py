import  os
# IMGDEPTH = 24
# testPath = 'E:/Lianxin_LITS/lxData_rs_600_cut_280/test_npy/'
#
# list1 = ['11', 10]
# print(type(list1))
#
# test_dirs = os.listdir(testPath + '/vol/')
#
# if not(os.path.exists(testPath + '/vol/')):
#     print("ol")
#
# for tDir in test_dirs:
#     name_pre = tDir[:-4]
#     file_split = name_pre.split('-')
#     a = int(file_split[-1]) * IMGDEPTH
#     print(type(file_split))





import numpy as np
import tensorflow as tf

nparr = np.array([[1,2],[3,4]])


a = tf.constant([[[0.8, 0.1, 0.1], [0.2, 0.2, 0.3]],[[0.4, 0.4, 0.5], [1.3, 1.3, 0.7]]]) #2x2x3
#a = tf.constant([[[0, 0, 1], [0, 0, 1]],[[1, 0, 1], [0, 1, 1]]])
#a = tf.constant([1,1])
ashape = (a.shape)
sss = a.get_shape()
b = a[...,-1]
arg = tf.argmax(a, axis = 2)
weights = tf.get_variable('w', shape=[1, 1, 1, 1, 2],
                              initializer=tf.contrib.layers.xavier_initializer())
weights2 = tf.get_variable('w2', shape=[1, 1, 1, 1, 3],
                              initializer=tf.contrib.layers.xavier_initializer())
tvars = tf.trainable_variables()

classNum = tf.constant(2)
in_shape = tf.shape(a)
aaa= tf.reshape(classNum, (1,))
out_shape = tf.concat([in_shape, aaa], 0)

a = tf.reshape(a, (-1,))
dense_shape = tf.stack([tf.shape(a)[0], classNum], 0)
a = tf.to_int64(a)
ids = tf.range(tf.to_int64(dense_shape[0]), dtype= tf.int64)
ids = tf.stack([ids, a], axis= 1)
one_hot = tf.SparseTensor(indices= ids, values= tf.ones_like(a, dtype= tf.float32), dense_shape= tf.to_int64(dense_shape))
one_hot = tf.sparse_reshape(one_hot, out_shape)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
f = sess.run(aaa)
g= sess.run(in_shape)
h = sess.run(out_shape)
i = sess.run(one_hot)
print(ashape)

print(sess.run(b))
print(i)
print('ok')

