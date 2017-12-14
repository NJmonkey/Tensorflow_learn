#-*- coding: utf-8 -*-
'''
@Time    : 2017/12/3 0003 14:58
@Author  : xiaohe
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#载入数据集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#定义批次大小
batch_size = 50  #每一次以矩阵的形式放入100张图片
#计算有多少批次
n_batch = mnist.train.num_examples // batch_size

#初始化权值
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1) #生成截断的正态分布
    return tf.Variable(initial)
#初始化偏置
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
#卷积层
def conv2d_nn(x, W):
    # x input tensor of shape [batch, in_height, in_width, in_channels]
    # W filter / kernel tensor of shape [filter_height, filter_width, in_channels, out_channels]
    # strides[0]=strides[3]=1, strides[1]表示x轴方向的步长，strides[2]表示y轴方向的步长
    # padding 表示卷积的方式，VALID表示卷积之后图案较小，SAME表示卷积之后大小不变，此时卷积核超过图片的部分补0
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#池化层
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

#定义两个占位符
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

#把x的格式转化成4D的向量[batch, in_height, in_width, in_channels]
#因为一开始传入的图片是一维的1*784
x_image = tf.reshape(x, [-1, 28, 28, 1])

#初始化第一个卷积层的权值和偏置
W_conv1 = weight_variable([5, 5, 1, 32]) #卷积核大小5*5，32个卷积核，1个采样通道
b_conv1 = bias_variable([32]) #每一个卷积核共享一个偏置

#把x_image和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
h_conv1 = tf.nn.relu(conv2d_nn(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1) #选出最大的

#初始化第二层卷积层
W_conv2 = weight_variable([5, 5, 32, 64]) #从上一层得到的32个平面图中进行卷积提取特征
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d_nn(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2) #选出最大的

#28*28的图片第一次卷积之后还是28*28，第一I次池化后为14*14
#第二次卷积轴为14*14，第二次池化之后为7*7
#经过上面的操作得到64张7*7的平面图

#初始化第一个全连接层的权值
W_fc1 = weight_variable([7*7*64, 1024]) #输入的神经元为7*7*64，下一层的全连接有1024个神经元
b_fc1 = bias_variable([1024]) #全连接层对应1024个偏置

#把第二个池化层的输入扁平化成1维
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
#求第一个全连接层的输出
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#keep_prob表示神经元输出的概率
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#初始化第二个全连接层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

#计算输出
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#交叉熵损失函数
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
#使用AdamOptimizer进行优化
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#结果保存在一个布尔列表中
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
#计算准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for epoch in range(21):
    for batch in range(n_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.7})
                                     #这里不能用测试集所有的数据，不然显卡内存不够会报错，我们取1000进行测试
    acc = sess.run(accuracy, feed_dict={x: mnist.test.images[:1000],
                                        y: mnist.test.labels[:1000], keep_prob: 1.0})
    print('iter:' + str(epoch) + ', Test Accuracy:' + str(acc))
sess.close()
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for epoch in range(21):
#         for batch in range(n_batch):
#             batch_xs, batch_ys = mnist.train.next_batch(batch_size)
#             sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.7})
#         acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
#         print('iter:' + str(epoch) + ', Test Accuracy:' + str(acc))
# a = 10
# b = 50
# sum = 0
# for i in range(a):
#     testSet = mnist.test.next_batch(b)
#     c = accuracy.eval(feed_dict={x: testSet[0], y: testSet[1], keep_prob: 1.0})
#     sum += c * b
#     #print("test accuracy %g" %  c)
# print("test accuracy %g" %  (sum / (b * a)))











