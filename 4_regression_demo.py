#-*- coding: utf-8 -*-
'''
@Time    : 2017/11/30 0030 21:41
@Author  : xiaohe
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#使用numpy生成200个随机点
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
noise = np.random.normal(0, 0.02, x_data.shape) #形状和x_data一样
y_data = np.square(x_data) + noise   #是一个二次函数

#定义两个占位符
x = tf.placeholder(tf.float32, [None, 1]) #形状None表示行数可以为任意值，列为1列
#这里的None要根据最后传入的值的形状确定，最终我们是把x_data传给x，所以这里的None和x_data的行数相同，为200
y = tf.placeholder(tf.float32, [None, 1])

#定义网络中间层
Weights_L1 = tf.Variable(tf.random_normal([1, 10])) #L1层的权值随机初始化，形状为1*10
biases_L1 = tf.Variable(tf.zeros([1, 10])) #偏置层与权值对应
Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + biases_L1
L1 = tf.nn.tanh(Wx_plus_b_L1) #定义激活函数为tanh 激活函数处理后作为L1层的输出

#定义输出层
Weights_L2 = tf.Variable(tf.random_normal([10, 1]))
biases_L2 = tf.Variable(tf.zeros([1, 1]))
Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + biases_L2
predicts = tf.nn.tanh(Wx_plus_b_L2)

#平方差损失函数
loss = tf.reduce_mean(tf.square(y - predicts))
#使用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(2000):
        sess.run(train_step, feed_dict={x: x_data, y: y_data})
    #获得预测值
    prediction_values = sess.run(predicts, feed_dict={x: x_data})

    plt.figure()
    plt.scatter(x_data, y_data) #散点为样本点
    plt.plot(x_data, prediction_values, 'r-', lw=5) #线宽为5
    plt.show()
