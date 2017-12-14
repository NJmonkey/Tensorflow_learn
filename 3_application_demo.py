#-*- coding: utf-8 -*-
'''
@Time    : 2017/11/30 0030 21:03
@Author  : xiaohe
'''
import tensorflow as tf
import numpy as np

#使用numpy生成100个随机点
x_data = np.random.rand(100)
y_data = x_data * 0.1 + 0.2

#构造一个线性模型
b = tf.Variable(0.) #初始值为0,可以随意定义
k = tf.Variable(0.)
y = x_data * k + b

#平方损失函数
loss = tf.reduce_mean(tf.square(y_data - y))
#定义一个梯度下降法优化器进行训练
optimizer = tf.train.GradientDescentOptimizer(0.2)
#最小化损失函数
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for step in range(201):
        sess.run(train)  #每一次都要run一下train，这样每一次都会最小化一下loss
        if step % 20 == 0:
            print(step, sess.run([k, b]))



























