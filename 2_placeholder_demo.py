#-*- coding: utf-8 -*-
'''
@Time    : 2017/11/30 0030 20:58
@Author  : xiaohe
'''
import tensorflow as tf

#占位符，传入值

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)
with tf.Session() as sess:
    #feed_dict的数据以字典的形式传入
    print(sess.run(output, feed_dict={input1:[7.], input2:[2.]}))

