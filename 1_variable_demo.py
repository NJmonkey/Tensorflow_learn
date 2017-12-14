#-*- coding: utf-8 -*-
'''
@Time    : 2017/11/30 0030 20:20
@Author  : xiaohe
'''
import tensorflow as tf

state = tf.Variable(0, name='counter') #定义变量op
one = tf.constant(1)                   #定义常量op
new_value = tf.add(state, one)         #加法运算op
update = tf.assign(state, new_value)   #赋值op


# init = tf.initialize_all_variables() #会有警告，现在已经更新成global_variables_initializer()
init = tf.global_variables_initializer() #有变量时就要运行这句程序

with tf.Session() as sess:
    sess.run(init)    #在会话里run一下才能执行
    for _ in range(3):
        sess.run(update)
        # print(sess.run(new_value))
        print(sess.run(state))






