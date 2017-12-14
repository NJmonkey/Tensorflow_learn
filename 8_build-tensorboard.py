#-*- coding: utf-8 -*-
'''
@Time    : 2017/12/2 0002 15:05
@Author  : xiaohe
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#载入数据集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#定义批次大小
batch_size = 100  #每一次以矩阵的形式放入100张图片
#计算有多少批次
n_batch = mnist.train.num_examples // batch_size

#命名空间
with tf.name_scope('input'):
    #定义两个占位符
    x = tf.placeholder(tf.float32, [None, 784], name='x_input')
    y = tf.placeholder(tf.float32, [None, 10], name='y_input')

#定义神经网络  没有隐藏层 输入层过来就是输出层，输出层神经元有10个
#把每一个变量都起个名字
with tf.name_scope('layer'):
    with tf.name_scope('weights'):
        w = tf.Variable(tf.zeros([784, 10]))
    with tf.name_scope('biases'):
        b = tf.Variable(tf.zeros([10]))
    with tf.name_scope('wx_plus_b'):
        wx_plus_b = tf.matmul(x, w) + b
    with tf.name_scope('softmax'):
        predicts = tf.nn.softmax(wx_plus_b)

#平方差损失函数
# loss = tf.reduce_mean(tf.square(y - predicts))

#交叉熵损失函数
#起名字 进行可视化
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=predicts))

#采用梯度下降法
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

#初始化变量
init = tf.global_variables_initializer()

with tf.name_scope('accuracy_calculate'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(predicts, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    #http://localhost:6006/
    #这里需要注意的是，生成文件的文件夹中不能出现空格，不然浏览器会找不到文件
    writer = tf.summary.FileWriter('F:/Tensorboard_logs/', sess.graph)
    for epcho in range(1):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print('iter:' + str(epcho) + ', Test Accuracy:' + str(acc))

#可视化神经网络，最重要的就是给网络的每一个部分起名字，最后运行生成一个events.out.tfevents文件，在文件所在目录
#打开dos窗口，进入py3，输入tensorboard --logdir=文件所在路径，再进入谷歌浏览器输入网址http://localhost:6006/