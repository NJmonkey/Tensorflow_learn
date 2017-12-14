#-*- coding: utf-8 -*-
'''
@Time    : 2017/12/2 0002 20:03
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

#参数概要
#这个函数的作用就是可以计算传入值的平均值，最大值和最小值，最后画出直方图
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean) #平均值，第一个参数相当于起个名字
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev', stddev) #标准差
        tf.summary.scalar('max', tf.reduce_max(var)) #这里其实都是简写，上面的平均值分两步写的
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var) #直方图

#命名空间
with tf.name_scope('input'):
    #定义两个占位符
    x = tf.placeholder(tf.float32, [None, 784], name='x_input')
    y = tf.placeholder(tf.float32, [None, 10], name='y_input')

#定义神经网络  没有隐藏层 输入层过来就是输出层，输出层神经元有10个
#把每一个变量都起个名字
with tf.name_scope('layer'):
    with tf.name_scope('weights'):
        W = tf.Variable(tf.zeros([784, 10]))
        #这里就可以用到上面定义的函数，对权值进行观测,可以看到W变化过程中的最大值，最小值，标准差等
        variable_summaries(W)
    with tf.name_scope('biases'):
        b = tf.Variable(tf.zeros([10]))
        variable_summaries(b)
    with tf.name_scope('wx_plus_b'):
        Wx_plus_b = tf.matmul(x, W) + b
    with tf.name_scope('softmax'):
        predicts = tf.nn.softmax(Wx_plus_b)

#平方差损失函数
# loss = tf.reduce_mean(tf.square(y - predicts))

#交叉熵损失函数
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=predicts))
    #观测loss
    tf.summary.scalar('loss', loss)
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
        tf.summary.scalar('accuracy', accuracy)

#合并所有的summary
merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('F:/Tensorboard_logs/', sess.graph)
    for epcho in range(21):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y: batch_ys})
        writer.add_summary(summary, epcho)
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print('iter:' + str(epcho) + ', Test Accuracy:' + str(acc))

