#-*- coding: utf-8 -*-
'''
@Time    : 2017/12/1 0001 21:05
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

#定义三个占位符
x = tf.placeholder(tf.float32, [None, 784]) #每次传入100批数据，最后这个None就是100
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)  #这个变量就代表对网络进行dropout，保留多少的神经元

#定义神经网络
#输入层 输入神经元784个
W1 = tf.Variable(tf.truncated_normal([784, 500], stddev=0.1))
b1 = tf.Variable(tf.zeros([500]) + 0.1)
L1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
L1_drop = tf.nn.dropout(L1, keep_prob)  #添加dropout
#第一层隐层
W2 = tf.Variable(tf.truncated_normal([500, 300], stddev=0.1))
b2 = tf.Variable(tf.zeros([300]) + 0.1)
L2 = tf.nn.tanh(tf.matmul(L1_drop, W2) + b2)
L2_drop = tf.nn.dropout(L2, keep_prob)  #添加dropout
#第二层隐层
W3 = tf.Variable(tf.truncated_normal([300, 100], stddev=0.1))
b3 = tf.Variable(tf.zeros([100]) + 0.1)
L3 = tf.nn.tanh(tf.matmul(L2_drop, W3) + b3)
L3_drop = tf.nn.dropout(L3, keep_prob)  #添加dropout
#输出层  输出神经元10个
W4 = tf.Variable(tf.truncated_normal([100, 10], stddev=0.1))
b4 = tf.Variable(tf.zeros([10]) + 0.1)
predicts = tf.nn.softmax(tf.matmul(L3_drop, W4) + b4)  #最后一层为softmax分类器


#平方差损失函数
# loss = tf.reduce_mean(tf.square(y - predicts))

#交叉熵损失函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=predicts))

#采用梯度下降法优化
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

#初始化变量
init = tf.global_variables_initializer()

#tf.equal函数是比较两个数是否相等，相等返回True，否则返回False，所以最终correct_prediction里面存放的都是
#一堆True或者False
correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(predicts, 1))
                            #这个就是把真实的标签和预测的标签进行比较
                            #tf.cast函数的作用就是把布尔类型变量转换成其他类型，这里转换成float32
                            #转换之后就会把True变成1.0，把False变成0.0
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epcho in range(21):
        for batch in range(n_batch): #一共要循环n_batch次，循环完之后代表所有图片都训练了一遍
            #xs用于保存数据，ys用于保存标签
            batch_xs, batch_ys = mnist.train.next_batch(batch_size) #每次转入100张数字
                                                    #根据keep_prob的不同取值，决定每层保留多少神经元
                                                    #1.0代表完全保留，不进行drop；0.5代表保留一半
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.7})
        #测试数据所得到的准确率
        test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images,
                                                 y: mnist.test.labels, keep_prob: 0.7})
        #训练数据所得到的准确率
        train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images,
                                                 y: mnist.train.labels, keep_prob: 0.7})

        print('iter:' + str(epcho) + ', Test Accuracy:' + str(test_acc) + ', Train Accuracy:'
              + str(train_acc))