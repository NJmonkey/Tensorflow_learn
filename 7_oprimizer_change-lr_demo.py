#-*- coding: utf-8 -*-
'''
@Time    : 2017/12/2 0002 10:50
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

#定义两个占位符
x = tf.placeholder(tf.float32, [None, 784]) #每次传入100批数据，最后这个None就是100
y = tf.placeholder(tf.float32, [None, 10])

lr = tf.Variable(0.001, dtype=tf.float32)

#定义神经网络  没有隐藏层 输入层过来就是输出层，输出层神经元有10个
w = tf.Variable(tf.zeros([784, 10])) #输入层
b = tf.Variable(tf.zeros([10]))      #输出层
predicts = tf.nn.softmax(tf.matmul(x, w) + b) #predicts得到的是每个数字的概率

#平方差损失函数
# loss = tf.reduce_mean(tf.square(y - predicts))

#交叉熵损失函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=predicts))

#采用梯度下降法进行优化
# train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
#改变优化器 这里可以用许多种优化器，任意一种，会使用就行
train_step = tf.train.AdamOptimizer(lr).minimize(loss)


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
        #这样定义学习率保证了刚开始的时候学习率比较大，随着训练次数增加，学习率越来越小
        sess.run(tf.assign(lr, 0.001*(0.95**epcho)))

        for batch in range(n_batch): #一共要循环n_batch次，循环完之后代表所有图片都训练了一遍

            learning_rate = sess.run(lr)
            #xs用于保存数据，ys用于保存标签
            batch_xs, batch_ys = mnist.train.next_batch(batch_size) #每次转入100张数字
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print('iter:' + str(epcho) + ', Test Accuracy:' + str(acc) +
              ', Learning Rate:' + str(learning_rate))