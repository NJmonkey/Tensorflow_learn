#-*- coding: utf-8 -*-
'''
@Time    : 2017/12/1 0001 16:17
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

#定义神经网络  没有隐藏层 输入层过来就是输出层，输出层神经元有10个
w = tf.Variable(tf.zeros([784, 10])) #输入层
b = tf.Variable(tf.zeros([10]))      #输出层
predicts = tf.nn.softmax(tf.matmul(x, w) + b) #predicts得到的是每个数字的概率

#平方差损失函数
# loss = tf.reduce_mean(tf.square(y - predicts))

#交叉熵损失函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=predicts))

#采用梯度下降法
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
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print('iter:' + str(epcho) + ', Test Accuracy:' + str(acc))