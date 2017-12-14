#-*- coding: utf-8 -*-
'''
@Time    : 2017/12/2 0002 20:53
@Author  : xiaohe
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector

#载入数据集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
#运行次数
max_steps = 1001
#图片数量
image_num = 3000
#文件路径
DIR = 'F:\Tensorflow/'

sess = tf.Session()

#载入图片 保存在embedding中 可使用Ctrl+鼠标点击查看stack方法的作用
embedding = tf.Variable(tf.stack(mnist.test.images[:image_num]), trainable=False, name='embedding')

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

#显示图片
with tf.name_scope('input_reshape'):
                                    #第一个-1表示维度不确定，因为前面传入的是None
                                    #最后一个1表示是黑白图片，信道为1，彩色图片为3
    image_shape_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shape_input, 10)

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
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(predicts, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

#产生matadata文件
#首先检查有没有这样一个文件，有的话就删除原文件
if tf.gfile.Exists(DIR + 'projector/projector/metadata.tsv'):
    tf.gfile.DeleteRecursively(DIR + 'projector/projector/metadata.tsv')
#如果没有这个文件，就写入一个这个文件，即生成matadata文件
with open(DIR + 'projector/projector/metadata.tsv', 'w') as f:
    labels = sess.run(tf.argmax(mnist.test.labels[:], 1)) #拿到测试集所有的标签
    for i in range(image_num):
        #把拿到的图片的labels写入到matadata文件中，这里取3000张图片
        f.write(str(labels[i]) + '\n')

#合并所有的summary
merged = tf.summary.merge_all()

projector_writer = tf.summary.FileWriter(DIR + 'projector/projector', sess.graph)
saver = tf.train.Saver()  #创建一个 Saver 来管理模型中的所有变量
config = projector.ProjectorConfig()
embed = config.embeddings.add()
embed.tensor_name = embedding.name
embed.metadata_path = DIR + 'projector/projector/metadata.tsv'
embed.sprite.image_path = DIR + 'projector/data/mnist_10k_sprite.png'
embed.sprite.single_image_dim.extend([28, 28]) #把大的图片中的每个数字切分出来
projector.visualize_embeddings(projector_writer, config)

for i in range(max_steps):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    #下面是固定的配置
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    #把配置好的数据放入到run()函数中
    summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y: batch_ys},
                          options=run_options, run_metadata=run_metadata)
    projector_writer.add_run_metadata(run_metadata, 'step%03d' % i)
    projector_writer.add_summary(summary, i)
    if i % 100 == 0:
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print('iter:' + str(i) + ', Test Accuracy:' + str(acc))
saver.save(sess, DIR + 'projector/projector/a_model.ckpt', global_step=max_steps)
projector_writer.close()
sess.close()


