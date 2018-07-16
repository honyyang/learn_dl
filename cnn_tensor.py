#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf



# 神经网络类
class Network(object):
    def __init__(self,
            conv1_filter_number=6, 
            conv2_filter_number=16, 
            hide_dense_units=100):
        '''
        构造函数
        '''
        self.this_graph = tf.Graph()
        self.this_session = tf.Session(graph=self.this_graph)
        with self.this_graph.as_default():
            self.input_tensor = tf.placeholder(tf.float32, [None, 784])
            self.label_tensor = tf.placeholder(tf.float32, [None, 10])
            input_image = tf.reshape(self.input_tensor, [-1, 28, 28, 1])
            # Convolutional Layer #1
            conv1 = tf.layers.conv2d(
                  inputs=input_image,
                  filters=conv1_filter_number,
                  kernel_size=[5, 5],
                  padding="same",
                  activation=tf.nn.relu)
            # Pooling Layer #1
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
            # Convolutional Layer #2 and Pooling Layer #2
            conv2 = tf.layers.conv2d(
                    inputs=pool1,
                    filters=conv2_filter_number,
                    kernel_size=[5, 5],
                    padding="same",
                    activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
            # Dense Layer
            pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * conv2_filter_number])
            dense = tf.layers.dense(inputs=pool2_flat, units=hide_dense_units, activation=tf.sigmoid)
            # Logits Layer
            output_tensor = tf.layers.dense(inputs=dense, units=10, activation=tf.sigmoid)
            self.predict_tensor = output_tensor
            self.loss = tf.losses.mean_squared_error(labels=self.label_tensor, 
                    predictions=self.predict_tensor)
            self.learning_rate = tf.placeholder(tf.float32, shape=())
            self.train = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
            with self.this_session.as_default():
                tf.global_variables_initializer().run()


    def predict(self, sample):
        '''
        使用神经网络实现预测
        sample: 输入样本
        '''
        with self.this_session.as_default():
            output = self.predict_tensor.eval({self.input_tensor: [sample]})
        return output

    def training(self, labels, data_set, rate, epoch):
        '''
        训练函数
        labels: 样本标签
        data_set: 输入样本
        rate: 学习速率
        epoch: 训练轮数
        '''
        for i in range(epoch):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d], 
                    data_set[d], rate)

    def train_one_sample(self, label, sample, rate):
        with self.this_session.as_default():
            self.train.run({self.label_tensor: [label], 
                self.input_tensor: [sample], self.learning_rate: rate})

    def calc_gradient(self, label):
        pass

    def update_weight(self, rate):
        pass

    def dump(self):
        pass

    def calc_loss(self, label, output):
        label_tensor = tf.convert_to_tensor(label, tf.float32)
        output_tensor = tf.convert_to_tensor(output, tf.float32)
        loss = tf.losses.mean_squared_error(labels=label_tensor, predictions=output_tensor)
        with tf.Session():
            return loss.eval()

    def gradient_check(self, sample_feature, sample_label):
        pass
        '''
        梯度检查
        network: 神经网络对象
        sample_feature: 样本的特征
        sample_label: 样本的标签

        # 获取网络在当前样本下每个连接的梯度
        self.predict(sample_feature)
        self.calc_gradient(sample_label)

        # 检查梯度
        epsilon = 10e-4
        for fc in self.layers:
            for i in range(fc.W.shape[0]):
                for j in range(fc.W.shape[1]):
                    fc.W[i,j] += epsilon
                    output = self.predict(sample_feature)
                    err1 = self.loss(sample_label, output)
                    fc.W[i,j] -= 2*epsilon
                    output = self.predict(sample_feature)
                    err2 = self.loss(sample_label, output)
                    expect_grad = (err1 - err2) / (2 * epsilon)
                    fc.W[i,j] += epsilon
                    print 'weights(%d,%d): expected - actural %.4e - %.4e' % (
                        i, j, expect_grad, fc.W_grad[i,j])
        '''


from input_data import get_training_data_set

def train_data_set():
    data_set, labels =  get_training_data_set()
    return labels, data_set

def test():
    labels, data_set = train_data_set()
    net = Network(6, 16, 100)
    rate = 0.3
    mini_batch = 100
    epoch = 10
    for i in range(epoch):
        net.training(labels, data_set, rate, mini_batch)
        print(np.around(net.predict(data_set[-1]),decimals=3).reshape(10))
        rate /= 2


def gradient_check():
    '''
    梯度检查
    '''
    labels, data_set = train_data_set()
    net = Network(8, 3, 8)
    net.gradient_check(data_set[0], labels[0])
    return net

if __name__ == '__main__':
    test()
