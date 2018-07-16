#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf



# 神经网络类
class Network(object):
    def __init__(self, layers):
        '''
        构造函数
        '''
        self.this_graph = tf.Graph()
        self.this_session = tf.Session(graph=self.this_graph)
        with self.this_graph.as_default():
            self.input_tensor = tf.placeholder(tf.float32, [None, layers[0]])
            output_tensor = self.input_tensor
            for i in range(len(layers) - 1):
                output_tensor = tf.layers.dense(output_tensor, layers[i+1],
                        activation=tf.sigmoid)
            self.predict_tensor = output_tensor
            self.label_tensor = tf.placeholder(tf.float32, [None, layers[len(layers)-1]])
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
            output = self.predict_tensor.eval({self.input_tensor: np.array([sample], dtype=np.float32)})
        return output[0]

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


from bp import train_data_set


def transpose(args):
    return map(
        lambda arg: map(
            lambda line: np.array(line).reshape(len(line), 1)
            , arg)
        , args
    )


class Normalizer(object):
    def __init__(self):
        self.mask = [
            0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80
        ]

    def norm(self, number):
        return map(lambda m: 0.9 if number & m else 0.1, self.mask)

    def denorm(self, vec):
        binary = map(lambda i: 1 if i > 0.5 else 0, vec)
        for i in range(len(self.mask)):
            binary[i] = binary[i] * self.mask[i]
        return reduce(lambda x,y: x + y, binary)

def train_data_set():
    normalizer = Normalizer()
    data_set = []
    labels = []
    for i in range(0, 256):
        n = normalizer.norm(i)
        data_set.append(n)
        labels.append(n)
    return labels, data_set

def correct_ratio(network):
    normalizer = Normalizer()
    correct = 0.0;
    for i in range(256):
        if normalizer.denorm(network.predict(normalizer.norm(i))) == i:
            correct += 1.0
    print('correct_ratio: %.2f%%' % (correct / 256 * 100))


def test():
    labels, data_set = train_data_set()
    net = Network([8, 8, 8])
    rate = 0.1
    mini_batch = 200
    epoch = 10
    for i in range(epoch):
        net.training(labels, data_set, rate, mini_batch)
        print('after epoch %d loss: %f' % (
            (i + 1),
            net.calc_loss(labels[-1], net.predict(data_set[-1]))
        ))
        rate /= 2
    correct_ratio(net)


def gradient_check():
    '''
    梯度检查
    '''
    labels, data_set = transpose(train_data_set())
    net = Network([8, 3, 8])
    net.gradient_check(data_set[0], labels[0])
    return net

if __name__ == '__main__':
    test()
