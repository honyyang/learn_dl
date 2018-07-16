#!/usr/bin/env python
# -*- coding: UTF-8 -*-


import random
import numpy as np
from activators import SigmoidActivator, IdentityActivator, ReluActivator
from fc_batch import FullConnectedLayer
from cnn_batch import ConvLayer, MaxPoolingLayer


# 神经网络类
class Network(object):
    def __init__(self,
            conv1_filter_number=6, 
            conv2_filter_number=16, 
            hide_dense_units=100):
        '''
        构造函数
        '''
        # Convolutional Layer #1
        self.conv1 = ConvLayer(input_width=28, input_height=28, 
                 channel_number=1, filter_width=5, 
                 filter_height=5, filter_number=conv1_filter_number, 
                 zero_padding=2, stride=1, activator=ReluActivator(),
                 learning_rate=0.001)
        # Pooling Layer #1
        self.pool1 = MaxPoolingLayer(input_width=28, input_height=28, 
                 channel_number=conv1_filter_number, filter_width=2, 
                 filter_height=2, stride=2)
        # Convolutional Layer #2 and Pooling Layer #2
        self.conv2 = ConvLayer(input_width=14, input_height=14, 
                 channel_number=conv1_filter_number, filter_width=5, 
                 filter_height=5, filter_number=conv2_filter_number, 
                 zero_padding=2, stride=1, activator=ReluActivator(),
                 learning_rate=0.001)
        self.pool2 = MaxPoolingLayer(input_width=14, input_height=14, 
                 channel_number=conv2_filter_number, filter_width=2, 
                 filter_height=2, stride=2)

        self.conv2_filter_number = conv2_filter_number
        # Dense Layer
        self.dense = FullConnectedLayer(input_size=7 * 7 * conv2_filter_number, 
                output_size=hide_dense_units, activator=SigmoidActivator())
        # Logits Layer
        self.logits = FullConnectedLayer(input_size=hide_dense_units, 
                output_size=10, activator=SigmoidActivator())

    def predict(self, sample):
        '''
        使用神经网络实现预测
        sample: 输入样本
        '''
        # List to conv input
        output = np.array(sample).reshape(-1, 1, 28, 28)
        output = self.conv1.forward(output)
        output = self.pool1.forward(output)
        output = self.conv2.forward(output)
        output = self.pool2.forward(output)
        # Flat to dense input
        output = output.reshape(-1, 7 * 7 * self.conv2_filter_number, 1)
        output = self.dense.forward(output)
        output = self.logits.forward(output)
        return output

    def training(self, labels, data_set, rate, batch, epoch):
        '''
        训练函数
        labels: 样本标签
        data_set: 输入样本
        rate: 学习速率
        epoch: 训练轮数
        '''
        gens = range(len(data_set))
        random.shuffle(gens)
        def next_batch(iters):
            batch_labels = []
            batch_data_set = []
            for i in range(batch):
                batch_labels.append(labels[gens[iters * batch + i]])
                batch_data_set.append(data_set[gens[iters * batch + i]])
            return batch_labels, batch_data_set
        iterations = int(len(data_set) / batch)
        for i in range(epoch):
            for iters in range(iterations):
                batch_labels, batch_data_set = next_batch(iters)
                self.train_one_sample(batch_labels, 
                    batch_data_set, rate)

    def train_one_sample(self, label, sample, rate):
        self.predict(sample)
        self.calc_gradient(label)
        self.update_weight(rate)

    def calc_gradient(self, label):
        label_array = np.array(label).reshape(self.logits.output.shape)
        delta = np.zeros(self.logits.output.shape)
        for k in range(len(delta)):
            delta[k] = -self.logits.activator.backward(
                self.logits.output[k]
            ) * (label_array[k] - self.logits.output[k])
        delta = self.logits.backward(delta, SigmoidActivator())
        delta = self.dense.backward(delta, ReluActivator())
        # Flat to conv input
        delta = delta.reshape(-1, self.conv2_filter_number, 7, 7)
        delta = self.pool2.backward(delta)
        delta = self.conv2.backward(delta)
        delta = self.pool1.backward(delta)
        delta = self.conv1.backward(delta)
        return delta

    def update_weight(self, rate):
        self.conv1.update(rate)
        self.conv2.update(rate)
        self.dense.update(rate)
        self.logits.update(rate)

    def dump(self):
        pass

    def loss(self, output, label):
        output_array = np.array(output).flatten()
        label_array = np.array(label).flatten()
        return 0.5 * ((label_array - output_array) * (label_array - output_array)).sum()

    def gradient_check(self, sample_feature, sample_label):
        '''
        梯度检查
        network: 神经网络对象
        sample_feature: 样本的特征
        sample_label: 样本的标签
        '''

        # 获取网络在当前样本下每个连接的梯度
        self.predict(sample_feature)
        self.calc_gradient(sample_label)

        '''
        # 检查梯度 -- dense layer
        epsilon = 10e-4
        for fc in (self.dense, ):
            for i in range(min(5,fc.W.shape[0])):
                for j in range(min(5,fc.W.shape[1])):
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
        # 检查梯度 -- conv layer
        cl = self.conv1
        epsilon = 10e-4
        for d in range(cl.filters[0].weights_grad.shape[0]):
            for i in range(min(5,cl.filters[0].weights_grad.shape[1])):
                for j in range(min(5,cl.filters[0].weights_grad.shape[2])):
                    cl.filters[0].weights[d,i,j] += epsilon
                    output = self.predict(sample_feature)
                    err1 = self.loss(sample_label, output)
                    cl.filters[0].weights[d,i,j] -= 2*epsilon
                    output = self.predict(sample_feature)
                    err2 = self.loss(sample_label, output)
                    expect_grad = (err1 - err2) / (2 * epsilon)
                    cl.filters[0].weights[d,i,j] += epsilon
                    print 'weights(%d,%d,%d): expected - actural %f - %f' % (
                        d, i, j, expect_grad, cl.filters[0].weights_grad[d,i,j])


from input_data import get_training_data_set
from input_data import show

def train_data_set():
    data_set, labels =  get_training_data_set()
    return labels, data_set

def test():
    labels, data_set = train_data_set()
    print labels[-1]
    show(data_set[-1])
    net = Network(6, 16, 100)
    rate = 0.1
    mini_batch = 3
    epoch = 10
    for i in range(epoch):
        net.training(labels, data_set, rate, mini_batch, 10)
        print np.around(net.predict(data_set[-1]),decimals=3).flatten()
        rate /= 2


def gradient_check():
    '''
    梯度检查
    '''
    labels, data_set = train_data_set()
    net = Network(6, 16, 100)
    net.gradient_check(data_set[0], labels[0])
    return net

if __name__ == '__main__':
    test()
