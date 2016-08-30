# encoding=utf-8

import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    使用CNN用于情感分析
    整个CNN架构包括词嵌入层，卷积层，max-pooling层和softmax层
    """
    def __init__(
      self, sequence_length, num_classes,vocab_size,embedding_size, embedding_table,
            filter_sizes, num_filters, l2_reg_lambda=0.0):

        # 输入，输出，dropout的placeholder
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # 词嵌入层
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(embedding_table,name="W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # 生成卷积层和max-pooling层
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # 将max-pooling层的各种特征整合在一起
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # 添加全连接层，用于分类
        with tf.name_scope("full-connection"):
            W_fc1 = tf.Variable(tf.truncated_normal([num_filters_total,500], stddev=0.1))
            b_fc1 = tf.Variable(tf.constant(0.1,shape=[500]))
            self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool_flat, W_fc1) + b_fc1)

        # 添加dropout层用于缓和过拟化
        with tf.name_scope("dropout"):
            # self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
            self.h_drop = tf.nn.dropout(self.h_fc1, self.dropout_keep_prob)

        # 产生最后的输出和预测
        with tf.name_scope("output"):
            # W = tf.get_variable(
            #     "W",
            #     shape=[num_filters_total, num_classes],
            #     initializer=tf.contrib.layers.xavier_initializer())
            W = tf.get_variable(
                "W",
                shape=[500, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # 定义模型的损失函数
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # 定义模型的准确率
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")




