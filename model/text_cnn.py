import tensorflow as tf
import numpy as np

class TextCNN(object):
    def __init__(self, fixed_input_length, num_classes, vocal_size, embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        """
        :param fixed_input_length: the input should have same input length, if shorter than this, use <pad> to complete
        :param num_classes:        how many classes you need to class
        :param vocal_size:         how many words your corpus have
        :param embedding_size:     the dim of embedding vector. eg dim=300 for word2vec
        :param filter_sizes:       size of filters which mean the number of words we want cover, eg: [3, 4, 5]
        :param num_filters:        The number of filters per filter size
        """
        self.input_x = tf.placeholder(tf.int32, [None, fixed_input_length], name='input_x')
        self.input_y = tf.placeholder(tf.int32, [None, num_classes], name='input_y')

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        l2_loss = tf.constant(0.0)

        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            self.W = tf.Variable(
                tf.random_uniform([vocal_size, embedding_size], -1.0, 1.0), name='W'
            )
            # in this case, the vector is trained from scratch, but if we use word2vec then we don't need to
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expand = tf.expand_dims(self.embedded_chars, -1)

        # create conv+pool layer for each filter
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope('conv-pool-%s'%filter_size):
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')

                conv = tf.nn.conv2d(
                    self.embedded_chars_expand,
                    W,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='conv'
                )

                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, fixed_input_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool"
                )

                pooled_outputs.append(pooled)

        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

if __name__ == '__main__':
    pass



