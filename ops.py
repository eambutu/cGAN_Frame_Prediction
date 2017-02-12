import tensorflow as tf


class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name='batch_norm'):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=train,
                                            scope=self.name)


def lrelu(x, leak=0.2, name='lrelu'):
    return tf.maximum(x, leak*x)


def linear(input, output_size, scope=None, stddev=0.02, bias_start=0.0):
    shape = input.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("b", [output_size],
                               initializer=tf.constant_initializer(bias_start))

        return tf.matmul(input, matrix) + bias


def conv2d(input, output_dim, len, stride, stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [len, len, input.get_shape()[-1], output_dim],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input, w, strides=[1, stride, stride, 1], padding='SAME')

        b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, b), conv.get_shape())

        return conv


# The transpose(gradient) of conv2d, after deconvolutional networks
def deconv2d(input, output_shape, len, stride, stddev=0.02, name="deconv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [len, len, output_shape[-1], input.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        deconv = tf.nn.conv2d_transpose(input, w, output_shape=output_shape,
                                        strides=[1, stride, stride, 1])
        b = tf.get_variable('b', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, b), deconv.get_shape())

        return deconv
