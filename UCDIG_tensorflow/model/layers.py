import tensorflow as tf


def fc_layer(input_tensor, output_size, scope=None):
    shape = input_tensor.get_shape().as_list()

    with tf.variable_scope(scope or 'fc'):
        W = tf.get_variable('W', [shape[1], output_size], tf.float32,
                            initializer=tf.random_normal_initializer(stddev=0.02))
        bias = tf.get_variable('bias', [output_size],
                            initializer=tf.constant_initializer(0.0))
        return tf.matmul(input_tensor, W) + bias


def conv_layer(input_tensor, output_dim, 
                k_h=5, k_w=5, scope=None, stddev=0.02, step_h=2, step_w=2):
    shape = input_tensor.get_shape().as_list()

    with tf.variable_scope(scope or 'conv'):
        W = tf.get_variable('W', [k_h, k_w, shape[-1], output_dim],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_tensor, W, strides=[1, step_h, step_w, 1], padding='SAME')
        bias = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, bias), conv.get_shape())
        return tf.nn.relu(conv)


def maxpool_layer(input_tensor, k_h=5, k_w=5, step_h=2, step_w=2, padding='SAME'):
    return tf.nn.max_pool(input_tensor,
                          ksize=[1, k_h, k_w, 1],
                          strides=[1, step_h, step_w, 1],
                          padding=padding)


def deconv_layer(input_tensor, output_shape,
                 k_h=5, k_w=5, scope=None, stddev=0.02, step_h=2, step_w=2):
    shape = input_tensor.get_shape().as_list()

    with tf.variable_scope(scope or 'deconv'):
        W = tf.get_variable('W', [k_h, k_w, output_shape[-1], shape[-1]], tf.float32,
                            initializer=tf.random_normal_initializer(stddev=stddev))

        deconv = tf.nn.conv2d_transpose(input_tensor, W, output_shape=output_shape,
                                        strides=[1, step_h, step_w, 1])
        bias = tf.get_variable('bias', [output_shape[-1]],
                               initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, bias), deconv.get_shape())
        return deconv


class batch_norm:
    """Code modification of http://stackoverflow.com/a/33950177"""
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum

            self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
            self.name = name

    def __call__(self, x, train=True):
        shape = x.get_shape().as_list()

        if train:
            with tf.variable_scope(self.name) as scope:
                self.beta = tf.get_variable("beta", [shape[-1]],
                                    initializer=tf.constant_initializer(0.0))
                self.gamma = tf.get_variable("gamma", [shape[-1]],
                                    initializer=tf.random_normal_initializer(1., 0.02))
                
                try:
                    batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
                except:
                    batch_mean, batch_var = tf.nn.moments(x, [0, 1], name='moments')
                    
                ema_apply_op = self.ema.apply([batch_mean, batch_var])
                self.ema_mean, self.ema_var = self.ema.average(batch_mean), self.ema.average(batch_var)

                with tf.control_dependencies([ema_apply_op]):
                    mean, var = tf.identity(batch_mean), tf.identity(batch_var)
        else:
            mean, var = self.ema_mean, self.ema_var

        normed = tf.nn.batch_norm_with_global_normalization(
                x, mean, var, self.beta, self.gamma, self.epsilon, scale_after_normalization=True)

        return normed
