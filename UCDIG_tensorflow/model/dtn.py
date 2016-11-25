import os
import time
import numpy as np
from glob import glob
import tensorflow as tf
from model.layers import conv_layer, maxpool_layer, deconv_layer, fc_layer,\
                         batch_norm
from model.utils import *


class DTN:
    def __init__(self, sess):
        self.num_epoch = 25
        self.batch_size = 64

        self.output_size = 32
        self.sample_size = 64
        self.image_size = 108

        self.sess = sess

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.alpha = 20000
        self.beta = 1

        self.proj_dir = os.path.dirname(os.path.dirname(
                                            os.path.dirname(
                                                os.path.abspath(__file__))))
        self.sample_dir = os.path.join(self.proj_dir, 'sample')
        self.data_dir = os.path.join(self.proj_dir, 'data')

        self.build_net()

    def test(self):
        src_images = get_image_from_mat(os.path.join(self.data_dir, 'extra_32x32.mat'))
        print(src_images.shape)
        target_images = get_image_from_csv(os.path.join(self.data_dir, 'mnist_train.csv'))
        print(target_images.shape)

    def build_net(self):
        self.src_images = tf.placeholder(tf.float32, [self.batch_size] +
                                         [32, 32, 3])
        self.target_images = tf.placeholder(tf.float32,
                                            [self.batch_size] +
                                            [self.output_size,
                                             self.output_size, 3])

        self.f_src_input = self.f(self.src_images)
        self.g_src_input = self.g(self.f_src_input)
        self.D1, self.D1_logits = self.discriminator(self.g_src_input)
        self.f_target_input = self.f(self.target_images, reuse=True)
        self.g_target_input = self.g(self.f_target_input, reuse=True)
        self.f_g_src_input = self.f(self.g_src_input, reuse=True)
        self.S = self.g(self.f_src_input, reuse=True)
        self.D2, self.D2_logits = self.discriminator(self.g_target_input,
                                                     reuse=True)
        self.D3, self.D3_logits = self.discriminator(self.target_images,
                                                     reuse=True)

        self.d1_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                          self.D1_logits,
                                          np.array([np.array([1.0, 0.0, 0.0])
                                                   for i in range(
                                                       self.batch_size)])))
        self.d2_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                      self.D2_logits,
                                      np.array([np.array([0.0, 1.0, 0.0])
                                                for i in range(
                                                      self.batch_size)])))
        self.d3_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                      self.D3_logits,
                                      np.array([np.array([0.0, 0.0, 1.0])
                                               for i in range(
                                                      self.batch_size)])))
        self.d_loss = self.d1_loss + self.d2_loss + self.d3_loss

        self.g_gang_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                          self.D1_logits,
                                          np.array([np.array([0.0, 0.0, 1.0])
                                                    for i in range(
                                                            self.batch_size)]))) + \
                           tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                          self.D2_logits,
                                          np.array([np.array([0.0, 0.0, 1.0])
                                                    for i in range(
                                                            self.batch_size)])))
        #self.g_const_loss = tf.reduce_mean(tf.squared_difference(self.f_src_input, self.f_g_src_input))
        #self.g_tid_loss = tf.reduce_mean(tf.squared_difference(self.target_images, self.g_target_input))
        self.g_const_loss = tf.reduce_mean(tf.square(tf.sub(self.f_src_input, self.f_g_src_input)))
        self.g_tid_loss = tf.reduce_mean(tf.square(tf.sub(self.target_images, self.g_target_input)))

        self.g_loss = self.g_gang_loss + self.alpha * self.g_const_loss + self.beta * self.g_tid_loss

    def f(self, image, reuse=False):
        """
        image:
            32 x 32 x 3

        network:
            1. conv2d filter size 64, maxpool, relu
            2. conv2d filter size 128, maxpool, relu
            3. conv2d filter size 256, maxpool, relu
            4. conv2d filter size 128, maxpool, relu
        """
        if reuse:
            tf.get_variable_scope().reuse_variables()

        conv1 = conv_layer(image, 64, scope='f_conv1')
        conv1 = maxpool_layer(conv1)  # 8 x 8 x 64

        conv2 = conv_layer(conv1, 128, scope='f_conv2')
        covn2 = maxpool_layer(conv2)  # 4 x 4 x 128

        conv3 = conv_layer(conv2, 256, scope='f_conv3')
        covn3 = maxpool_layer(conv3) # 2 x 2 x 256

        conv4 = conv_layer(conv3, 128, scope='f_conv4')
        covn4 = maxpool_layer(conv4) # 1 x 1 x 128

        return conv4

    def g(self, f_output, reuse=False):
        """
        f_output:
            1 x 1 x 128

        network:
            1. deconv2d 2 x 2 x 64, batch normal relu
            2. deconv2d 4 x 4 x 32, batch normal relu
            3. deconv2d 16 x 16 x 16, batch normal relu
            4. deconv2d 32 x 32 x 3, batch normal relu
        """
        if reuse:
            tf.get_variable_scope().reuse_variables()

        deconv0 = deconv_layer(f_output, [self.batch_size, 2, 2, 128], scope='g_dc0')
        deconv0 = tf.nn.relu(self.g_bn0(deconv0))

        deconv1 = deconv_layer(deconv0, [self.batch_size, 4, 4, 64], scope='g_dc1')
        deconv1 = tf.nn.relu(self.g_bn1(deconv1))
        
        deconv2 = deconv_layer(deconv1, [self.batch_size, 8, 8, 32], scope='g_dc2')
        deconv2 = tf.nn.relu(self.g_bn2(deconv2))

        deconv3 = deconv_layer(deconv2, [self.batch_size, 16, 16, 16], scope='g_dc3')
        deconv3 = tf.nn.relu(self.g_bn3(deconv3))
        
        deconv4 = deconv_layer(deconv3, [self.batch_size, 32, 32, 3], scope='g_dc4')

        return tf.nn.tanh(deconv4)

    def discriminator(self, image, reuse=False):
        """
        image:
            32 x 32 x 3

        network:
            4 batch-normalization relu convolution layer
        """
        if reuse:
            tf.get_variable_scope().reuse_variables()

        conv0 = tf.nn.relu(conv_layer(image, 64, scope='d_conv0'))
        conv1 = tf.nn.relu(self.d_bn1(conv_layer(conv0, 128, scope='d_conv1')))
        conv2 = tf.nn.relu(self.d_bn2(conv_layer(conv1, 256, scope='d_conv2')))
        conv3 = tf.nn.relu(self.d_bn3(conv_layer(conv2, 512, scope='d_conv3')))
        fc4 = fc_layer(tf.reshape(conv3, [self.batch_size, -1]), 3, scope='d_fc1')
        return tf.nn.softmax(fc4), fc4
    
    def train(self):
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        d_optim = tf.train.AdamOptimizer(learning_rate=0.002, beta1=0.5) \
                                         .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(learning_rate=0.002, beta1=0.5) \
                                         .minimize(self.g_loss, var_list=self.g_vars)

        
        src_images = get_image_from_mat(os.path.join(self.data_dir, 'extra_32x32.mat'))
        target_images = get_image_from_csv(os.path.join(self.data_dir, 'mnist_train.csv'))

        sample_images = src_images[0:self.sample_size]
        sample_images = np.array(sample_images) / 127.5 - 1
        sample_images = sample_images.astype(np.float32)
        save_images(sample_images, [8, 8],
                    '{}/samples.png'.format(self.sample_dir))

        tf.initialize_all_variables().run()

        counter = 1
        start_time = time.time()

        for epoch in range(self.num_epoch):
            batch_idxs = min(len(src_images), len(target_images)) // self.batch_size

            for idx in range(0, batch_idxs):
                batch_src_images = src_images[idx*self.batch_size: (idx+1)*self.batch_size]
                batch_src_images = np.array(batch_src_images) / 127.5 - 1
                batch_src_images.astype(np.float32)

                batch_target_images = target_images[idx*self.batch_size: (idx+1)*self.batch_size]
                batch_target_images = np.array(batch_target_images) / 127.5 - 1
                batch_target_images.astype(np.float32)

                self.sess.run([g_optim], feed_dict={self.src_images: batch_src_images, self.target_images: batch_target_images})
                self.sess.run([d_optim], feed_dict={self.src_images: batch_src_images, self.target_images: batch_target_images})
                self.sess.run([g_optim], feed_dict={self.src_images: batch_src_images, self.target_images: batch_target_images})

                d_error = self.d_loss.eval({self.src_images: batch_src_images, self.target_images: batch_target_images})
                g_error = self.g_loss.eval({self.src_images: batch_src_images, self.target_images: batch_target_images})
                g_gang_error = self.g_gang_loss.eval({self.src_images: batch_src_images, self.target_images: batch_target_images})
                g_const_error = self.g_const_loss.eval({self.src_images: batch_src_images, self.target_images: batch_target_images})
                g_tid_error = self.g_tid_loss.eval({self.src_images: batch_src_images, self.target_images: batch_target_images})

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, g_gang_loss: %.8f, g_const_loss: %.8f, g_tid_loss: %.8f" \
                    % (epoch, idx, batch_idxs,
                        time.time() - start_time, d_error, g_error, g_gang_error, self.alpha*g_const_error, self.beta*g_tid_error))
                if np.mod(counter, 10) == 1:
                    samples = self.sess.run(
                        self.S,
                        feed_dict={self.src_images: sample_images}
                    )
                    save_images(samples, [8, 8],
                                '{}/train_{:02d}_{:04d}.png'.format(self.sample_dir, epoch, idx))
