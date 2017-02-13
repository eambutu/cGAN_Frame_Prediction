"""
https://github.com/carpedm20/DCGAN-tensorflow
"""
import time
import os

import tensorflow as tf
import numpy as np

from ops import *
from utils import *


class FlowGAN(object):
    def __init__(self, sess, data_file, data_dir, dataset_name, input_height,
                 input_width, output_height, output_width, is_crop,
                 batch_size=64, z_dim=100, gf_dim=64, df_dim=64, gfc_dim=1024,
                 dfc_dim=1024, c_dim=3, checkpoint_dir=None):
        """
        Args:
            sess: tensorflow session
            batch_size: size of batch
            y_dim: Dimension of dim for y
            z_dim: Dimension of dim for z
            gf_dim: Dimension of gen filters in first conv layer
            df_dim: Dimension of disc filters in first conv layer
            gfc_dim: Dimension of gen units for fully connected layer
            dfc_dim: Dimension of disc units for fully connected layer
            c_dim: Dimension of image color. 3 or 1
        """
        self.sess = sess

        self.data_file = data_file
        self.data_dir = data_dir
        self.dataset_name = dataset_name

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.is_crop = is_crop

        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.c_dim = c_dim

        # batch normalization: deals with poor initialization
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')
        self.d_bn4 = batch_norm(name='d_bn4')

        self.gi_bn1 = batch_norm(name='gi_bn1')
        self.gi_bn2 = batch_norm(name='gi_bn2')

        self.gz_bn0 = batch_norm(name='gz_bn0')
        self.gz_bn1 = batch_norm(name='gz_bn1')

        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')
        self.g_bn4 = batch_norm(name='g_bn4')

        self.checkpoint_dir = checkpoint_dir
        self._build_model()

    def _build_model(self):
        if self.is_crop:
            image_dims = [self.output_height, self.output_width, (self.c_dim+2)]
        else:
            image_dims = [self.input_height, self.input_height, (self.c_dim+2)]

        self.first_frames = tf.placeholder(
            tf.float32, [self.batch_size] + image_dims, name='first_frames')
        self.last_frames = tf.placeholder(
            tf.float32, [self.batch_size] + image_dims, name='last_frames')
        self.sample_inputs = tf.placeholder(
            tf.float32, [self.sample_num] + image_dims, name='sample_inputs')

        first_frames = self.first_frames
        last_frames = self.last_frames

        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')

        self.G = self.generator(first_frames, self.z)
        self.D, self.D_logits = self.discriminator(last_frames)

        self.sampler = self.sampler(self.z)
        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D_logits, labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D_logits, labels=tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D_logits_, labels=tf.ones_like(self.D_)))

        self.d_loss = self.d_loss_real + self.d_loss_fake

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, config):
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)

        tf.initialize_all_variables().run()

        sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))

        # TODO: get sample files
        sample_files = 'blah'
        sample = ['blah']
        sample_inputs = 'blah'

        counter = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in xrange(config.epoch):
            data_size = -1
            with open(self.data_file, 'r') as fin:
                data_size = len(fin.readlines())
            num_batches = min(len(data_size, config.train_size)) // config.batch_size

            for idx in xrange(0, num_batches):
                batch_idxs = [i for i in xrange(idx*config.batch_size,
                                                (idx+1)*config.batch_size)]
                batch_images = get_images(self.data_dir, self.data_file,
                                          batch_idxs, self.output_height,
                                          self.output_width, True)
                f_frames = batch_images[:, :, :, :5]
                l_frames = batch_images[:, :, :, 5:]

                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
                    .astype(np.float32)

                # Update D network
                _ = d_optim.eval({self.first_frames: f_frames,
                                  self.last_frames: l_frames, self.z: batch_z})

                # Update G network
                _ = g_optim.eval({self.first_frame: f_frames, self.z: batch_z})
                _ = g_optim.eval({self.first_frame: f_frames, self.z: batch_z})

                errD_fake = self.d_loss_fake.eval({self.first_frame: f_frames,
                                                   self.z: batch_z})
                errD_real = self.d_loss_real.eval({self.last_frames: l_frames})
                errG = self.g_loss.eval({self.first_frames: f_frames,
                                         self.z: batch_z})

                counter += 1
                print "Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                    % (epoch, idx, batch_idxs, time.time() - start_time,
                       errD_fake + errD_real, errG)

                if counter % 100 == 1:
                    # Save the images from sampler
                    print 'Hello'

                if counter % 500 == 1:
                    self.save(self.checkpoint_dir, counter)

    def discriminator(self, image, y=None, reuse=False):
        with tf.variable_scope('D') as scope:
            if reuse:
                scope.reuse_variables()

            h0 = lrelu(conv2d(image, self.df_dim, 5, 2, name='d_h0_conv'))
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, 5, 2, name='d_h1_conv')))
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, 5, 2, name='d_h2_conv')))
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, 5, 2, name='d_h3_conv')))
            h4 = lrelu(self.d_bn4(conv2d(h3, self.df_dim*2, 5, 2, name='d_h4_conv')))
            h5 = linear(tf.reshape(h4, [self.batch_size, -1]), 1, 'd_h5_lin')

            return tf.nn.sigmoid(h5), h5

    def generator(self, image, z):
        with tf.variable_scope('G') as scope:
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_h4, s_h8, s_h16 = \
                int(s_h/2), int(s_h/4), int(s_h/8), int(s_h/16)
            s_w2, s_w4, s_w8, s_w16 = \
                int(s_w/2), int(s_w/4), int(s_w/8), int(s_w/16)

            # convolutional layers on image
            i0 = lrelu(conv2d(image, self.gf_dim, 5, 2, name='g_i0_conv'))
            i1 = lrelu(self.gi_bn1(conv2d(i0, self.gf_dim*2, 5, 2, name='g_i1_conv')))
            i2 = lrelu(self.gi_bn2(conv2d(i1, self.gf_dim*4, 5, 2, name='g_i2_conv')))

            # project 'z' and reshape
            self.z_ = linear(z, self.gf_dim*2*s_h15*s_w16, 'g_z0_lin')

            self.z0 = tf.reshape(self.z_, [-1, s_h16, s_w16, self.gf_dim*2])
            z0 = tf.nn.relu(self.gz_bn0(self.z0))

            self.z1 = deconv2d(z0, [self.batch_size, s_h8, s_w8, self.gf_dim*2],
                               4, 2, name='g_z1')
            z1 = tf.nn.relu(self.gz_bn1(self.z1))

            # deconvolution and convolution layers
            self.h0 = tf.pack([i2, z1])

            self.h1 = deconv2d(h0, [self.batch_size, s_h4, s_w4, self.gf_dim*4],
                               4, 2, name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(self.h1))

            self.h2 = deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim*2],
                               4, 2, name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(self.h2))

            self.h3 = conv2d(h2, self.gf_dim*2, 3, 1, name='g_h3_conv')
            h3 = tf.nn.relu(self.g_bn3(self.h3))

            self.h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.gf_dim],
                               4, 2, name='g_h4')
            h4 = tf.nn.relu(self.g_bn4(self.h4))

            self.h5 = conv2d(h4, self.gf_dim, 3, 1, name='g_h3_conv')

            return tf.nn.tanh(h5)

    def sampler(self, z):
        with tf.variable_scope('G') as scope:
            scope.reuse_variables()

            s_h, s_w = self.output_height, self.output_width
            s_h2, s_h4, s_h8, s_h16 = \
                int(s_h/2), int(s_h/4), int(s_h/8), int(s_h/16)
            s_w2, s_w4, s_w8, s_w16 = \
                int(s_w/2), int(s_w/4), int(s_w/8), int(s_w/16)

            # project 'z' and reshape
            z_ = linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin')

            h0 = tf.reshape(z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(h0, train=False))

            h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(h1, train=False))

            h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2, train=False))

            h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3')
            h3 = tf.nn.relu(self.g_bn3(h3, train=False))

            h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')

            return tf.nn.tanh(h4)


    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.dataset_name, self.batch_size,
            self.output_height, self.output_width)

    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Success to read {}".format(ckpt_name))
            return True
        else:
            print(" [*] Failed to find a checkpoint")
            return False
