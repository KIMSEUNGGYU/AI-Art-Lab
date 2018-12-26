#############################################
# Date          : 2018.03.25
# Programmer    : Seounggyu Kim
# description   : CAN 모델
# Update Date   : 2018.04.30
# Update        : DCGAN 텐서보드 추가
#############################################

import os
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *

class DCGAN(object):
    def __init__(self, sess):
        self.sess = sess
        self.data = glob(os.path.join("./data", 'celebA', '*.jpg'))


    def build_model(self):
        ## Creating a variable
        self.real_image = tf.placeholder(tf.float32, [64, 256, 256, 3], name = 'real_images')
        self.random_noise = tf.placeholder(tf.float32, [None, 100], name='random_noise')

        #### tensorboard
        self.random_noise_summary = tf.summary.histogram("random_noise_summary", self.random_noise)

        ##  model build
        # Creating generator / discriminator
        self.generator = self.generator(self.random_noise)
        self.discriminator_police_sigmoid, self.discriminator_police = self.discriminator(self.real_image, reuse=False)
        self.discriminator_thief_sigmoid, self.discriminator_thief = self.discriminator(self.generator, reuse=True)
        self.sampler = self.sampler(self.random_noise)

        #### tensorboard
        self.discriminator_police_summary = tf.summary.histogram("discriminator_police_summary", self.discriminator_police_sigmoid)
        self.discriminator_thief_summary = tf.summary.histogram("discriminator_thief_summary", self.discriminator_thief_sigmoid)
        self.generator_summary = tf.summary.image("generator_summary", self.generator)


        ## Creating loss function - Find cost
        # real discriminator cost
        self.discriminator_police_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits = self.discriminator_police,
                                                    labels = tf.ones_like(self.discriminator_police_sigmoid)))
        # fake discriminator cost
        self.discriminator_thief_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits = self.discriminator_thief,
                                                    labels = tf.zeros_like(self.discriminator_thief_sigmoid)))

        #### tensorboard
        self.discriminator_police_loss_summary = tf.summary.scalar("discriminator_police_loss_summary", self.discriminator_police_loss)
        self.discriminator_thief_loss_summary = tf.summary.scalar("discriminator_thief_loss_summary", self.discriminator_thief_loss)

        # generator cost
        self.generator_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits = self.discriminator_thief,
                                                    labels = tf.ones_like(self.discriminator_thief_sigmoid)))
        # discriminator cost
        self.discriminator_loss = self.discriminator_police_loss + self.discriminator_thief_loss

        #### tensorboard
        self.generator_loss_summary = tf.summary.scalar("generator_loss_summary", self.generator_loss)
        self.discriminator_loss_summary = tf.summary.scalar("discriminator_loss_summary", self.discriminator_loss)

        t_vars = tf.trainable_variables()

        self.discriminator_vars = [var for var in t_vars if 'd_' in var.name]
        self.generator_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self):
        # Creating Optimizer
        discriminator_optimizer = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.discriminator_loss, var_list=self.discriminator_vars)
        generator_optimizer = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.generator_loss, var_list=self.generator_vars)

        #### tensorboard
        generator_optimizer_summary = tf.summary.merge([self.random_noise_summary, self.discriminator_thief_summary, self.generator_summary,
                                                        self.discriminator_thief_loss_summary, self.generator_loss_summary])
        discriminator_optimizer_summary = tf.summary.merge([self.random_noise_summary, self.discriminator_police_summary,
                                                            self.discriminator_police_loss_summary, self.discriminator_loss_summary])
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        tf.global_variables_initializer().run()


        ## Creating sample -> test part
        sample_random_noise = np.random.uniform(-1, 1, size=(64, 100))
        sample_images_path = self.data[0 : 64]
        sample_images_ = [get_image(sample_image_path,
                            input_height = 256,
                            input_width = 256,
                            resize_height = 256,
                            resize_width = 256,
                            crop=False) for sample_image_path in sample_images_path]
        sample_images = np.array(sample_images_).astype(np.float32)

        # checkpoint variable
        counter = 1

        # checkpoint load
        # checkpoint_dir_path = os.path.join(self.checkpoint_dir, self.checkpint_dir_model)
        checkpoint_dir_path = os.path.join('checkpoint', 'celebA')
        could_load, checkpoint_counter = checkpoint_load(self.sess, self.saver, 'checkpoint', 'celebA')
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        ## training
        for epoch in xrange(35):
            batch_index = min(len(self.data), np.inf) // 64

            for index in xrange(0, batch_index):
                ## Creating batch -> training part
                batch_images_path = self.data[index * 64 : (index + 1) * 64]
                batch_images_ = [get_image(batch_image_path,
                                           input_height = 256,
                                           input_width = 256,
                                           resize_height = 256,
                                           resize_width = 256,
                                           crop=False) for batch_image_path in batch_images_path]
                batch_images = np.array(batch_images_).astype(np.float32)
                batch_random_noise = np.random.uniform(-1, 1, [64, 100]).astype(np.float32)

                ## Update
                # Update D network
                _, summary = self.sess.run([discriminator_optimizer, discriminator_optimizer_summary],
                                                                        feed_dict={ self.real_image: batch_images,
                                                                                    self.random_noise: batch_random_noise })
                self.writer.add_summary(summary, counter)

                # Update G network
                _, summary = self.sess.run([generator_optimizer, generator_optimizer_summary],
                                                                    feed_dict={ self.random_noise:  batch_random_noise })
                self.writer.add_summary(summary, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, summary = self.sess.run([generator_optimizer, generator_optimizer_summary],
                                                                    feed_dict={ self.random_noise: batch_random_noise })
                self.writer.add_summary(summary, counter)

                errD_fake = self.discriminator_thief_loss.eval({ self.random_noise: batch_random_noise })
                errD_real = self.discriminator_police_loss.eval({ self.real_image: batch_images })
                errG = self.generator_loss.eval({self.random_noise: batch_random_noise})

                # global value --> checkpoint value
                counter += 1

                # if np.mod(counter, 10) == 1:
                print("Epoch: [%2d/%2d] [%4d/%4d] , d_loss: %.8f, g_loss: %.8f" % (epoch, 25, index, batch_index, errD_fake+errD_real, errG))

                if np.mod(counter, 100) == 1:
                    try:
                        samples, d_loss, g_loss = self.sess.run([self.sampler, self.discriminator_loss, self.generator_loss],
                                                                feed_dict={self.random_noise: sample_random_noise,
                                                                           self.real_image: sample_images, }, )
                        save_images(samples, image_manifold_size(samples.shape[0]),'./{}/train_{:02d}_{:04d}.png'.format('samples', epoch, index))
                        print("[SAVE IMAGE] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
                    except Exception as e:
                        print("image save error! ", e)

                ## checkpoint save
                if np.mod(counter, 500) == 1:
                    print("[SAVE CHECKPOINT]")
                    checkpoint_save(self.sess, self.saver, checkpoint_dir_path, counter)


    def discriminator(self, input_, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            discriminator_layer0 = lrelu(conv2d(input_, 32, name='d_h0_conv'))                                          # [256, 256, 3], 32 => (128, 128, 32)
            discriminator_layer1 = lrelu(batch_norm(conv2d(discriminator_layer0, 64, name='d_h1_conv'), 'd_bn1'))       # (?, 64, 64, 64)
            discriminator_layer2 = lrelu(batch_norm(conv2d(discriminator_layer1, 128, name='d_h2_conv'), 'd_bn2'))      # (?, 32, 32, 128)
            discriminator_layer3 = lrelu(batch_norm(conv2d(discriminator_layer2, 256, name='d_h3_conv'), 'd_bn3'))      # (?, 16, 16, 256)
            discriminator_layer4 = lrelu(batch_norm(conv2d(discriminator_layer3, 512, name='d_h4_conv'), 'd_bn4'))      # (?, 8, 8, 512)
            discriminator_layer5 = lrelu(batch_norm(conv2d(discriminator_layer4, 512, name='d_h5_conv'), 'd_bn5'))      # (?, 4, 4, 512)

            shape = np.product(discriminator_layer5.get_shape()[1:].as_list())
            discriminator_layer6 = tf.reshape(discriminator_layer5, [-1, shape])
            discriminator_output  = linear(discriminator_layer6, 1, 'd_ro_lin')                                           # (?, 1)

            # discriminator_layer0 = lrelu(conv2d(input_, 128, name='d_h0_conv'))                                         # (64, 64, 64, 3), 128  -> (?, 32, 32, 128)
            # discriminator_layer1 = lrelu(batch_norm(conv2d(discriminator_layer1, 256, name='d_h1_conv'), 'd_bn1'))      # (?, 32, 32, 128), 256 -> (?, 16, 16, 256)
            # discriminator_layer2 = lrelu(batch_norm(conv2d(discriminator_layer2, 512, name='d_h2_conv'), 'd_bn2'))      # (?, 16, 16, 256), 512 -> (?, 8, 8, 512)
            # discriminator_layer3 = lrelu(batch_norm(conv2d(discriminator_layer3, 1024, name='d_h3_conv'), 'd_bn3'))     # (?, 8, 8, 512), 1024  -> (?, 4, 4, 1024)
            # discriminator_output = linear(tf.reshape(discriminator_layer5, [64, -1]), 1, 'd_h4_lin')                    # (64, 1)

            return tf.nn.sigmoid(discriminator_output), discriminator_output                                              # (64, 1)

    def generator(self, random_noise):
        with tf.variable_scope("generator") as scope:
            generator_linear = linear(random_noise, 64 * 16 * 4 * 4, 'g_h0_lin')                                        # (?, 100), 16,384 -> (100, 16,384)
            generator_reshape = tf.reshape(generator_linear, [-1, 4, 4, 64 * 16])                                       # (?, 4, 4, 1024)
            generator_input = tf.nn.relu(batch_norm(generator_reshape , 'g_bn0'))                                       # (?, 4, 4, 1024)

            generator_layer1 = deconv2d(generator_input, [64, 8, 8, 64 * 16], name='g_layer1')                          # (?, 8, 8, 512)
            generator_layer1 = tf.nn.relu(batch_norm(generator_layer1, 'g_bn1'))                                        # (?, 8, 8, 512)

            generator_layer2 = deconv2d(generator_layer1, [64, 16, 16, 64 * 8], name='g_layer2')                        # (?, 16, 16, 256)
            generator_layer2 = tf.nn.relu(batch_norm(generator_layer2, 'g_bn2'))                                        # (?, 16, 16, 256)

            generator_layer3 = deconv2d(generator_layer2, [64, 32, 32, 64 * 4], name='g_layer3')                        # (?, 32, 32, 128)
            generator_layer3 = tf.nn.relu(batch_norm(generator_layer3, 'g_bn3'))                                        # (?, 32, 32, 128)

            generator_layer4 = deconv2d(generator_layer3, [64, 64, 64, 64 * 2], name='g_layer4')                        # (?, 64, 64, 128)
            generator_layer4 = tf.nn.relu(batch_norm(generator_layer4, 'g_bn4'))                                        # (?, 64, 64, 128)

            generator_layer5 = deconv2d(generator_layer4, [64, 128, 128, 64], name='g_layer5')                          # (?, 128, 128, 64)
            generator_layer5 = tf.nn.relu(batch_norm(generator_layer5, 'g_bn5'))                                        # (?, 128, 128, 64)

            generator_output = deconv2d(generator_layer5, [64, 256, 256, 3], name='g_output')                           # (?, 256, 256, 3)
            generator_output = tf.nn.tanh(generator_output)                                                             # (?, 256, 256, 3)

            return generator_output                                                                                     # (?, 64, 64, 3)

    def sampler(self, random_noise):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            sampler_linear = linear(random_noise, 64 * 16 * 4 * 4, 'g_h0_lin')                                          # (?, 100), 16,384 -> (100, 16,384)
            sampler_reshape = tf.reshape(sampler_linear, [-1, 4, 4, 64 * 16])                                           # (?, 4, 4, 1024)
            sampler_input = tf.nn.relu(batch_norm(sampler_reshape, 'g_bn0', train=False))                               # (?, 4, 4, 1024)

            sampler_layer1 = deconv2d(sampler_input, [64, 8, 8, 64 * 16], name='g_layer1')                              # (?, 8, 8, 512)
            sampler_layer1 = tf.nn.relu(batch_norm(sampler_layer1, 'g_bn1', train=False))                               # (?, 8, 8, 512)

            sampler_layer2 = deconv2d(sampler_layer1, [64, 16, 16, 64 * 8], name='g_layer2')                            # (?, 16, 16, 256)
            sampler_layer2 = tf.nn.relu(batch_norm(sampler_layer2, 'g_bn2', train=False))                               # (?, 16, 16, 256)

            sampler_layer3 = deconv2d(sampler_layer2, [64, 32, 32, 64 * 4], name='g_layer3')                            # (?, 32, 32, 128)
            sampler_layer3 = tf.nn.relu(batch_norm(sampler_layer3, 'g_bn3', train=False))                               # (?, 32, 32, 128)

            sampler_layer4 = deconv2d(sampler_layer3, [64, 64, 64, 64 * 2], name='g_layer4')                            # (?, 64, 64, 128)
            sampler_layer4 = tf.nn.relu(batch_norm(sampler_layer4, 'g_bn4', train=False))                               # (?, 64, 64, 128)

            sampler_layer5 = deconv2d(sampler_layer4, [64, 128, 128, 64], name='g_layer5')                              # (?, 128, 128, 64)
            sampler_layer5 = tf.nn.relu(batch_norm(sampler_layer5, 'g_bn5', train=False))                               # (?, 128, 128, 64)

            sampler_output = deconv2d(sampler_layer5, [64, 256, 256, 3], name='g_output')                               # (?, 256, 256, 3)
            sampler_output = tf.nn.tanh(sampler_output)                                                                 # (?, 256, 256, 3)

            return sampler_output                                                                                       # (?, 64, 64, 3)
