#############################################
# Date          : 2018.03.25
# Programmer    : Seounggyu Kim
# description   : CAN 모델
# Update Date   : 2018.04.21
# Update        : 텐서 보드 추가
#############################################

import os
import sys
import tensorflow as tf
import numpy as np
import re
from glob import glob

from six.moves import xrange
from random import shuffle

from ops import *
from utils import *



class CAN(object):
    def __init__(self, sess):
        self.sess = sess
        self.data = glob(os.path.join("./data", 'wikiart', '*/*.jpg'))########## '*/*.jpg' 해야 모든 파일 읽어 옴 (디렉터리 안의 이미지)

        self.sample_size = 32
        self.batch_size  = 32
        self.epoch = 100

        self.label_dim = 27             # wikiart class num
        self.random_noise_dim = 100

        self.input_size = 256
        self.output_size = 256

        self.sample_dir = 'samples'
        self.checkpoint_dir = 'checkpoint'
        self.checkpint_dir_model = 'wikiart'
        self.data_dir = 'data'

        ## get label(classification) data
        self.label_dict = {}
        path_list = glob('./data/wikiart/**/', recursive=True)[1:]
        print('!!!!!11', path_list)
        for i, elem in enumerate(path_list):
            self.label_dict[elem[15:-1]] = i

        ## Check required directory and make directory
        if not os.path.exists(self.checkpoint_dir):
            print('NO checkpoint directory => Make checkpoint directory')
            os.makedirs(self.checkpoint_dir)

        if not os.path.exists(self.sample_dir):
            print('NO sample directory => Make sample directory')
            os.makedirs(self.sample_dir)

        if not os.path.exists(self.data_dir) or not self.data:
            # print(self.data)
            print('\nPROCESS END / 프로세스 종료 합니다. ')
            print('원인: No data directory or No image data')
            sys.exit(1)

    def build_model(self):
        ## Creating a variable
        self.y  = tf.placeholder(tf.float32, [None, 27], name = 'y')
        self.real_image = tf.placeholder(tf.float32, [self.batch_size, 256 , 256 , 3], name = 'real_images')
        self.random_noise = tf.placeholder(tf.float32, [None, self.random_noise_dim], name='random_noise')

        #### tensorboard
        self.random_noise_summary = tf.summary.histogram("random_noise_summary", self.random_noise)
        # z_sum

        ##  model build
        # Creating generator / discriminator
        self.generator  = self.generator(self.random_noise)
        self.discriminator_police_sigmoid, self.discriminator_police, self.discriminator_police_class_softmax, self.discriminator_police_class = self.discriminator(self.real_image, reuse=False)
        self.discriminator_thief_sigmoid, self.discriminator_thief, self.discriminator_thief_class_softmax, self.discriminator_thief_class = self.discriminator(self.generator, reuse=True)
        self.sampler = self.sampler(self.random_noise)

        #### tensorboard
        self.discriminator_police_summary = tf.summary.histogram("discriminator_police_summary", self.discriminator_police_sigmoid)
        # d_sum

        self.discriminator_police_class_summary = tf.summary.histogram("discriminator_police_class_summary", self.discriminator_police_class_softmax)
        # d_c_sum

        self.discriminator_thief_summary = tf.summary.histogram("discriminator_thief_summary", self.discriminator_thief_sigmoid)
        # d__sum
        self.discriminator_thief_class_summary = tf.summary.histogram("discriminator_thief_class_summary", self.discriminator_thief_class_softmax)
        # d_c__sum
        self.generator_summary = tf.summary.image("generator_summary", self.generator)
        # G_sum


        ## Find Accuracy
        # classifcation real_label 와 진짜 판별기 라벨
        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.discriminator_police_class, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


        ## Creating loss function - Find cost
        # real discriminator cost
        self.discriminator_police_loss
            = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits = self.discriminator_police,
                labels = tf.ones_like(self.discriminator_police_sigmoid)))

        # fake discriminator cost
        self.discriminator_thief_loss
            = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits = self.discriminator_thief,
                labels = tf.ones_like(self.discriminator_thief_sigmoid)))

        # classific_discriminator cost
        self.discriminator_loss_class_real
            = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits = self.discriminator_police_class,
                labels = 1.0 * self.y))

        # generator classific cost
        self.generator_loss_class_fake
            = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits = self.discriminator_thief_class,
                labels = (1.0 / 27) *
                tf.ones_like(self.discriminator_thief_class_softmax)))







        # generator cost
        self.generator_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits = self.discriminator_thief,
                                                    labels = tf.ones_like(self.discriminator_thief_sigmoid)))

        # generator, discriminator loss
        self.generator_loss = self.generator_loss_fake + 1.0 * self.generator_loss_class_fake                                           #
        self.discriminator_loss = self.discriminator_police_loss + self.discriminator_thief_loss + self.discriminator_loss_class_real   # 1 + 0 + 1 = 2


        #### tensorboard
        self.discriminator_police_loss_summary = tf.summary.scalar("discriminator_police_loss_summary", self.discriminator_police_loss)
        # d_loss_real_sum

        self.discriminator_thief_loss_summary = tf.summary.scalar("discriminator_thief_loss_summary", self.discriminator_thief_loss)
        # d_loss_fake_sum

        self.discriminator_police_class_loss_summary = tf.summary.scalar("discriminator_police_class_loss", self.discriminator_loss_class_real)
        # d_loss_class_real_sum
        self.generator_loss_class_fake_summary = tf.summary.scalar("generator_loss_class_fake", self.generator_loss_class_fake)
        # g_loss_class_fake_sum

        self.generator_loss_summary = tf.summary.scalar("generator_loss_summary", self.generator_loss)
        # g_loss_sum
        self.discriminator_loss_summary = tf.summary.scalar("discriminator_loss_summary", self.discriminator_loss)
        # d_loss_sum

        t_vars = tf.trainable_variables()
        self.discriminator_vars = [var for var in t_vars if 'd_' in var.name]
        self.generator_vars = [var for var in t_vars if 'g_' in var.name]
        # Creating checkpoint saver
        self.saver = tf.train.Saver()

    def train(self):
        # Creating Optimizer
        discriminator_optimizer = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.discriminator_loss,var_list=self.discriminator_vars)
        generator_optimizer = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.generator_loss, var_list=self.generator_vars)

        #### tensorboard
        generator_optimizer_summary = tf.summary.merge([self.random_noise_summary, self.discriminator_thief_summary, self.generator_summary,
                                                        self.discriminator_thief_loss_summary, self.generator_loss_summary])

        discriminator_optimizer_summary = tf.summary.merge([self.random_noise_summary, self.discriminator_police_summary,
                                                            self.discriminator_police_loss_summary, self.discriminator_loss_summary,
                                                            self.discriminator_police_class_loss_summary, self.generator_loss_class_fake_summary ])
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)


        tf.global_variables_initializer().run()

        ## Creating sample -> test part
        sample_random_noise = np.random.normal(0, 1, [self.sample_size, self.random_noise_dim]).astype(np.float32)

        shuffle(self.data)
        sample_images_path = self.data[0 : self.sample_size]
        sample_images_ = [get_image(sample_image_path,
                            input_height = self.input_size,
                            input_width = self.input_size,
                            resize_height = self.output_size,
                            resize_width = self.output_size,
                            crop=False) for sample_image_path in sample_images_path]
        sample_images = np.array(sample_images_).astype(np.float32)
        sample_labels = get_y(sample_images_path, self.label_dim, self.label_dict)      # get label(classification)

        # checkpoint variable
        counter = 1

        # checkpoint load
        checkpoint_dir_path = os.path.join(self.checkpoint_dir, self.checkpint_dir_model)
        could_load, checkpoint_counter = checkpoint_load(self.sess, self.saver, self.checkpoint_dir, self.checkpint_dir_model)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        ## training
        for epoch in xrange(self.epoch):
            shuffle(self.data)

            batch_index = min(len(self.data), np.inf) // self.batch_size
            for index in xrange(0, batch_index):
                ## Creating batch -> training part
                batch_images_path = self.data[index * self.batch_size : (index + 1) * self.batch_size]
                batch_images_ = [get_image(batch_image_path,
                                           input_height = self.input_size,
                                           input_width = self.input_size,
                                           resize_height = self.output_size,
                                           resize_width = self.output_size,
                                           crop=False) for batch_image_path in batch_images_path]
                batch_images = np.array(batch_images_).astype(np.float32)
                batch_labels = get_y(batch_images_path, self.label_dim, self.label_dict)            # get label(classification)
                batch_random_noise = np.random.normal(0, 1, [self.batch_size, self.random_noise_dim]).astype(np.float32)

                ## Update
                # Update D network
                _, summary = self.sess.run([discriminator_optimizer, discriminator_optimizer_summary],
                                                                            feed_dict={ self.real_image: batch_images,
                                                                                        self.random_noise: batch_random_noise,
                                                                                        self.y : batch_labels })
                self.writer.add_summary(summary, counter)

                # Update G network
                _, summary = self.sess.run([generator_optimizer, generator_optimizer_summary],
                                                                            feed_dict={ self.random_noise:  batch_random_noise })
                self.writer.add_summary(summary, counter)

                errD_fake = self.discriminator_thief_loss.eval({ self.random_noise: batch_random_noise, self.y : batch_labels })
                errD_real = self.discriminator_police_loss.eval({ self.real_image: batch_images,  self.y : batch_labels })
                ## 변경
                # errG = self.generator_loss.eval({self.random_noise: batch_random_noise })
                errG = self.generator_loss.eval({self.random_noise: batch_random_noise,  self.y : batch_labels })

                ## Find cost value
                errD_class_real = self.discriminator_loss_class_real.eval({ self.real_image : batch_images, self.y : batch_labels })
                errG_class_fake = self.generator_loss_class_fake.eval({ self.real_image : batch_images, self.random_noise : batch_random_noise })
                accuracy = self.accuracy.eval({ self.real_image: batch_images, self.y: batch_labels})

                # global value --> checkpoint value
                counter += 1
                print("Epoch: [%2d] [%4d/%4d], d_loss: %.8f, g_loss: %.8f" % (epoch, index, batch_index, errD_fake + errD_real + errD_class_real, errG))
                print("Discriminator class acc: %.2f" % (accuracy))

                ## image save
                if np.mod(counter, 100) == 1:
                    try:
                        samples = self.sess.run(self.sampler, feed_dict={self.random_noise: sample_random_noise, self.real_image: sample_images, self.y : sample_labels })
                        save_images(samples, image_manifold_size(samples.shape[0]),'./{}/train_{:02d}_{:04d}.png'.format('samples', epoch, index))
                        print("[SAVE IMAGE]")
                    except Exception as e:
                        print("image save error! ", e)

                ## checkpoint save
                if np.mod(counter, 500) == 1:
                    print("[SAVE CHECKPOINT]")
                    checkpoint_save(self.sess, self.saver, checkpoint_dir_path, counter)

    ## discriminator
    def discriminator(self, input_, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables() # 데이터 공유하면
            #! padding -> SAME -> VALID => ops.py 파일에 있음
            discriminator_layer0 = lrelu(conv2d(input_, 32, name='d_h0_conv'))                                          # [256, 256, 3], 32 => (128, 128, 32)
            discriminator_layer1 = lrelu(batch_norm(conv2d(discriminator_layer0, 64, name='d_h1_conv'), 'd_bn1'))       # (?, 64, 64, 64)
            discriminator_layer2 = lrelu(batch_norm(conv2d(discriminator_layer1, 128, name='d_h2_conv'), 'd_bn2'))      # (?, 32, 32, 128)
            discriminator_layer3 = lrelu(batch_norm(conv2d(discriminator_layer2, 256, name='d_h3_conv'), 'd_bn3'))      # (?, 16, 16, 256)
            discriminator_layer4 = lrelu(batch_norm(conv2d(discriminator_layer3, 512, name='d_h4_conv'), 'd_bn4'))      # (?, 8, 8, 512)
            discriminator_layer5 = lrelu(batch_norm(conv2d(discriminator_layer4, 512, name='d_h5_conv'), 'd_bn5'))      # (?, 4, 4, 512)

            shape = np.product(discriminator_layer5.get_shape()[1:].as_list())                                          #
            discriminator_layer6 = tf.reshape(discriminator_layer5, [-1, shape])                                        #
            discriminator_output  = linear(discriminator_layer6, 1, 'd_ro_lin')                                         # (?, 1)

            discriminator_layer7 = lrelu(linear(discriminator_layer6, 1024, 'd_h8_lin'))                                #
            discriminator_layer8 = lrelu(linear(discriminator_layer7, 512, 'd_h9_lin'))                                 #
            discriminator_class_output = linear(discriminator_layer8, 27, 'd_co_lin')                                   #
            discriminator_class_output_softmax = tf.nn.softmax(discriminator_class_output)                              # (?, 27)

            return tf.nn.sigmoid(discriminator_output), discriminator_output, discriminator_class_output_softmax, discriminator_class_output

    ## generator
    def generator(self, random_noise):
        with tf.variable_scope("generator") as scope:
            generator_linear = linear(random_noise, 64 * 4 * 4 * 16, 'g_h0_lin')                                        # ([?, 100], 16,384])
            generator_reshape = tf.reshape(generator_linear, [-1, 4, 4, 64 * 16])                                       # (?, 4, 4, 1024)
            generator_input = tf.nn.relu(batch_norm(generator_reshape , 'g_bn0'))                                       # (?, 4, 4, 1024)

            generator_layer1 = deconv2d(generator_input, [self.batch_size, 8, 8, 64 * 16], name='g_layer1')             # (?, 8, 8, 1024)
            generator_layer1 = tf.nn.relu(batch_norm(generator_layer1, 'g_bn1'))                                        # (?, 8, 8, 1024)

            generator_layer2 = deconv2d(generator_layer1, [self.batch_size, 16, 16, 64 * 8], name='g_layer2')           # (?, 16, 16, 512)
            generator_layer2 = tf.nn.relu(batch_norm(generator_layer2, 'g_bn2'))                                        # (?, 16, 16, 512)

            generator_layer3 = deconv2d(generator_layer2, [self.batch_size, 32, 32, 64 * 4], name='g_layer3')           # (?, 32, 32, 256)
            generator_layer3 = tf.nn.relu(batch_norm(generator_layer3, 'g_bn3'))                                        # (?, 32, 32, 256)

            generator_layer4 = deconv2d(generator_layer3, [self.batch_size, 64, 64, 64 * 2], name='g_layer4')           # (?, 64, 64, 128)
            generator_layer4 = tf.nn.relu(batch_norm(generator_layer4, 'g_bn4'))                                        # (?, 64, 64, 128)

            generator_layer5 = deconv2d(generator_layer4, [self.batch_size, 128, 128, 64], name='g_layer5')             # (?, 128, 128, 64)
            generator_layer5 = tf.nn.relu(batch_norm(generator_layer5, 'g_bn5'))                                        # (?, 128, 128, 64)

            generator_output = deconv2d(generator_layer5, [self.batch_size, 256, 256, 3], name='g_output')              # (?, 256, 256, 3)
            generator_output = tf.nn.tanh(generator_output)                                                             # (?, 256, 256, 3)

            return generator_output                                                                                     # (?, 256, 256, 3)

    ## sampler
    def sampler(self, random_noise):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            sampler_linear = linear(random_noise, 64 * 4 * 4 * 16, 'g_h0_lin')                                          # ([?, 100], 16,384])
            sampler_reshape = tf.reshape(sampler_linear, [-1, 4, 4, 64 * 16])                                           # (?, 4, 4, 1024)
            sampler_input = tf.nn.relu(batch_norm(sampler_reshape, 'g_bn0', train=False))                               # (?, 4, 4, 1024)

            sampler_layer1 = deconv2d(sampler_input, [self.batch_size, 8, 8, 64 * 16], name='g_layer1')                 # (?, 8, 8, 1024)
            sampler_layer1 = tf.nn.relu(batch_norm(sampler_layer1, 'g_bn1', train=False))                               # (?, 8, 8, 1024)

            sampler_layer2 = deconv2d(sampler_layer1, [self.batch_size, 16, 16, 64 * 8], name='g_layer2')               # (?, 16, 16, 512)
            sampler_layer2 = tf.nn.relu(batch_norm(sampler_layer2, 'g_bn2', train=False))                               # (?, 16, 16, 512)

            sampler_layer3 = deconv2d(sampler_layer2, [self.batch_size, 32, 32, 64 * 4], name='g_layer3')               # (?, 32, 32, 256)
            sampler_layer3 = tf.nn.relu(batch_norm(sampler_layer3, 'g_bn3', train=False))                               # (?, 32, 32, 256)

            sampler_layer4 = deconv2d(sampler_layer3, [self.batch_size, 64, 64, 64 * 2], name='g_layer4')               # (?, 64, 64, 128)
            sampler_layer4 = tf.nn.relu(batch_norm(sampler_layer4, 'g_bn4', train=False))                               # (?, 64, 64, 128)

            sampler_layer5 = deconv2d(sampler_layer4, [self.batch_size, 128, 128, 64], name='g_layer5')                 # (?, 128, 128, 64)
            sampler_layer5 = tf.nn.relu(batch_norm(sampler_layer5, 'g_bn5', train=False))                               # (?, 128, 128, 64)

            sampler_output = deconv2d(sampler_layer5, [self.batch_size, 256, 256, 3], name='g_output')                  # (?, 256, 256, 3)
            sampler_output = tf.nn.tanh(sampler_output)                                                                 # (?, 256, 256, 3)

            return sampler_output                                                                                       # (?, 256, 256, 3)
