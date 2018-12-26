#############################################
# Date          : 2018.03.25
# Programmer    : Seounggyu Kim
# description   : 모델(CAN) 관련 함수
# Update Date   : 2018.04.04
# Update        : CAN fuction
#############################################

import os
import tensorflow as tf
from utils import *
import re




def batch_norm(inputs, scope_name, train=True ):
    return tf.contrib.layers.batch_norm(inputs, decay=0.9, updates_collections=None,epsilon=1e-5, scale=True, is_training = train, scope = scope_name)

def conv2d(input_, output_dim, stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
        conv_w = tf.get_variable('conv_w', [4, 4, input_.get_shape()[-1], output_dim],initializer=tf.truncated_normal_initializer(stddev=stddev))
        # conv = tf.nn.conv2d(input_, conv_w, strides=[1, 2, 2, 1], padding='SAME') # 변경
        conv = tf.nn.conv2d(input_, conv_w, strides=[1, 2, 2, 1], padding='SAME')
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        return conv

def deconv2d(input_, output_shape, stddev=0.02, name="deconv2d"):
    with tf.variable_scope(name):
        input_shape = input_.get_shape().as_list()
        deconv_w = tf.get_variable('deconv_w', [5, 5, output_shape[-1], input_shape[-1]],initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(input_, deconv_w, output_shape=output_shape, strides=[1, 2, 2, 1])
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        return deconv

############################# model #############################
def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

def linear(input_, output_size, scope=None, stddev=0.02,):
    shape = input_.get_shape().as_list()
    with tf.variable_scope(scope or "Linear"):
        matrix
            = tf.get_variable("Matrix", [shape[1], output_size],
                tf.float32, tf.random_normal_initializer(stddev=stddev))
        bias
        = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(0.0))

    return tf.matmul(input_, matrix) + bias
############################# Checkpoint #############################
def checkpoint_save(sess, saver, checkpoint_path_dir, counter):
    model_name = "DCGAN.model"
    # checkpoint_path_dir = os.path.join('checkpoint', 'wikiart')
    if not os.path.exists(checkpoint_path_dir):
        os.makedirs(checkpoint_path_dir)
    saver.save(sess, os.path.join(checkpoint_path_dir, model_name),
         global_step=counter)

def checkpoint_load(sess, saver, checkpoint_dir, checkpoint_dir_model):
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, checkpoint_dir_model)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
      # 체크포인트 파일 counter 부분 가져오기
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0

############################# get label #############################
def get_y(sample_inputs, label_dim, label_dict):
    ret = []
    for sample in sample_inputs:
        _, _, _, lab_str, _ = sample.split('/', 4)
        ret.append(np.eye(label_dim)[np.array(label_dict[lab_str])])

    return ret
