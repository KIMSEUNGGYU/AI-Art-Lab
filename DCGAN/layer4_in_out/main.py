import tensorflow as tf
from DCGAN import *

def main(_):
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth=True

    with tf.Session(config=run_config) as sess:
        dcgan = DCGAN(sess)
        dcgan.build_model()
        dcgan.train()


if __name__ == '__main__':
    tf.app.run()
