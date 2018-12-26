#############################################
# Date          : 2018.03.25
# Programmer    : Seounggyu Kim
# description   : main
# Update Date   : 2018.04.19
# Update        : CAN DCGAN 모델 실행 코드 추가
#############################################


import tensorflow as tf
from DCGAN import *

def main(_):
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth=True

    with tf.Session(config=run_config) as sess:
        can = CAN(sess)
        can.build_model()
        can.train()

        # dcgan = DCGAN(sess)
        # dcgan.build_model()
        # dcgan.train()

if __name__ == '__main__':
    tf.app.run()
