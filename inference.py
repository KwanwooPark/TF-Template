'''
Inference Code In Tensorflow Template
By Kwanwoo Park, 2022.
'''

import os
import cv2
import numpy as np
import tensorflow as tf
import glob
from model import base
from config_kp import cfg

def test_lanenet_image(sess):
    file_list = glob.glob(cfg.input_path)
    file_list = sorted(file_list)

    for index, full_path in enumerate(file_list):
        print(index)
        os.makedirs(cfg.save_folder, exist_ok=True)
        save_name = cfg.save_folder + " %06d" %index

        image = cv2.imread(full_path)
        image = cv2.resize(image, (cfg.width, cfg.height))
        image = np.array(image, np.float32) / 255

        out_image = sess.run(infer, feed_dict={input_tensor: [image]})
    return


if __name__ == '__main__':
    if len(cfg.gpus) == 0:
        sess_config = tf.ConfigProto(device_count={'GPU':0})
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpus)
        sess_config = tf.ConfigProto(allow_soft_placement=True)

    sess = tf.Session(config=sess_config)
    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, cfg.height, cfg.width, 3], name='input_tensor')
    net = base.Model(phase='test', reuse=False)
    results = net.inference(input_tensor=input_tensor, name='model')
    infer = results['pred']

    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
        saver.restore(sess=sess, save_path=cfg.load_weight_path)
        test_lanenet_image(sess)
    sess.close()
