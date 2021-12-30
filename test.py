import argparse
import os.path as ops
import time
import os
import cv2
import numpy as np
import tensorflow as tf
import glob
import struct
import random
from networks import model
from tools.test_tools import *
from tensorflow.python.tools import inspect_checkpoint as chkp
from config_kp import cfg

def test_lanenet_image(video_path, sess):
    file_list = glob.glob(video_path)
    file_list = sorted(file_list)

    last_hm = np.zeros((22, 50))
    for index, full_path in enumerate(file_list):
        print(index)
        # if index % 4 > 0:
        #     continue
        os.makedirs(cfg.save_folder, exist_ok=True)

        save_name = cfg.save_folder + " %06d" %index

        image = cv2.imread(full_path)

        image = cv2.resize(image, (800, 450))
        image = image[98:, :, :]

        image = np.array(image, np.float32) / 255

        out_image = sess.run(infer, feed_dict={input_tensor: [image]})

        pred_hm = out_image[0][0] # 50 * 22
        pred_kp = out_image[1][0] # 50 * 22 * point_num
        pred_hm = pred_hm * 0.6 + last_hm * 0.4
        last_hm = pred_hm

        lane_list = extract_list(pred_hm, pred_kp, cfg.thr_extract)
        image_raw = Draw_lane(image.copy() * 255, lane_list)

        lane_list = lane_NMS(lane_list, cfg.thr_nms)
        image_nms = Draw_lane(image.copy() * 255, lane_list)
        cv2.imwrite(save_name + '_2nms.png', image_nms)

    return


if __name__ == '__main__':
    if cfg.gpu_num < 0:
        sess_config = tf.ConfigProto(device_count={'GPU':0})
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu_num)
        sess_config = tf.ConfigProto(allow_soft_placement=True)

    sess = tf.Session(config=sess_config)

    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, cfg.height, cfg.width, 3], name='input_tensor')

    net = model.Network(phase='test', reuse=False)

    results = net.inference(input_tensor=input_tensor, name='model')
    infer = results['predict']

    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
        saver.restore(sess=sess, save_path=cfg.weight_path)
        test_lanenet_image(cfg.input_path, sess)
    sess.close()
