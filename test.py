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
from model import model
from tools.infer_tools_v2 import *
from tensorflow.python.tools import inspect_checkpoint as chkp

row = 480
col = 800

def test_lanenet_image(video_path, sess):
    file_list = glob.glob(video_path)
    file_list = sorted(file_list)
    tracker_list = []

    for index, full_path in enumerate(file_list):
        img_name = full_path.split('/')[-1]
        folder_name = full_path.split('/')[-2]

        save_folder = './results/' + folder_name + '/'
        os.makedirs(save_folder, exist_ok=True)\
        # os.makedirs(save_folder + '/img/', exist_ok=True)

        image = cv2.imread(full_path)
        image_org = image.copy()

        # image_total = np.zeros((480 * 2, 800 * 2, 3), np.uint8)

        # image = image[:, 60: -60, :]
        print(index)
        # image = cv2.resize(image, (800, 480))
        image = np.array(image, np.float32) / 255

        out_image = sess.run(infer, feed_dict={input_tensor: [image]})

        pred_hm = out_image[0][0]
        pred_os = out_image[1][0]
        pred_kp = out_image[2][0]
        img_name = img_name.replace('.jpg', '').replace('.png', '')
        save_to_bin(pred_hm, save_folder + img_name + '_hm.bin')
        save_to_bin(pred_os, save_folder + img_name + '_os.bin')
        save_to_bin(pred_kp, save_folder + img_name + '_kp.bin')
        cv2.imwrite(save_folder + img_name + '_0ORG.png', image_org)

        lane_list = extract_list(pred_hm, pred_os, pred_kp, 0.3)
        image_RAW = np.array(Draw_lane(image.copy() * 255, lane_list), dtype=np.uint8)
        cv2.imwrite(save_folder + img_name + '_1RAW.png', image_RAW)

        lane_list = lane_NMS(lane_list, pred_hm, 30)

        image_NMS = np.array(Draw_lane(image.copy() * 255, lane_list), dtype=np.uint8)
        cv2.imwrite(save_folder + img_name + '_2NMS.png', image_NMS)

        tracker_list = tracking_v2(tracker_list, lane_list, 50, 0.5)
        image_TRA = np.array(Draw_lane(image.copy() * 255, tracker_list), dtype=np.uint8)
        cv2.imwrite(save_folder + img_name + '_3TRA.png', image_TRA)

        # image_total[0:480, 0:800, :] = image * 255
        # image_total[480:960, 0:800, :] = image_NMS
        # image_total[0:480, 800:1600, :] = image_RAW
        # image_total[480:960, 800:1600, :] = image_TRA

        # cv2.imwrite(save_folder + img_name, image_total)

        # os.makedirs(save_folder + '/xml/', exist_ok=True)
        # lane_list_save = lane_revision(lane_list.copy(), (0, 60), 2.25)
        # save_lane(folder_name, img_name, full_path, save_folder + '/xml/', lane_list_save)

    return

# class LANE():
#     def __init__(self):
#         self.ch = 0
#         self.cw = 0
#         self.ch_int = 0
#         self.cw_int = 0
#
#         self.h = np.zeros((16,), np.float)
#         self.w = np.zeros((16,), np.float)
#         self.p = np.zeros((4,), np.float)
#         self.socre = 0
#         self.miss = 0
#         self.catch = 1
#         self.appear = False
if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    # sess_config = tf.ConfigProto(device_count={'GPU':0})
    sess = tf.Session(config=sess_config)

    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, row, col, 3], name='input_tensor')

    net = model.Network(phase='test', reuse=False)

    results = net.inference(input_tensor=input_tensor, name='model')
    infer = results['predict']

    model_path = './pretrained/weights.ckpt-6251'
    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
        saver.restore(sess=sess, save_path=model_path)
        test_lanenet_image('./testset/R9/lane*/*.*', sess)
    sess.close()
