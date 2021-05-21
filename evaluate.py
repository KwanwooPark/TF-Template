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
from tools.infer_tools import *
from tensorflow.python.tools import inspect_checkpoint as chkp
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from config import cfg
from dataset import Datasets

def test_lanenet_image(sess):
    total_TP = 0
    total_FP = 0
    total_FN = 0
    count = 0
    for _ in range(val_iteration):
        out_image, raw_images, hm_labels, os_labels, kp_labels = sess.run([infer, image_tensor, hm_tensor, os_tensor, kp_tensor])
        for i in range(cfg.batch_size):
            pred_hm = out_image[0][i]
            pred_os = out_image[1][i]
            pred_kp = out_image[2][i]
            raw_image = raw_images[i]
            hm_label = hm_labels[i]
            os_label = os_labels[i]
            kp_label = kp_labels[i]

            lane_list = extract_list(pred_hm, pred_os, pred_kp, 0.55)

            lane_list = lane_sorting(lane_list)
            # image_raw = Draw_lane(image.copy() * 255, lane_list)
            # image_raw = np.array(image_raw, dtype=np.uint8)
            lane_list = lane_NMS(lane_list, 50)
            # image_NMS = Draw_lane(raw_image.copy() * 255, lane_list)
            # image_NMS = np.array(image_NMS, dtype=np.uint8)

            gt_list = extract_list(hm_label, os_label, kp_label, 0.9999)
            # image_GT = Draw_lane(raw_image.copy() * 255, gt_list)
            # image_GT = np.array(image_GT, dtype=np.uint8)

            count += 1
            # cv2.imwrite('./eval/%06d_pred.png' % (count), image_NMS)
            # cv2.imwrite('./eval/%06d_gt.png' % (count), image_GT)

            TP, FP, FN = evaluate(gt_list, lane_list, 19)
            total_TP += TP
            total_FN += FN
            total_FP += FP
        Precision = total_TP / (total_TP + total_FP)
        Recall = total_TP / (total_TP + total_FN)
        F1 = (2 * Precision * Recall) / (Precision + Recall)
        print(Precision, Recall, F1)

    return


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    # sess_config = tf.ConfigProto(device_count={'GPU':0})
    sess = tf.Session(config=sess_config)

    net = model.Network(phase='test', reuse=False)

    val_datasets = Datasets()
    val_iterator, val_iteration = val_datasets.Generate_dataset('val_list.json')
    image_tensor, hm_tensor, os_tensor, kp_tensor = val_iterator.get_next()
    results = net.inference(input_tensor=image_tensor, name='model')
    infer = results['predict']

    model_path = './pretrained/weights.ckpt-6251'
    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        sess.run(val_iterator.initializer)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
        saver.restore(sess=sess, save_path=model_path)
        test_lanenet_image(sess)
    sess.close()

