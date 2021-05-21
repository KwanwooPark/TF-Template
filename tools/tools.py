import numpy as np
import cv2
import tensorflow as tf
import random
import os
import cv2
import os.path as ops
import glob
from numpy.lib.stride_tricks import as_strided
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from config import cfg


color_list = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 128, 0), (0, 255, 128), (128, 0, 255)]
for r in range(3):
    for g in range(3):
        for b in range(3):
            color_list.append((r * 62 + 127, g * 62 + 127, b * 62 + 127))
del color_list[0]
random.shuffle(color_list)

def pool2d(A, kernel_size, stride, padding, pool_mode='max'):
    # Padding
    A = np.pad(A, padding, mode='constant')

    # Window view of A
    output_shape = ((A.shape[0] - kernel_size)//stride + 1,
                    (A.shape[1] - kernel_size)//stride + 1)
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(A, shape = output_shape + kernel_size,
                        strides = (stride*A.strides[0],
                                   stride*A.strides[1]) + A.strides)
    A_w = A_w.reshape(-1, *kernel_size)

    # Return the result of pooling
    if pool_mode == 'max':
        return A_w.max(axis=(1,2)).reshape(output_shape)
    elif pool_mode == 'avg':
        return A_w.mean(axis=(1,2)).reshape(output_shape)

def make_point_image_for_prediction(img, hm, os, kp, threshold=0.3, need_nms=True): # w * h * 8 * 2
    color_list = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 128, 0), (0, 255, 128), (128, 0, 255)]
    if need_nms:
        hm_max = pool2d(hm, kernel_size=11, stride=1, padding=5, pool_mode='max')
        keep = (hm_max == hm).astype(np.float)
        hm = hm * keep

    inds = np.where(hm >= threshold)
    Hs = inds[0]
    Ws = inds[1]
    count = 0
    for ind in range(Hs.size):
        count += 1
        Center_h_int = Hs[ind]
        Center_w_int = Ws[ind]
        Center_h = Center_h_int * cfg.rate + os[Center_h_int, Center_w_int, 0]
        Center_w = Center_w_int * cfg.rate + os[Center_h_int, Center_w_int, 1]
        cv2.circle(img, (int(Center_w + 0.5), int(Center_h + 0.5)), 5, color=color_list[count % 6], thickness=2)
        points = kp[Center_h_int, Center_w_int, :, :]

        for p in range(16):
                h = int(points[p, 0] + Center_h + 0.5)
                w = int(points[p, 1] + Center_w + 0.5)
                cv2.circle(img, (w, h), 3, color=color_list[count % 6], thickness=-1)

    return img

def make_GT_image(img, hm, os, kp):
    color_list = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 128, 0), (0, 255, 128), (128, 0, 255)]
    inds = np.where(hm == 1)
    Hs = inds[0]
    Ws = inds[1]
    count = 0
    for ind in range(Hs.size):
        count += 1
        Center_h_int = Hs[ind]
        Center_w_int = Ws[ind]
        Center_h = Center_h_int * cfg.rate + os[Center_h_int, Center_w_int, 0]
        Center_w = Center_w_int * cfg.rate + os[Center_h_int, Center_w_int, 1]
        cv2.circle(img, (int(Center_w + 0.5), int(Center_h + 0.5)), 5, color=color_list[count % 6], thickness=2)
        points = kp[Center_h_int, Center_w_int, :, :]
        for p in range(16):
            h = int(points[p, 0] + Center_h + 0.5)
            w = int(points[p, 1] + Center_w + 0.5)
            cv2.circle(img, (w, h), 3, color=color_list[count % 6], thickness=-1)
    return img

def load_val_images(image_path):
    image = cv2.imread(image_path)
    image = np.array(image, np.float32) / 255
    return image


def record_results_val(out_image, index):
    out_image = out_image[0]
    image = out_image[0][0]
    pred_hm = out_image[1][0][0]
    pred_os = out_image[1][1][0]
    pred_kp = out_image[1][2][0]

    image = np.array(image * 255, dtype=np.uint8)
    cv2.imwrite('./tmp/val_%03d_1image.png' % (index), image)

    pred = make_point_image_for_prediction(image, pred_hm, pred_os, pred_kp, threshold=0.3, need_nms=True)
    cv2.imwrite('./tmp/val_%03d_2pred.png' % (index), pred)
    return


def record_results(out_lists, save_dir='./tmp/'):
    os.makedirs(save_dir, exist_ok=True)

    images = []
    label_hms = []
    label_oss = []
    label_kps = []

    pred_hms = []
    pred_kps = []
    pred_oss = []

    for out_image in out_lists:
        images.append(out_image[0])
        label_hms.append(out_image[1])
        label_oss.append(out_image[2])
        label_kps.append(out_image[3])
        pred_hms.append(out_image[4][0])
        pred_oss.append(out_image[4][1])
        pred_kps.append(out_image[4][2])

    images = np.concatenate(images, 0)
    label_hms = np.concatenate(label_hms, 0)
    label_oss = np.concatenate(label_oss, 0)
    label_kps = np.concatenate(label_kps, 0)
    pred_hms = np.concatenate(pred_hms, 0)
    pred_oss = np.concatenate(pred_oss, 0)
    pred_kps = np.concatenate(pred_kps, 0)

    for index, image in enumerate(images):
        image = images[index]
        label_hm = label_hms[index]
        label_os = label_oss[index]
        label_kp = label_kps[index]
        pred_hm = pred_hms[index]
        pred_os = pred_oss[index]
        pred_kp = pred_kps[index]

        image = np.array(image * 255, dtype=np.uint8)
        cv2.imwrite(save_dir + '/%03d_0image.png' % (index), image)

        label = make_GT_image(image.copy(), label_hm, label_os, label_kp)
        cv2.imwrite(save_dir + '/%03d_1label.png' % (index), label)

        pred = make_point_image_for_prediction(image.copy(), pred_hm, pred_os, pred_kp, threshold=0.3, need_nms=True)
        cv2.imwrite(save_dir + '/%03d_2pred.png' % (index), pred)

        label_hm = np.array(label_hm * 255, np.uint8)
        label_hm = cv2.resize(label_hm, (cfg.width, cfg.height), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(save_dir + '/%03d_3labelhm.png' % (index), label_hm)

        pred_hm = np.array(pred_hm * 255, np.uint8)
        pred_hm = cv2.resize(pred_hm, (cfg.width, cfg.height), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(save_dir + '/%03d_4predhm.png' % (index), pred_hm)

    return

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads
