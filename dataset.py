import tensorflow as tf
import random
import cv2
import numpy as np
import os
import json
from config import cfg
import glob
import warnings
warnings.simplefilter('ignore', np.RankWarning)

def make_lane_image(img, hm, os, kp):
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
        cv2.circle(img, (int(Center_w + 0.5), int(Center_h+ 0.5)), 5, color=color_list[count % 6], thickness=2)
        points = kp[Center_h_int, Center_w_int, :, :]
        for p in range(16):
            h = int(points[p, 0] + Center_h + 0.5)
            w = int(points[p, 1] + Center_w + 0.5)
            cv2.circle(img, (w, h), 3, color=color_list[count % 6], thickness=-1)

    return img


def gaussian_radius(det_size, min_overlap=0.7):
  height, width = det_size

  a1  = 1
  b1  = (height + width)
  c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
  sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
  r1  = (b1 + sq1) / 2

  a2  = 4
  b2  = 2 * (height + width)
  c2  = (1 - min_overlap) * width * height
  sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
  r2  = (b2 + sq2) / 2

  a3  = 4 * min_overlap
  b3  = -2 * min_overlap * (height + width)
  c3  = (min_overlap - 1) * width * height
  sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
  r3  = (b3 + sq3) / 2
  return min(r1, r2, r3)

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    y, x = center[0], center[1]

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def lane_process(lanes, flip):
    label_hm = np.zeros((int(cfg.height / cfg.rate), int(cfg.width / cfg.rate)), np.float32)
    label_os = np.zeros((int(cfg.height / cfg.rate), int(cfg.width / cfg.rate), 2), np.float32)
    label_kp = np.zeros((int(cfg.height / cfg.rate), int(cfg.width / cfg.rate), 16, 2), np.float32) + 100
    for lane in lanes:
        if flip:
            CH = lane[0]
            CW = cfg.width - lane[1]
            length_H = lane[2]
            length_W = lane[3]
            point_H = np.array(lane[4])
            point_W = cfg.width - np.array(lane[5])
        else:
            CH = lane[0]
            CW = lane[1]
            length_H = lane[2]
            length_W = lane[3]
            point_H = np.array(lane[4])
            point_W = np.array(lane[5])

        CH_int = int(CH / cfg.rate)
        CW_int = int(CW / cfg.rate)

        radius = gaussian_radius((length_H / cfg.rate,  length_W / cfg.rate))
        radius = max(0, int(radius + 0.5))
        draw_gaussian(label_hm, (CH_int, CW_int), radius)

        label_os[CH_int, CW_int, 0] = CH - (CH_int * cfg.rate)
        label_os[CH_int, CW_int, 1] = CW - (CW_int * cfg.rate)

        label_kp[CH_int, CW_int, :, 0] = np.array(point_H) - CH
        label_kp[CH_int, CW_int, :, 1] = np.array(point_W) - CW

    return label_hm, label_os, label_kp

class Datasets():
    def __init__(self):
        super(Datasets, self).__init__()

    def read_image(self, img_path):
        img_path = img_path.decode("utf-8")
        img_path = './dataset/' + img_path

        label_path = img_path.replace('/images/', '/jsons/').replace('.png', '.json')
        image = cv2.imread(img_path)
        lanes = json.load(open(label_path))
        image = image.astype(np.float32) / 255
        flip = np.random.randint(2) > 0

        hm, os, kp = lane_process(lanes, flip)
        if flip:
            image = np.flip(image, axis=1)

        return image, hm, os, kp

    def argument(self, image, hm, os, kp):
        image.set_shape([cfg.height, cfg.width, 3])
        hm.set_shape([int(cfg.height / cfg.rate), int(cfg.width / cfg.rate)])
        os.set_shape([int(cfg.height / cfg.rate), int(cfg.width / cfg.rate), 2])
        kp.set_shape([int(cfg.height / cfg.rate), int(cfg.width / cfg.rate), 16, 2])

        image = tf.image.random_brightness(image, 0.15)
        image = tf.image.random_contrast(image, 0.667, 1.5)
        image = tf.image.random_hue(image, 0.15)
        image = tf.image.random_saturation(image, 0.667, 1.5)
        image = tf.clip_by_value(image, 0, 1)
        image = image

        return image, hm, os, kp

    def shuffling(self):
        random.shuffle(self.image_list)

    def Generate_dataset(self, list_name):
        self.image_list = json.load(open('./dataset/' + list_name))
        random.shuffle(self.image_list)
        total_iteration = int(len(self.image_list) / cfg.batch_size)
        dataset = tf.data.Dataset.from_tensor_slices((self.image_list))
        dataset = dataset.map(lambda image_list: tuple(tf.py_func(self.read_image, [image_list], [tf.float32, tf.float32, tf.float32, tf.float32])))
        dataset = dataset.map(self.argument)
        dataset = dataset.repeat()
        dataset = dataset.batch(cfg.batch_size)

        iterator = dataset.make_initializable_iterator()
        return iterator, total_iteration


if __name__ == '__main__':
    datasets = Datasets()
    train_iterator, total_iteration = datasets.Generate_dataset('test_list.json')
    tf_images, tf_hm, tf_os, tf_kp = train_iterator.get_next()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # sess_config = tf.ConfigProto(device_count={'GPU':0})
    sess_config = tf.ConfigProto()

    sess = tf.Session(config=sess_config)

    sess.run(train_iterator.initializer)
    count = 1
    for i in range(1000):
        train_image, train_hm, train_os, train_kp = sess.run([tf_images, tf_hm, tf_os, tf_kp])
        for b in range(cfg.batch_size):
            image = train_image[b]
            hm = train_hm[b]
            os = train_os[b]
            kp = train_kp[b]

            pred_img = make_lane_image(image * 255, hm, os, kp)
            image = np.array(image * 255, np.uint8)
            cv2.imwrite('./tmp/%06d.png'%count, image)
            cv2.imwrite('./tmp/%06d_.png' % count, pred_img)
            count += 1
            print(count)

    # image_list = glob.glob('./dataset/images/*')
    # label_list = []
    # for text in image_list:
    #     text = text.replace('/images/', '/jsons/').replace('.png', '.json')
    #     label_list.append(text)
    # count = 1
    # for i in range(10000):
    #     img_path = image_list[i]
    #     json_path = label_list[i]
    #     image = cv2.imread(img_path)
    #     image = np.array(image, np.float) / 255
    #     lanes = json.load(open(json_path))
    #     hm, os, kp = lane_process(lanes)
    #
    #     pred_img = make_lane_image(image, hm, os, kp)
    #     cv2.imwrite('./tmp/%06d_s1.png' % count, image * 255)
    #     print(count)
    #     # image = np.array(image * 255, np.uint8)
    #     # cv2.imwrite('./tmp/%06d_s1.png'%count, image)
    #     # cv2.imwrite('./tmp/%06d_s16.png' % count, pred_img)
    #     count += 1