import tensorflow as tf
import random
import cv2
import numpy as np
import os
import json
from config_kp import cfg
import glob
import warnings
warnings.simplefilter('ignore', np.RankWarning)


def make_lane_image(img, hm, kp):
    color_list = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 128, 0), (0, 255, 128), (128, 0, 255)]
    inds = np.where(hm == 1)

    Hs = inds[0]
    Ws = inds[1]

    count = 0
    for ind in range(Hs.size):
        count += 1
        TH_int = Hs[ind]
        TW_int = Ws[ind]
        # if hm[TH_int, TW_int] == 1:
        #     continue
        hs = kp[TH_int, TW_int, 0:cfg.point_num] + TH_int + 0.5
        ws = kp[TH_int, TW_int, cfg.point_num:cfg.point_num*2] + TW_int + 0.5
        for p in range(cfg.point_num):
            h = int(hs[p] * cfg.rate + 0.5)
            w = int(ws[p] * cfg.rate + 0.5)
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

def gaussian2D(shape, sigma_x=1, sigma_y=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-((x * x) / (2 * sigma_x * sigma_x) + (y * y) / (2 * sigma_y * sigma_y)))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center, radius_x, radius_y, k=1):
    diameter_x = 2 * radius_x + 1
    diameter_y = 2 * radius_y + 1

    gaussian = gaussian2D((diameter_y, diameter_x), sigma_x=diameter_x / 6, sigma_y=diameter_y / 6)

    y, x = center[0], center[1]

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius_x), min(width - x, radius_x + 1)
    top, bottom = min(y, radius_y), min(height - y, radius_y + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius_y - top:radius_y + bottom, radius_x - left:radius_x + right]
    hh = np.array([])
    ww = np.array([])

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        gaussian_ind = np.where(masked_gaussian > masked_heatmap)
        hh = gaussian_ind[0] + y - top
        ww = gaussian_ind[1] + x - left

        np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)
    return heatmap, hh, ww


def lane_process(lanes, flip):
    label_hm = np.zeros((int(cfg.height / cfg.rate), int(cfg.width / cfg.rate)), np.float32)
    label_kp = np.zeros((int(cfg.height / cfg.rate), int(cfg.width / cfg.rate), cfg.point_num * 2), np.float32)
    for lane in lanes:
        TH, TW = lane[0]
        h = np.array(lane[1])
        w = np.array(lane[2])

        if flip:
            w = cfg.width - w - 1
            TW = cfg.width - TW - 1

        w_max = np.max(w)
        w_min = np.min(w)

        length_H = h[-1] - h[0]
        length_W = w_max - w_min

        TH /= cfg.rate
        TW /= cfg.rate

        TH_int = int(TH)
        TW_int = int(TW)

        h = h[cfg.point_pos]
        w = w[cfg.point_pos]

        # radius = gaussian_radius((length_H / (cfg.rate),  length_W / (cfg.rate)))
        radius_x = max(0, int(length_W/96 + 0.5))
        radius_y = max(0, int(length_H/32 + 0.5))

        label_hm, hh, ww = draw_gaussian(label_hm, (TH_int, TW_int), radius_x, radius_y)

        for indd in range(hh.size):
            label_kp[hh[indd], ww[indd], 0:cfg.point_num] = (h / cfg.rate) - (hh[indd] + 0.5)
            label_kp[hh[indd], ww[indd], cfg.point_num:cfg.point_num * 2] = (w / cfg.rate) - (ww[indd] + 0.5)

        label_kp[TH_int, TW_int, 0:cfg.point_num] = (h / cfg.rate) - (TH_int + 0.5)
        label_kp[TH_int, TW_int, cfg.point_num:cfg.point_num * 2] = (w / cfg.rate) - (TW_int + 0.5)

    return label_hm, label_kp

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

        hm, kp = lane_process(lanes, flip)
        if flip:
            image = np.flip(image, axis=1)

        return image, hm, kp

    def argument(self, image, hm, kp):
        image.set_shape([cfg.height, cfg.width, 3])
        hm.set_shape([int(cfg.height / cfg.rate), int(cfg.width / cfg.rate)])
        kp.set_shape([int(cfg.height / cfg.rate), int(cfg.width / cfg.rate), cfg.point_num * 2])

        image = tf.image.random_brightness(image, 0.15)
        image = tf.image.random_contrast(image, 0.667, 1.5)
        image = tf.image.random_hue(image, 0.15)
        image = tf.image.random_saturation(image, 0.667, 1.5)
        image = tf.clip_by_value(image, 0, 1)
        image = image

        return image, hm, kp

    def shuffling(self):
        random.shuffle(self.image_list)

    def Generate_dataset(self, list_name):
        self.image_list = json.load(open('./dataset/' + list_name))
        random.shuffle(self.image_list)
        total_iteration = int(len(self.image_list) / cfg.batch_size)
        dataset = tf.data.Dataset.from_tensor_slices((self.image_list))
        dataset = dataset.map(lambda image_list: tuple(tf.py_func(self.read_image, [image_list], [tf.float32, tf.float32, tf.float32])))
        dataset = dataset.map(self.argument)
        dataset = dataset.repeat()
        dataset = dataset.batch(cfg.batch_size)

        iterator = dataset.make_initializable_iterator()
        return iterator, total_iteration


if __name__ == '__main__':
    # datasets = Datasets()
    # train_iterator, total_iteration = datasets.Generate_dataset('train_list.json')
    # tf_images, tf_hm, tf_kp = train_iterator.get_next()
    #
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # sess_config = tf.ConfigProto()
    #
    # sess = tf.Session(config=sess_config)
    #
    # sess.run(train_iterator.initializer)
    # count = 1
    # for i in range(1000):
    #     train_image, train_hm, train_kp = sess.run([tf_images, tf_hm, tf_kp])
    #     for b in range(cfg.batch_size):
    #         image = train_image[b]
    #         hm = train_hm[b]
    #         kp = train_kp[b]
    #
    #         pred_img = make_lane_image(image * 255, hm, kp)
    #         cv2.imwrite('./tmp/%06d_.png' % count, pred_img)
    #
    #         # image = np.array(image * 255, np.uint8)
    #         # cv2.imwrite('./tmp/%06d.png'%count, image)
    #
    #         hm = cv2.resize(hm, (800, 352))
    #         cv2.imwrite('./tmp/%06d.png' % count, hm*255)
    #         count += 1
    #         print(count)
    #
    image_list = json.load(open('./dataset/' + "train_list.json"))
    label_list = []
    for text in image_list:
        text = text.replace('/images/', '/jsons/').replace('.png', '.json')
        label_list.append(text)
    count = 1
    # for i in range(10000):
    #     img_path = "./dataset/" + image_list[i]
    #     json_path = "./dataset/" + label_list[i]
    #     image = cv2.imread(img_path)
    #     image = np.array(image, np.float) / 255
    #     lanes = json.load(open(json_path))
    #     hm, kp = lane_process(lanes, False)
    #
    #     pred_img = make_lane_image(image, hm, kp)
    #     cv2.imwrite('./tmp/%06d_s1.png' % count, image * 255)
    #     print(count)
    #     # image = np.array(image * 255, np.uint8)
    #     # cv2.imwrite('./tmp/%06d_s1.png'%count, image)
    #     # cv2.imwrite('./tmp/%06d_s16.png' % count, pred_img)
    #     count += 1

    for i, a in enumerate(label_list):
        json_path = "./dataset/" + label_list[i]
        json_path = "./dataset/" + "./Curves/jsons/045847.json"

        print(label_list[i])
        lanes = json.load(open(json_path))
        check_json(lanes)