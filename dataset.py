'''
DataLoader for Training In Tensorflow Template
By Kwanwoo Park, 2022.
'''
import tensorflow as tf
import random
import cv2
import numpy as np
import os
import json
import glob
from config_kp import cfg

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
    datasets = Datasets()
    train_iterator, total_iteration = datasets.Generate_dataset('train_list.json')
    tf_images, tf_hm, tf_kp = train_iterator.get_next()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    sess_config = tf.ConfigProto()

    sess = tf.Session(config=sess_config)

    sess.run(train_iterator.initializer)
    count = 1
    for i in range(1000):
        train_image, train_hm, train_kp = sess.run([tf_images, tf_hm, tf_kp])
        for b in range(cfg.batch_size):
            image = train_image[b]
            hm = train_hm[b]
            kp = train_kp[b]

            pred_img = make_lane_image(image * 255, hm, kp)
            cv2.imwrite('./tmp/%06d_.png' % count, pred_img)

            # image = np.array(image * 255, np.uint8)
            # cv2.imwrite('./tmp/%06d.png'%count, image)

            hm = cv2.resize(hm, (800, 352))
            cv2.imwrite('./tmp/%06d.png' % count, hm*255)
            count += 1
            print(count)

