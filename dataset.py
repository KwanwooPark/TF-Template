'''
DataLoader for Training In Tensorflow Template.
By Kwanwoo Park, 2022.
'''
import tensorflow as tf
import random
import cv2
import os
import numpy as np
import json
from config_kp import cfg

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'

class Datasets():
    def __init__(self, is_training):
        super(Datasets, self).__init__()
        self.is_training = is_training

    def Generator(self):
        for data in self.data_list:
            img_path, label = data
            if self.is_training:
                img_path = './data/train/' + img_path
            else:
                img_path = './data/val/' + img_path
            image = cv2.imread(img_path)
            image = image.astype(np.float32) / 255
            yield (image, label)

    def argument(self, image, label):
        image = tf.image.random_flip_left_right(image)
        # image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, 0.15)
        image = tf.image.random_contrast(image, 0.667, 1.5)
        image = tf.image.random_hue(image, 0.15)
        image = tf.image.random_saturation(image, 0.667, 1.5)
        image = tf.clip_by_value(image, 0, 1)

        label = tf.one_hot(label, depth=cfg.class_num, dtype=tf.float32)

        return image, label

    def shuffling(self):
        random.shuffle(self.data_list)

    def Generate_dataset(self, list_name):
        self.data_list = json.load(open(list_name))
        self.shuffling()
        dats_size = cfg.batch_size * len(cfg.gpus)
        dataset = tf.data.Dataset.from_generator(self.Generator, (tf.float32, tf.int32), (tf.TensorShape([cfg.height, cfg.width, 3]), tf.TensorShape([])))
        dataset = dataset.map(self.argument)
        dataset = dataset.shuffle(dats_size)
        dataset = dataset.batch(dats_size)
        dataset = dataset.repeat()

        iterator = dataset.make_initializable_iterator()
        total_iteration = int(len(self.data_list) / (dats_size))
        return iterator, total_iteration

### This is for test. ###
if __name__ == '__main__':
    with tf.Graph().as_default(), tf.device('/device:CPU:0'):
        datasets = Datasets(is_training=True)
        train_iterator, total_iteration = datasets.Generate_dataset(cfg.train_list)
        tf_images, tf_labels = train_iterator.get_next()
        tf_images = tf.split(tf_images, len(cfg.gpus), axis=0)
        tf_labels = tf.split(tf_labels, len(cfg.gpus), axis=0)
        data_lists = []
        for i in cfg.gpus:
            with tf.device('/gpu:%d' % i):
                tf_pred = tf.reduce_mean(tf_images[i], axis=3)
                data_lists.append([tf_images[i], tf_labels[i], tf_pred])

        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        sess_config.gpu_options.allocator_type = 'BFC'
        sess = tf.Session(config=sess_config)

        with sess.as_default():
            sess.run(train_iterator.initializer)
            datasets.shuffling()
            while 1:
                data_loaded = sess.run(data_lists)
                for i in range(len(cfg.gpus)):
                    datas = data_loaded[i]
                    images, labels, preds = datas
                    for b in range(cfg.batch_size):
                        image = images[b]
                        label = labels[b]
                        pred = preds[b]
                        print(label)
                        print(pred)
                        print()
                        cv2.imshow("preview", image)
                        cv2.waitKey(1)
                print()
                print()
                print()
