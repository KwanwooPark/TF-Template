import argparse
import math
import os
import os.path as ops
import time
import random
import cv2
import glog as log
import numpy as np
import tensorflow as tf
import shutil
import glob
from dataset import Datasets
from tensorflow.python.tools import inspect_checkpoint as chkp
from model import model
from tools.tools import average_gradients
from tools.tools import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--num_gpu', type=int, default=[0, 1, 2, 3])
    parser.add_argument('-e', '--train_epochs', type=int, default=1000)

    return parser.parse_args()


def train():
    with tf.Graph().as_default(), tf.device('/device:CPU:0'):
        train_epochs = args.train_epochs
        weights_path = cfg.weight_path

        train_tensor_list = []
        grads_list = []
        train_summary_list = []

        gs_tensor = tf.get_variable(
            'global_step', [], initializer=tf.constant_initializer(0.),
            trainable=False)

        lr_tensor = tf.train.cosine_decay(0.000010, gs_tensor, 100000, alpha=0.05)
        lr_summary = tf.summary.scalar(name='LR', tensor=lr_tensor)
        train_summary_list.append(lr_summary)

        optimizer = tf.train.MomentumOptimizer(lr_tensor, momentum=0.9, use_nesterov=True)

        print("######################## DRAW GRAPH OF TRAIN########################")
        train_net = model.Network(phase='tra', reuse=False)

        hmloss_sum = tf.constant(0.0, tf.float32)
        osloss_sum = tf.constant(0.0, tf.float32)
        kploss_sum = tf.constant(0.0, tf.float32)
        ttloss_sum = tf.constant(0.0, tf.float32)

        dataset_list = []
        iterator_list =[]

        for i in args.num_gpu:
            with tf.device('/gpu:%d' %i):
                print("######################## LOAD DATASET ########################")
                datasets = Datasets()
                dataset_list.append(datasets)

                iterator_train, total_iteration = datasets.Generate_dataset('train_list.json')
                iterator_list.append(iterator_train)

                image_tensor, hm_tensor, os_tensor, kp_tensor = iterator_train.get_next()
                '''
                image_tensor : batch * row * col * 3
                hm_tensor    : batch * row/s * col/s 
                os_tensor    : batch * row/s * col/s * 2
                kp_tensor    : batch * row/s * col/s * max_point + 1 * 3 (w, h, prob)
                '''
                out_train_tensor = train_net.compute_loss(image_tensor, hm_tensor, os_tensor, kp_tensor)

                hm_lt = out_train_tensor['hm_loss']
                os_lt = out_train_tensor['os_loss']
                kp_lt = out_train_tensor['kp_loss']
                reg_lt = out_train_tensor['reg_loss']

                predict_tensor = out_train_tensor['predict']

                total_lt = hm_lt + os_lt + kp_lt + reg_lt

                grads = optimizer.compute_gradients(total_lt)
                grads_list.append(grads)

                train_tensor_list.append([image_tensor, hm_tensor, os_tensor, kp_tensor, predict_tensor])

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

                ttloss_sum += total_lt
                hmloss_sum += hm_lt
                osloss_sum += os_lt
                kploss_sum += kp_lt

        train_loss_list = [ttloss_sum, hmloss_sum, osloss_sum, kploss_sum, reg_lt]
        train_hm_scalar = tf.summary.scalar(name='hm_loss', tensor=hmloss_sum)
        train_os_scalar = tf.summary.scalar(name='os_loss', tensor=osloss_sum)
        train_kp_scalar = tf.summary.scalar(name='kp_loss', tensor=kploss_sum)
        train_tt_scalar = tf.summary.scalar(name='total_loss', tensor=ttloss_sum)
        train_summary_list.append(train_hm_scalar)
        train_summary_list.append(train_os_scalar)
        train_summary_list.append(train_kp_scalar)
        train_summary_list.append(train_tt_scalar)


        print("######################## GRADIENTS UPDATE ########################")
        grads = average_gradients(grads_list)
        apply_gradient_op = optimizer.apply_gradients(grads, global_step=gs_tensor)
        variable_averages = tf.train.ExponentialMovingAverage(0.9999, gs_tensor)
        variables_to_average = tf.trainable_variables()
        variables_averages_op = variable_averages.apply(variables_to_average)
        batchnorm_updates_op = tf.group(*update_ops)
        train_op = tf.group(apply_gradient_op, variables_averages_op, batchnorm_updates_op)

        print("######################## DRAW GRAPH OF VAL ########################")
        val_net = model.Network(phase='val', reuse=True)
        with tf.device('/gpu:3'):
            val_datasets = Datasets()
            val_iterator, _ = val_datasets.Generate_dataset('val_list.json')
            image_tensor, hm_tensor, os_tensor, kp_tensor = val_iterator.get_next()
            out_train_tensor = val_net.compute_loss(image_tensor, hm_tensor, os_tensor, kp_tensor)
            predict_tensor = out_train_tensor['predict']
            val_loss = out_train_tensor['hm_loss'] + out_train_tensor['os_loss'] + out_train_tensor['kp_loss']
            val_list = [image_tensor, hm_tensor, os_tensor, kp_tensor, predict_tensor, val_loss]
        val_loss_summary = tf.summary.scalar(name='val_loss', tensor=val_loss)
        train_summary_list.append(val_loss_summary)
        summary_tensor = tf.summary.merge(train_summary_list)

        print("######################## DEFINE SESSION ########################")
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        # sess_config.gpu_options.per_process_gpu_memory_fraction = 0.9
        sess_config.gpu_options.allow_growth = True
        sess_config.gpu_options.allocator_type = 'BFC'

        sess = tf.Session(config=sess_config)

        with sess.as_default():
            print("######################## SET SAVER ########################")
            train_start_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
            model_save_dir = './ckpt/{:s}/'.format(str(train_start_time))

            os.makedirs(model_save_dir, exist_ok=True)
            model_name = 'weights.ckpt'
            model_save_path = ops.join(model_save_dir, model_name)
            sess.run(tf.global_variables_initializer())

            if weights_path is not None:
                variables = tf.global_variables()
                variables_ckpt = tf.contrib.framework.list_variables(weights_path)
                variables_ckpt = [v[0] + ':0' for v in variables_ckpt]

                variables_to_restore = [v for v in variables if v.name in variables_ckpt]
                variables_to_not = [v for v in variables if not v.name in variables_ckpt]
                print('not loaded node :')
                print(variables_to_not)

                saver = tf.train.Saver(variables_to_restore, max_to_keep=4)
                saver.restore(sess=sess, save_path=weights_path)
                saver = tf.train.Saver(tf.global_variables(), max_to_keep=4)
                sess.run(tf.initialize_variables([gs_tensor]))

            tboard_save_path = model_save_dir + 'tboard/'
            os.makedirs(tboard_save_path, exist_ok=True)

            summary_writer = tf.summary.FileWriter(tboard_save_path)

            print("######################## START_RUN ########################")
            for epoch in range(train_epochs):
                for dataset in dataset_list:
                    dataset.shuffling()
                for iterator in iterator_list:
                    sess.run(iterator.initializer)
                sess.run(val_iterator.initializer)
                for iter in range(total_iteration):
                    _, train_lists, loss_lists, val_lists, lr, summary_train, global_step = \
                        sess.run([train_op, train_tensor_list, train_loss_list, val_list, lr_tensor, summary_tensor, gs_tensor])
                    val_loss_p = val_lists[5]

                    summary_writer.add_summary(summary=summary_train, global_step=global_step)

                    total_loss, hm_loss, os_loss, kp_loss, reg_loss = loss_lists
                    print('iter:%d/%d    total:%.6f    hm:%.6f   os:%.6f   kp:%.6f   val:%.6f   reg:%.6f   lr:%.6f'
                          %(epoch, iter, total_loss, hm_loss, os_loss, kp_loss, val_loss_p, reg_loss, lr))

                    if iter % 500 == 0:
                        saver.save(sess=sess, save_path=model_save_path, global_step=gs_tensor)
                        record_results(train_lists, save_dir='./tmp/')
                        record_results([val_lists], save_dir='./tmp_val/')

        return

if __name__ == '__main__':
    args = init_args()
    train()
