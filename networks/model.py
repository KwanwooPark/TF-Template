import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import tensorflow as tf
import numpy as np
from networks import loss
from networks import architecture
import tensorflow.contrib.slim as slim
from config_kp import cfg

class Network():
    def __init__(self, phase, reuse=False):
        super(Network, self).__init__()
        self._reuse = reuse

        self._frontend = architecture.Arch(phase=phase)
        self._backend = loss.BackEnd()

    def inference(self, input_tensor, name='model'):
        with tf.variable_scope(name_or_scope=name, reuse=self._reuse):
            network_results = self._frontend.build_model(
                input=input_tensor,
                reuse=self._reuse
            )

            hm_pred = network_results['hm']
            kp_pred = network_results['kp']

            # with tf.variable_scope("post_process"):
            #     _hm_pred = tf.layers.max_pooling2d(hm_pred, (3, 3), (1, 1), padding="SAME", data_format='channels_last')
            #     _keep = tf.cast(tf.equal(_hm_pred, hm_pred), tf.float32)
            #     hm_pred = hm_pred * _keep

            hm_pred = tf.identity(hm_pred, 'out_hm')
            kp_pred = tf.identity(kp_pred, 'out_kp')

            hm_pred = hm_pred[:, :, :, 0]
            ret = {
                'predict': [hm_pred,  kp_pred]
            }

            if not self._reuse:
                self._reuse = True

        return ret

    def compute_loss(self, image, hm_label, kp_label, name='model'):
        with tf.variable_scope(name_or_scope=name, reuse=self._reuse):
            network_results = self._frontend.build_model(
                input=image,
                reuse=self._reuse)

            hm_pred = network_results['hm']
            kp_pred = network_results['kp']
            hm_pred = hm_pred[:, :, :, 0]
            hm_loss = self._backend.hm_loss(label=hm_label, logit=hm_pred)
            kp_loss_p = self._backend.kp_loss(kp_label=kp_label, kp_logit=kp_pred, hm_label=hm_label)

            hm_pred = tf.stop_gradient(hm_pred)
            pos_inds = tf.cast(tf.greater(hm_pred, cfg.RPN_threshold), tf.float32)
            pos_inds *= tf.cast(tf.less(hm_label, 1), tf.float32)
            pos_inds *= tf.cast(tf.greater(hm_label, 0), tf.float32)
            kp_loss_n = self._backend.kp_loss(kp_label=kp_label, kp_logit=kp_pred, hm_label=hm_label, pos_inds=pos_inds)

            reg_loss = self._backend.regularize_loss()

            if not self._reuse:
                self._reuse = True

            ret = {
                'hm_loss': hm_loss * cfg.loss_lambdas[0],
                'kp_loss_p': kp_loss_p * cfg.loss_lambdas[1],
                'kp_loss_n': kp_loss_n * cfg.loss_lambdas[2],
                'reg_loss': reg_loss * cfg.loss_lambdas[3],
                'predict': [hm_pred, kp_pred]
            }
        return ret
