import tensorflow as tf
import numpy as np
from model import loss
from model import network
import tensorflow.contrib.slim as slim
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from config import cfg

class Network():
    def __init__(self, phase, reuse=False):
        super(Network, self).__init__()
        self._reuse = reuse

        self._frontend = network.Arch(phase=phase)
        self._backend = loss.BackEnd()

    def inference(self, input_tensor, name='model'):
        with tf.variable_scope(name_or_scope=name, reuse=self._reuse):
            network_results = self._frontend.build_model(
                input=input_tensor,
                reuse=self._reuse
            )

            hm_pred = network_results['hm']
            os_pred = network_results['os']
            kp_pred = network_results['kp']

            hm_pred = tf.identity(hm_pred, 'out_hm')
            os_pred = tf.identity(os_pred, 'out_os')
            kp_pred = tf.identity(kp_pred, 'out_kp')

            hm_pred = hm_pred[:, :, :, 0]
            kp_pred = tf.reshape(kp_pred, (-1, int(cfg.height / cfg.rate), int(cfg.width / cfg.rate), 16, 2))

            ret = {
                'predict': [hm_pred, os_pred, kp_pred]
            }

            if not self._reuse:
                self._reuse = True

        return ret

    def compute_loss(self, image, hm_label, os_label, kp_label, name='model'):
        with tf.variable_scope(name_or_scope=name, reuse=self._reuse):
            network_results = self._frontend.build_model(
                input=image,
                reuse=self._reuse)

            hm_pred = network_results['hm']
            os_pred = network_results['os']
            kp_pred = network_results['kp']

            hm_pred = hm_pred[:, :, :, 0]
            kp_pred = tf.reshape(kp_pred, (-1, int(cfg.height / cfg.rate), int(cfg.width / cfg.rate), 16, 2))

            hm_loss = self._backend.hm_loss(label=hm_label, logit=hm_pred)
            os_loss = self._backend.os_loss(os_label=os_label, os_logit=os_pred, hm_label=hm_label)
            kp_loss = self._backend.kp_loss(kp_label=kp_label, kp_logit=kp_pred, hm_label=hm_label)
            reg_loss = self._backend.regularize_loss()

            if not self._reuse:
                self._reuse = True

            ret = {
                'hm_loss': hm_loss * 1.0,
                'os_loss': os_loss * 0.1,
                'kp_loss': kp_loss * 0.1,
                'reg_loss': reg_loss * 0.000001,
                'predict': [hm_pred, os_pred, kp_pred]
            }
        return ret
