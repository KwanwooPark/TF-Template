import tensorflow as tf
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from config import cfg

class BackEnd():
    def __init__(self):
        super(BackEnd, self).__init__()

    def hm_loss(self, label, logit):
        '''
        :param label: batch * h * w
        :param logit: batch * h * w
        :return:
        '''
        pos_inds = tf.cast(tf.equal(label, 1), tf.float32)
        neg_inds = tf.cast(tf.less(label, 1), tf.float32)

        neg_weights = tf.pow(1 - label, 4)

        pos_loss = tf.log(logit + 1e-8) * tf.pow(1 - logit, 2) * pos_inds
        neg_loss = tf.log(1 - logit + 1e-8) * tf.pow(logit, 2) * neg_weights * neg_inds

        num_pos = tf.reduce_sum(pos_inds)
        pos_loss = tf.reduce_sum(pos_loss) * 10
        neg_loss = tf.reduce_sum(neg_loss)

        loss = -(pos_loss + neg_loss) / (tf.maximum(num_pos, tf.constant(1.0)))

        return loss

    def os_loss(self, os_label, os_logit, hm_label):
        '''
        :param hm_label: batch * h * w
        :param os_logit: batch * h * w * 2
        :param os_label: batch * h * w * 2
        :return:
        '''
        pos_inds = tf.cast(tf.equal(hm_label, 1), tf.float32)

        loss = tf.abs(os_label - os_logit)
        loss = tf.reduce_sum(loss, axis=3)
        loss = loss * pos_inds
        loss = tf.reduce_sum(loss) / (tf.reduce_sum(pos_inds) + 1e-6)
        return loss


    def kp_loss(self, hm_label, kp_label, kp_logit):
        '''
        :param hm_label: batch * h * w
        :param kp_label: batch * h * w * 16 * 2
        :param pl_logit: batch * h * w * 16 * 2
        :return:
        '''
        pos_inds = tf.cast(tf.equal(hm_label, 1), tf.float32)
        loss = tf.abs(kp_label - kp_logit)
        loss = tf.reduce_sum(loss, axis=[3, 4])
        loss = loss * pos_inds
        loss = tf.reduce_sum(loss) / (tf.reduce_sum(pos_inds) + 1e-6)
        # shape = wh_logit.get_shape().as_list()
        # weight = tf.constant([2, 1.3, 1.1, 0.9, 0.9, 0.6, 0.6, 0.6], shape=[1, 1, 1, 8, 1])
        # weight = tf.tile(weight, [8, shape[1], shape[2], 1, 2])  # batch_size=8
        # loss = loss * weight
        return loss

    def regularize_loss(self):
        l2_reg_loss = tf.constant(0.0, tf.float32)
        for vv in tf.trainable_variables():
            if 'bn' in vv.name or 'decoder' in vv.name:
                continue
            else:
                if not 'conv' in vv.name:
                    print(vv.name)
                l2_reg_loss = tf.add(l2_reg_loss, tf.nn.l2_loss(vv))
        return l2_reg_loss
