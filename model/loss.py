'''
DL Model's Loss In Tensorflow Template
By Kwanwoo Park, 2022.
'''
import tensorflow as tf
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from config_kp import cfg

class Losses():
    def __init__(self):
        super(Losses, self).__init__()

    def CrossEntropyLoss(self, label, pred):
        loss = -tf.reduce_sum(label * tf.math.log(pred), axis=1)
        loss = tf.reduce_mean(loss)
        return loss

    def regularize_loss(self):
        l2_reg_loss = tf.constant(0.0, tf.float32)
        for vv in tf.trainable_variables():
            if 'bn' in vv.name or 'head' in vv.name:
                continue
            else:
                if not 'conv' in vv.name:
                    print(vv.name)
                l2_reg_loss = tf.add(l2_reg_loss, tf.nn.l2_loss(vv))
        return l2_reg_loss
