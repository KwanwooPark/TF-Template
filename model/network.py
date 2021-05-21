import tensorflow as tf
from model.OPS import *
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import arg_scope

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from config import cfg

class Arch():
    def __init__(self, phase):
        super(Arch, self).__init__()
        self._phase = phase
        self._is_training = self._is_net_for_training()
        print('now_phase =================== ', self._is_training)
        self.outputs = {}

    def _is_net_for_training(self):
        if self._phase == 'train':
            return True
        else:
            return False

    def body(self, img):
        with tf.variable_scope('body'):
            # img = tf.div(img, 255, name='norm')

            net = self.Stem(img, 'Stem1')

            _net8 = self.down_block(net, 64, '_Down1')
            _net8 = self.main_block(_net8, 64, 1, '_Main1')

            _net16 = self.down_block(_net8, 128, '_Down2')
            _net16 = self.main_block(_net16, 128, 2, '_Main2')

            _net32 = self.down_block(_net16, 256, '_Down3')
            _net32 = self.main_block(_net32, 256, 3, '_Main3')

            _net16 = self.up_and_concat(_net32, _net16, 'comb1')
            _net8 = self.up_and_concat(_net16, _net8, 'comb2')

            net = self.Stem(img, 'Stem2')

            net8 = self.down_block(net, 64, 'Down1') + _net8
            net8 = self.main_block(net8, 64, 2, 'Main1')

            net16 = self.down_block(net8, 128, 'Down2') + _net16
            net16 = self.main_block(net16, 128, 3, 'Main2')

            net32 = self.down_block(net16, 256, 'Down3') + _net32
            net32 = self.main_block(net32, 256, 4, 'Main3')

            net32 = self.fusion_block(net32, _net32, 256, 'fusion1')
            net16 = self.up_and_concat(net32, net16, 'comb3')
            net16 = self.fusion_block(net16, _net16, 128, 'fusion2')
            net8 = self.up_and_concat(net16, net8, 'comb4')
            net8 = self.fusion_block(net8, _net8, 64, 'fusion3')
        return net8

    def head(self, feat):
        with tf.variable_scope('head'):
            hm = self.decode_block(feat, 1, 'out_hm')
            os = self.decode_block(feat, 2, 'out_os')
            kp = self.decode_block(feat, 32, 'out_kp')

            self.outputs['hm'] = tf.nn.sigmoid(hm)
            self.outputs['os'] = os
            self.outputs['kp'] = kp

    def Stem(self, net, name):
        with tf.variable_scope(name):
            net = Conv_(net, filter=8, kernel=3, stride=2, name='conv1')
            net = BN_(net, self._is_training, name='bn1')
            net = Relu_(net)
            net = Conv_(net, filter=16, kernel=3, stride=1, name='conv2')
            net = BN_(net, self._is_training, name='bn2')
            net = Relu_(net)
            net = Conv_(net, filter=32, kernel=3, stride=2, name='conv3')
            net = BN_(net, self._is_training, name='bn3')
            net = Relu_(net)
        return net

    def fusion_block(self, feat1, feat2, channel, name):
        with tf.variable_scope(name):
            feat_w = Conv_(feat1, filter=channel, kernel=1, stride=1, name='conv1')
            feat_w = tf.nn.sigmoid(feat_w)
            return feat1 * feat_w + feat2 * (1 - feat_w)


    def down_block(self, net, out_channel, name):
        with tf.variable_scope(name):
            _skip = Conv_(net, filter=out_channel, kernel=1, stride=2, name='conv1')

            net = Depth_conv_(net, filter=int(out_channel/2), kernel=3, name='d_conv1')
            net = BN_(net, self._is_training, name='bn1')
            net = Relu_(net)
            net = Depth_conv_(net, filter=out_channel, kernel=3, name='d_conv2')
            net = BN_(net, self._is_training, name='bn2')
            net = Maxpool_(net, kernel=3, stride=2, name='pool')
            net = net + _skip
            net = Relu_(net)
        return net

    def main_block(self, net, channel_out, iter, name):
        for i in range(iter):
            with tf.variable_scope(name + '/block_%s' % (str(i))):
                _res = net
                net = Depth_conv_(net, int(channel_out/2), 3, name='d_conv1')
                net = BN_(net, self._is_training, name='bn1')
                net = Relu_(net)
                net = Depth_conv_(net, channel_out, 3, name='d_conv2')
                net = BN_(net, self._is_training, name='bn2')
                net = net + _res
                net = Relu_(net)
        return net

    def up_and_concat(self, net_small, net_large, name):
        with tf.variable_scope(name):
            _shape = net_large.get_shape().as_list()
            net_small = Conv_(net_small, filter=_shape[3], kernel=1, stride=1, name='conv1')
            net_small = Resize_(net_small, (_shape[1], _shape[2]))
            net = Concat_((net_large, net_small))
            net = Conv_(net, filter=_shape[3], kernel=1, stride=1, name='conv2')
            net = Depth_conv_(net, _shape[3], 3, name='d_conv1')
        return net

    def decode_block(self, net, class_num, name):
        with tf.variable_scope(name):
            net = Conv_(net, 64, 1, name='conv1')
            net = Relu_(net)
            net = Conv_(net, 32, 1, name='conv2')
            net = Conv_(net, class_num, 1, name='conv3')
        return net


    def build_model(self, input, reuse=False):
        with tf.variable_scope(name_or_scope='net', reuse=reuse):
            feat = self.body(input)
            self.head(feat)
        return self.outputs


if __name__ == '__main__':
    image_train_tensor = tf.placeholder(tf.float32, shape=[cfg.batch_size, 480, 800, 3])
    net = Arch(phase='train')
    outputs = net.build_model(image_train_tensor)
    print(outputs)
