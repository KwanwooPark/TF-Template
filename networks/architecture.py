import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import tensorflow as tf
from networks.OPS import *
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import arg_scope
from config_kp import cfg

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

    def Stem(self, net, name):
        with tf.variable_scope(name):
            net = Conv_(net, filter=16, kernel=3, stride=1, name='conv1')
            net = BN_(net, self._is_training, name='bn1')
            net = Relu_(net)
            net = Conv_(net, filter=32, kernel=3, stride=2, name='conv2')
            net = BN_(net, self._is_training, name='bn2')
            net = Relu_(net)
        return net

    def Shared_block(self, net, kernel, channel, name):
        with tf.variable_scope(name):
            _kernel = tf.reduce_sum(kernel, axis=(0, 1), keepdims=True)

            _net1 = Conv_with_kernel(net, kernel=kernel, name='conv1')
            _net2 = Conv_with_kernel(net, kernel=_kernel, name='conv2')

            _weight = Conv_(net, filter=channel, kernel=3, stride=1, name='conv3')
            _weight = tf.sigmoid(_weight)

            _net = _net1 * _weight + _net2 * (1 - _weight)
        return net + _net

    def res2net_block(self, net, input_channel, out_channel, slice_num, name):
        with tf.variable_scope(name):
            mid_channel = out_channel//slice_num

            _net = Conv_(net, filter=out_channel, kernel=1, stride=1, name='conv1')
            _net = BN_(_net, self._is_training, name='bn1')
            _net = Relu_(_net)

            _net_list = Slice_(_net, out_channel, slice_num)

            _net = Conv_(_net_list[1], filter=mid_channel, kernel=3, stride=1, name='conv2')
            _net = BN_(_net, self._is_training, name='bn2')
            _net = Relu_(_net)

            net_concated = Concat_([_net_list[0], _net])

            for i in range(2, slice_num):
                _net = _net + _net_list[i]
                _net = Conv_(_net, filter=mid_channel, kernel=3, stride=1, name='conv'+str(i+1))
                _net = BN_(_net, self._is_training, name='bn'+str(i+1))
                _net = Relu_(_net)

                net_concated = Concat_([net_concated, _net])

            _net = Conv_(net_concated, filter=input_channel, kernel=1, stride=1, name='conv-1')
            _net = BN_(_net, self._is_training, name='bn-1')
            _net = Relu_(_net)

            return _net + net

    def down_block(self, net, out_channel, mid_channel, name):
        with tf.variable_scope(name):
            _skip = Conv_(net, filter=out_channel, kernel=1, stride=2, name='conv0')

            net = Conv_(net, filter=mid_channel, kernel=3, name='conv2')
            net = BN_(net, self._is_training, name='bn2')
            net = Relu_(net)

            net = Depth_conv_(net, filter=out_channel, kernel=3, name='dconv1')
            net = BN_(net, self._is_training, name='bn3')
            net = Maxpool_(net, kernel=3, stride=2, name='pool')
            net = Relu_(net + _skip)
        return net

    def main_block(self, net, channel_out, slice_num, iter, name):
        for i in range(iter):
            with tf.variable_scope(name + '/block_%s' % (str(i))):
                _res = net
                net = Depth_conv_(net, channel_out, 3, name='d_conv1')
                net = BN_(net, self._is_training, name='bn1')
                net = Relu_(net)

                net = self.res2net_block(net, channel_out, channel_out, slice_num, 'res2_b')

                net = Depth_conv_(net, channel_out, 3, name='d_conv2')
                net = BN_(net, self._is_training, name='bn2')
                net = Relu_(net)

                net = _res + net

        return net


    def body(self, img):
        with tf.variable_scope('body'):
            net = self.Stem(img, 'Stem1') # x2

            net = self.down_block(net, 64, 32, '_Down1')
            net = self.main_block(net, 64, 4, 3, '_Main1')
            net = self.Shared_block(net, self.shared_kernel1, 64, '_Shared1')

            net = self.down_block(net, 128, 64, '_Down2')
            net = self.main_block(net, 128, 4, 4, '_Main2')
            net = self.Shared_block(net, self.shared_kernel2, 128, '_Shared2')

            net = self.down_block(net, 384, 192, '_Down3')
            net = self.main_block(net, 384, 4, 8, '_Main3')
            net = self.Shared_block(net, self.shared_kernel3, 384, '_Shared3')

        return net

    def decode_block(self, net, channels, name):
        with tf.variable_scope(name):
            net = Conv_(net, channels[0], 1, name='conv1')
            net = BN_(net, self._is_training, name='bn1')
            net = Relu_(net)
            net = Conv_(net, channels[1], 1, name='conv2')
            net = Conv_(net, channels[2], 1, name='conv3')
        return net

    def head(self, feat):
        with tf.variable_scope('head'):
            hm = self.decode_block(feat, [128, 32, 1], 'out_hm')
            kp = self.decode_block(feat, [256, 128, cfg.point_num * 2], 'out_kp')

            self.outputs['hm'] = tf.nn.sigmoid(hm)
            self.outputs['os'] = os
            self.outputs['kp'] = kp

    def build_model(self, input, reuse=False):
        with tf.variable_scope(name_or_scope='net', reuse=reuse):
            self.shared_kernel1 = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(),
                                                  shape=[3, 3, 64, 64], name='shared_conv_kernel1')
            self.shared_kernel2 = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(),
                                                  shape=[3, 3, 128, 128], name='shared_conv_kernel2')
            self.shared_kernel3 = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(),
                                                  shape=[3, 3, 384, 384], name='shared_conv_kernel3')

            feat = self.body(input)
            self.head(feat)
        return self.outputs


if __name__ == '__main__':
    image_train_tensor = tf.placeholder(tf.float32, shape=[cfg.batch_size, 480, 800, 3])
    net = Arch(phase='train')
    outputs = net.build_model(image_train_tensor)
    print(outputs)

    sess_config = tf.ConfigProto(device_count={'GPU': 0})
    sess = tf.Session(config=sess_config)
    with sess.as_default():
        sess.run(tf.global_variables_initializer())

    for n in tf.get_default_graph().as_graph_def().node:
       if not 'save/' in n.name:
           print(n.name, n.op)

    output_node_names = ['net/head/Sigmoid', 'net/head/strided_slice', 'net/head/strided_slice_1', 'net/head/strided_slice_2']

    frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), output_node_names)

    with open(os.path.join(os.getcwd(), './graph.pb'), 'wb') as f:
        f.write(frozen_graph_def.SerializeToString())

    def load_pb(pb_model):
        with tf.gfile.GFile(pb_model, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')
            return graph

    graph = load_pb('./graph.pb')
    with graph.as_default():
       flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
       print('needs {} FLOPS after freezing'.format(flops.total_float_ops))

# needs 425874505728 FLOPS after freezing
# needs 252719044608 FLOPS after freezing
# needs 1236573616128 FLOPS after freezing
# needs 797226160128 FLOPS after freezing
