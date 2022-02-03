'''
DL Model's Architecture In Tensorflow Template
By Kwanwoo Park, 2022.
'''

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.OPS import *
from config_kp import cfg

class Arch():
    def __init__(self, phase):
        super(Arch, self).__init__()
        if phase == 'train':
            self._is_training = True
        else:
            self._is_training = False
        print('now_phase =================== ', self._is_training)
        self.feature = {}
        self.outputs = {}

    def Stem(self, net, name):
        with tf.variable_scope(name):
            net = Conv_(net, filter=32, kernel=7, stride=2, name='Stem_conv')
            net = BN_(net, self._is_training, name='Stem_bn')
            net = Relu_(net)
        return net

    def ResBlock(self, net, input_channel, iter, name):
        with tf.variable_scope(name):
            _net = Conv_(net, filter=input_channel, kernel=3, stride=2, name='conv1_1')
            # identity type
            net = Conv_(net, filter=input_channel, kernel=1, stride=2, name='conv_identity')

            _net = BN_(_net, self._is_training, name='bn1_1')
            _net = Relu_(_net)

            _net = Conv_(_net, filter=input_channel, kernel=3, stride=1, name='conv1_2')
            _net = BN_(_net, self._is_training, name='bn1_2')
            net = Relu_(_net + net)

            for i in range(2, iter+1):
                _net = Conv_(net, filter=input_channel, kernel=3, stride=1, name='conv%d_1'%i)
                _net = BN_(_net, self._is_training, name='bn%d_1'%i)
                _net = Relu_(_net)

                _net = Conv_(_net, filter=input_channel, kernel=3, stride=1, name='conv%d_2'%i)
                _net = BN_(_net, self._is_training, name='bn%d_2'%i)
                net = Relu_(_net + net)

            return net

    def body(self, img):
        with tf.variable_scope('body'):
            net = self.Stem(img, 'Stem') # x2

            net = self.ResBlock(net, 64, 2, 'ResBlock1')
            net = self.ResBlock(net, 128, 2, 'ResBlock2')
            net = self.ResBlock(net, 256, 2, 'ResBlock3')
            net = self.ResBlock(net, 512, 2, 'ResBlock4')
            self.feature['feat'] = net

    def head(self):
        with tf.variable_scope('head'):
            feat = GAP_(self.feature['feat'], keepdims=False)
            feat = FC_(feat, 1000, 'FC_1')
            feat = FC_(feat, cfg.class_num, 'FC_2')
            self.outputs['pred'] = tf.nn.softmax(feat)

    def run_model(self, input, reuse=False):
        with tf.variable_scope(name_or_scope='net', reuse=reuse):
            self.body(input)
            self.head()
        return self.outputs

### This is for test. ###
if __name__ == '__main__':
    image_train_tensor = tf.placeholder(tf.float32, shape=[cfg.batch_size, 32, 32, 3])
    net = Arch(phase='train')
    outputs = net.run_model(image_train_tensor)
    print(outputs)

    sess_config = tf.ConfigProto(device_count={'GPU': 0})
    sess = tf.Session(config=sess_config)
    with sess.as_default():
        sess.run(tf.global_variables_initializer())

    for n in tf.get_default_graph().as_graph_def().node:
       if not 'save/' in n.name:
           print(n.name, n.op)

    output_node_names = ['net/head/FC_2/dense/BiasAdd']

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
       # needs 4517004416 FLOPS after freezing
