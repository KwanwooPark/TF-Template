import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import tensorflow as tf
from model import loss
from model import architecture
from config_kp import cfg

class Model():
    def __init__(self, phase, reuse=False):
        super(Model, self).__init__()
        self._reuse = reuse
        self._forward = architecture.Arch(phase=phase)
        self._backward = loss.Losses()

    def inference(self, input_tensor, name='model'):
        with tf.variable_scope(name_or_scope=name, reuse=self._reuse):
            network_results = self._forward.run_model(
                input=input_tensor,
                reuse=self._reuse
            )
            ret = {'pred': network_results['pred']}
            self._reuse = True
        return ret

    def compute_loss(self, input_tensor, label_tensor, name='model'):
        with tf.variable_scope(name_or_scope=name, reuse=self._reuse):
            network_results = self._forward.run_model(
                input=input_tensor,
                reuse=self._reuse)
            pred_tensor = network_results['pred']

            cls_loss = self._backward.CrossEntropyLoss(label=label_tensor, pred=pred_tensor)
            reg_loss = self._backward.regularize_loss()

            ret = {
                'cls_loss': cls_loss,
                'reg_loss': reg_loss * 0.000001,
                'pred': pred_tensor
            }
            self._reuse = True
        return ret
