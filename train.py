'''
Train Code In Tensorflow Template
By Kwanwoo Park, 2022.
'''
import os.path as ops
import time

from model import base
from dataset import Datasets
from utils.train_utils import *
from config_kp import cfg
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'


class Trainer():
    def __init__(self):
        super(Trainer, self).__init__()
        self.gpus = [0, 1, 2, 3]
        self.GS_tensor = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0.), trainable=False)

    def TrainParam(self):
        self.LR_tensor = tf.train.cosine_decay(cfg.learning_rate, self.GS_tensor, cfg.max_iter, alpha=cfg.lowest_lr_rate)
        self.Optimizer = tf.train.AdamOptimizer(self.LR_tensor, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False)

        SessConfig = tf.ConfigProto(allow_soft_placement=True)
        SessConfig.gpu_options.allow_growth = True
        SessConfig.gpu_options.allocator_type = 'BFC'
        self.Sess = tf.Session(config=SessConfig)

    def Summary(self):
        Summaries_list = []

        LR_summary = tf.summary.scalar(name='LR', tensor=self.LR_tensor)
        Summaries_list.append(LR_summary)

        ClsLoss_summary = tf.summary.scalar(name='cls_loss', tensor=self.Loss_list[0])
        RegLoss_summary = tf.summary.scalar(name='reg_loss', tensor=self.Loss_list[1])
        TotalLoss_summary = tf.summary.scalar(name='total_loss', tensor=self.Loss_list[2])
        Summaries_list.append(ClsLoss_summary)
        Summaries_list.append(RegLoss_summary)
        Summaries_list.append(TotalLoss_summary)

        ValLoss_summary = tf.summary.scalar(name='val_loss', tensor=self.ValLoss)
        Summaries_list.append(ValLoss_summary)

        Tboard_Path = self.ModelSaveDir + 'tboard/'
        os.makedirs(Tboard_Path, exist_ok=True)
        self.SummaryWriter = tf.summary.FileWriter(Tboard_Path)
        self.Summary_tensor = tf.summary.merge(Summaries_list)

    def LoadTrainDataset(self):
        self.TrainDataset = Datasets(is_training=True)
        self.TrainIterator, self.TotalIteration = self.TrainDataset.Generate_dataset(cfg.train_list)
        self.Image_tensor, self.Label_tensor = self.TrainIterator.get_next()
        self.Image_tensor = tf.split(self.Image_tensor, len(cfg.gpus), axis=0)
        self.Label_tensor = tf.split(self.Label_tensor, len(cfg.gpus), axis=0)


    def BuildTrain(self):
        TrainModel = base.Model(phase='train', reuse=False)
        self.Pred_tensor = []
        self.Loss_list = [tf.constant(0.0, tf.float32), tf.constant(0.0, tf.float32), tf.constant(0.0, tf.float32)]
        Grad_list = []
        for i in self.gpus:
            with tf.device('/gpu:%d' % i):
                Out_tensor = TrainModel.compute_loss(self.Image_tensor[i], self.Label_tensor[i])

                ClsLoss_tensor = Out_tensor['cls_loss']
                RegLoss_tensor = Out_tensor['reg_loss']
                TotalLoss_tensor = ClsLoss_tensor + RegLoss_tensor
                self.Pred_tensor.append(Out_tensor['pred'])

                Grad_tensor = self.Optimizer.compute_gradients(TotalLoss_tensor)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

                Grad_list.append(Grad_tensor)
                self.Loss_list[0] += ClsLoss_tensor / len(self.gpus)
                self.Loss_list[1] += RegLoss_tensor / len(self.gpus)
                self.Loss_list[2] += TotalLoss_tensor / len(self.gpus)


        Grad = average_gradients(Grad_list)
        apply_gradient_op = self.Optimizer.apply_gradients(Grad, global_step=self.GS_tensor)
        variable_averages = tf.train.ExponentialMovingAverage(0.9999, self.GS_tensor)
        variables_to_average = tf.trainable_variables()
        variables_averages_op = variable_averages.apply(variables_to_average)
        batchnorm_updates_op = tf.group(*update_ops)
        self.TrainOP = tf.group(apply_gradient_op, variables_averages_op, batchnorm_updates_op)

    def BuildVal(self):
        ValModel = base.Model(phase='val', reuse=True)
        self.ValDataset = Datasets(is_training=False)
        self.ValIterator, self.ValIteration = self.ValDataset.Generate_dataset(cfg.val_list)
        self.ValImage_tensor, self.ValLabel_tensor = self.ValIterator.get_next()
        self.ValImage_tensor = tf.split(self.ValImage_tensor, len(cfg.gpus), axis=0)
        self.ValLabel_tensor = tf.split(self.ValLabel_tensor, len(cfg.gpus), axis=0)
        self.ValPred_tensor = []
        self.ValLoss = tf.constant(0.0, tf.float32)
        for i in self.gpus:
            with tf.device('/gpu:%d' % i):
                Out_tensor = ValModel.compute_loss(self.ValImage_tensor[i], self.ValLabel_tensor[i])
                self.ValPred_tensor.append(Out_tensor['pred'])
                self.ValLoss += Out_tensor['cls_loss'] / len(self.gpus)

    def LoadSaver(self):
        with self.Sess.as_default():
            self.ModelSaveDir = './ckpt/{:s}/'.format(str(time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))))
            os.makedirs(self.ModelSaveDir, exist_ok=True)
            SaveName = 'weights.ckpt'
            self.SavePath = ops.join(self.ModelSaveDir, SaveName)

            self.Sess.run(tf.global_variables_initializer())
            if cfg.load_weight_path is not None:
                variables = tf.global_variables()
                variables_ckpt = tf.contrib.framework.list_variables(cfg.load_weight_path)
                variables_ckpt = [v[0] + ':0' for v in variables_ckpt]

                variables_to_restore = [v for v in variables if v.name in variables_ckpt]
                variables_to_not = [v for v in variables if not v.name in variables_ckpt]
                print('not loaded node :')
                print(variables_to_not)

                saver = tf.train.Saver(variables_to_restore, max_to_keep=4)
                saver.restore(sess=self.Sess, save_path=cfg.load_weight_path)

            self.Saver = tf.train.Saver(tf.global_variables(), max_to_keep=4)
            if cfg.GS_reset:
                self.Sess.run(tf.initialize_variables([self.GS_tensor]))

    def RunTrain(self):
        with self.Sess.as_default():
            self.Sess.run(self.TrainIterator.initializer)
            self.Sess.run(self.ValIterator.initializer)
            for epoch in range(cfg.max_epoch):
                self.TrainDataset.shuffling()
                for iter in range(self.TotalIteration):
                    _, images, labels, preds, losses, lr, summary, gs = \
                        self.Sess.run([self.TrainOP, self.Image_tensor, self.Label_tensor, self.Pred_tensor,
                                       self.Loss_list, self.LR_tensor, self.Summary_tensor, self.GS_tensor])
                    self.SummaryWriter.add_summary(summary=summary, global_step=gs)

                    cls_loss, reg_loss, total_loss = losses
                    print('%d/%d:  total:%.6f    cls:%.6f   reg:%.6f   lr:%.6f'
                          % (epoch, iter, total_loss, cls_loss, reg_loss, lr))

                    if iter % cfg.val_iter == 0:
                        for Valiter in range(self.ValIteration):
                            valimages, vallabels, valpreds, vallosses = \
                                self.Sess.run([self.ValImage_tensor, self.ValLabel_tensor, self.ValPred_tensor, self.ValLoss])
                            print('VAL/%d:  valtotal:%.6f'
                                  % (Valiter, vallosses))

                    if (iter+1) % cfg.save_iter == 0:
                        self.Saver.save(sess=self.Sess, save_path=self.SavePath, global_step=self.GS_tensor)

    def Main(self):
        with tf.device('/device:CPU:0'):
            self.TrainParam()
            self.LoadTrainDataset()
            self.BuildTrain()
            self.BuildVal()
            self.LoadSaver()
            self.Summary()
            self.RunTrain()
        return


if __name__ == '__main__':
    trainer = Trainer()
    trainer.Main()