import numpy as np

class Config(object):
    pass

cfg = Config()
# Base
cfg.height = 352
cfg.width = 800
cfg.rate = 16.0
cfg.point_num = 8
cfg.point_pos = [0, 1, 2, 3, 6, 9, 12, 15] # max 16
cfg.weight_path = "./weights/weights.ckpt-604787"


# Train
cfg.learning_rate = 0.00015#0.000317
cfg.learning_rate = 0.00015#0.000317
cfg.lowest_lr_rate = 0.001
cfg.max_epoch = 10000
cfg.max_iter = 600000


cfg.Run_val = False
cfg.global_step_reset = False

cfg.batch_size = 12
cfg.RPN_threshold = 0.3
cfg.save_epoch = 500
cfg.loss_lambdas = [1.0, 1.0, 1.0, 0.000001] # hm , kpp, kpn, regression
cfg.train_save_folders = ['./tmp/', './tmp_val/']

cfg.threshold_extract = 0.1

# Test
cfg.gpu_num = 0 # -1 = CPU
cfg.input_path = './testset/Adam/20201016_175442/*/*/*/*.jpg'
cfg.save_folder = './results/'
cfg.thr_extract = 0.4
cfg.thr_nms = 60

cfg.video_path_list = ['./testset/2/*.jpg', # 1920 1080
                       './testset/Adam/210310_CheonAn/*.jpg', # 1280 720
                       './testset/Adam/20201016_165443/*.jpg',
                       './testset/Adam/20201016_174038/*.jpg',
                       './testset/Adam/20201016_175442/*.jpg',
                       './testset/deagu/20200615_114419/2/*.jpg',
                       './testset/deagu/20200615_114949/2/*.jpg',
                       './testset/IMAGES/*.jpg', # 800 480
                       './testset/jpg1/*.jpg', # 864 480
                       './testset/jpg3/*.jpg',
                       './testset/jpg4/*.jpg',
                       './testset/R9/R9_20210114_NIGHT_1_png/*.png' # 800 480
                       ]