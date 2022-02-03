'''
Config File In Tensorflow Template
By Kwanwoo Park, 2022.
'''

class Config(object):
    pass

cfg = Config()

cfg.height = 32
cfg.width = 32
cfg.batch_size = 4
cfg.class_num = 10
cfg.gpus = [0, 1, 2, 3]
cfg.train_list = "./data/train.json"
cfg.val_list = "./data/val.json"
cfg.input_path = "./testset/*"

cfg.learning_rate = 0.001
cfg.max_iter = 10000
cfg.lowest_lr_rate = 0.01
cfg.load_weight_path = None
cfg.max_epoch = 100
cfg.val_iter = 500
cfg.save_iter = 100
cfg.GS_reset = False