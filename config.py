import numpy as np

class Config(object):
    pass

cfg = Config()
cfg.batch_size = 32
cfg.height = 480
cfg.width = 800
cfg.rate = 8.0
cfg.weight_path = './pretrained/weights.ckpt-96252'
