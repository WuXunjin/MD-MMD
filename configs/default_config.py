#conding=utf-8

from munch import Munch

cfg = Munch()

# network
cfg.network = "alexnet"
cfg.pretrained = True

# learning
cfg.learning_rate = 1e-4
cfg.new_layer_learning_rate = 1e-3
cfg.momentum = 0.9 # SGD momentum
cfg.max_epoch = 100

# datasets
cfg.data_root = "/home/sunh/WorkSpace/MD-MMD/data/"
cfg.src = ["amazon", "dslr"]
cfg.tar = ["webcam"]
cfg.num_class = 31

# dataloader
cfg.batch_size = 64
cfg.src_val_ratio = 0.8


