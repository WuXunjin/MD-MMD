#conding=utf-8

from munch import Munch

cfg = Munch()

# network
cfg.extractor = "AlexNetExtractor"
cfg.classifier = "SingleClassifier"

# learning
cfg.learning_rate = 1e-4
cfg.new_layer_learning_rate = 1e-3
cfg.momentum = 0.9 # SGD momentum
cfg.max_epoch = 50

# datasets
cfg.dataset = "office31"
cfg.data_root = "/home/sunh/WorkSpace/MD-MMD/data/office31/"
cfg.src = ["amazon", "webcam"]
cfg.tar = ["dslr"]
cfg.num_class = 31

# dataloader
cfg.batch_size = 128
cfg.src_val_ratio = 0.95


