#-conding=utf-8

from munch import Munch

cfg = Munch()

# network
cfg.extractor = "AlexNetExtractor"
cfg.classifier = "NewAlexNetClassifier"
cfg.latent_domain_num = 1
cfg.bottleneck_size = 2048

# datasets
cfg.dataset = "office31"
cfg.data_root = "/home/sunh/WorkSpace/MD-MMD/data/" + cfg.dataset + "/"
cfg.src = ["dslr", "webcam"]
cfg.tar = ["amzaon"]
cfg.num_class = 31

# learning
cfg.learning_rate = 1e-4
cfg.new_layer_learning_rate = 1e-3
cfg.max_iter = 2000
cfg.early_stop_iter = 500
cfg.early_stop = False

#loss
cfg.mmd_type = "mmd"
#cfg.entropy_loss_weight = 1.0
cfg.entropy_loss_weight = 0.0
cfg.inter_MMD_loss_weight = 0.0
#cfg.cluster_loss_weight = 1.68
cfg.cluster_loss_weight = 0.0
#cfg.aux_entropy_loss_weight = 0.8
cfg.aux_entropy_loss_weight = 0.0
cfg.loss_factor_decay = 0

# dataloader
cfg.batch_size = 64
cfg.src_val_ratio = 0.8

