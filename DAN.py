#coding=utf-8

from network import extractor_dict, classifier_dict
import time
import torch
from torch import nn, optim
from data_loader import data_loader_dict
from utils import inv_lr_scheduler, cos_lr_scheduler


DEVICE = torch.device("cuda" if torch.cuda.is_available else "cpu")

class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()
    def forward(self, x):
        entropy =  -(nn.functional.softmax(x, dim=1) * nn.functional.log_softmax(x, dim=1)).mean()
        return entropy

class MMDLoss(nn.Module):
    def __init__(self):
        super(MMDLoss, self).__init__()

    def forward(self, src_features, tar_features):
        return 0.0

class DAN(): # based on ResNet 50
    def __init__(self, cfg):
        self.cfg = cfg

        self.features = extractor_dict.ResNet50Extractor().to(DEVICE)
        self.classifier = classifier_dict.SingleClassifier(
                in_features=self.features.out_features(),
                num_class=cfg.num_class).to(DEVICE)

    def train(self):
        start_time = time.time()
        self.features.train()
        self.classifier.train()

        # prepare data
        src_loader, tar_loader = data_loader_dict[self.cfg.dataset](self.cfg, val=False)
        src_iter_len, tar_iter_len = len(src_loader), len(tar_loader)
        print("data_size[src: {:.0f}, tar: {:.0f}]".format(len(src_loader.dataset), len(tar_loader.dataset)))

        #loss
        classifier_ciriterion = nn.CrossEntropyLoss()
        entropy_ciriterion = EntropyLoss()
        MMD_ciriterion = MMDLoss()

        #optimizer
        #optimizer = optim.SGD(
        #        self.features.get_param_groups(self.cfg.learning_rate) +
        #        self.classifier.get_param_groups(self.cfg.new_layer_learning_rate),
        #        momentum=self.cfg.momentum)
        optimizer = optim.Adam(
                self.features.get_param_groups(self.cfg.learning_rate) +
                self.classifier.get_param_groups(self.cfg.new_layer_learning_rate),
                weight_decay = 0.01)

        #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.7)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.max_iter)

        # train
        best_src_acc, best_tar_acc = 0.0, 0.0
        epoch_src_acc, epoch_tar_acc = 0.0, 0.0
        epoch_src_correct, epoch_tar_correct = 0.0, 0.0
        move_average_loss = 0.0
        move_factor = 0.9
        for _iter in range(self.cfg.max_iter):
            #inv_lr_scheduler(optimizer, _iter)
            scheduler.step()
            #cos_lr_scheduler(optimizer, _iter, self.cfg.max_iter)
            if _iter % src_iter_len == 0:
                src_iter = iter(src_loader)

                epoch_src_acc = epoch_src_correct / (src_iter_len * self.cfg.batch_size)
                print("\n" + "-" * 80)
                print("Iter[{:02d}/{:03d}] Acc[src: {:.4f}, tar: {:.4f}] Best Acc[src: {:.4f}, tar: {:.4f}]".format(
                    _iter, self.cfg.max_iter, epoch_src_acc, epoch_tar_acc, best_src_acc, best_tar_acc))
                print("-" * 80 + "\n")

                if epoch_src_acc > best_src_acc:
                    best_src_acc = epoch_src_acc
                epoch_src_correct = 0.0

            if _iter % tar_iter_len == 0:
                tar_iter = iter(tar_loader)

                epoch_tar_acc = epoch_tar_correct / (tar_iter_len * self.cfg.batch_size)
                print("\n" + "-" * 80)
                print("Iter[{:02d}/{:03d}] Acc[src: {:.4f}, tar: {:.4f}] Best Acc[src: {:.4f}, tar: {:.4f}]".format(
                    _iter, self.cfg.max_iter, epoch_src_acc, epoch_tar_acc, best_src_acc, best_tar_acc))
                print("-" * 80 + "\n")

                if epoch_tar_acc > best_tar_acc:
                    best_tar_acc = epoch_tar_acc
                epoch_tar_correct = 0.0


            X_src, y_src = src_iter.next()
            X_tar, y_tar = tar_iter.next()
            X_src, y_src = X_src.to(DEVICE), y_src.to(DEVICE)
            X_tar, y_tar = X_tar.to(DEVICE), y_tar.to(DEVICE)
            optimizer.zero_grad()

            # forward
            src_features = self.features(X_src)
            src_outputs  = self.classifier(src_features)
            tar_features = self.features(X_tar)
            tar_outputs  = self.classifier(tar_features)

            # loss
            classifier_loss = classifier_ciriterion(src_outputs, y_src)
            entropy_loss = -entropy_ciriterion(tar_outputs)
            MMD_loss = MMD_ciriterion(src_features, tar_features)

            #loss = classifier_loss
            loss = classifier_loss + \
                   entropy_loss * self.cfg.entropy_loss_weight * (self.cfg.max_iter - _iter) / self.cfg.max_iter + \
                   MMD_loss * self.cfg.MMD_loss_weight

            # optimize
            loss.backward()
            optimizer.step()

            # stat
            iter_loss = loss.item()
            move_average_loss = move_average_loss * move_factor + iter_loss * (1.0 - move_factor)

            pred_src = src_outputs.argmax(dim=1)
            pred_tar = tar_outputs.argmax(dim=1)
            epoch_src_correct += (y_src == pred_src).double().sum().item()
            epoch_tar_correct += (y_tar == pred_tar).double().sum().item()
            print("Iter[{:02d}/{:03d}] Loss[move_average_loss: {:.4f}, iter_loss: {:.4f}]".format(
                _iter, self.cfg.max_iter, move_average_loss, iter_loss))

        time_pass = time.time() - start_time
        print("Train finish in {:.0f}m {:.0f}s".format(time_pass // 60, time_pass % 60))

    def eval(self, x):
        pass


def main():
    from configs.DAN_config import cfg
    print("configs:\n", cfg)
    dan = DAN(cfg)
    dan.train()

if __name__ == "__main__":
    main()

