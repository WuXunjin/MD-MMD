#coding=utf-8

from network import extractor_dict, classifier_dict
import time
import math
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
    def __init__(self, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super(MMDLoss, self).__init__()
        self.kernel_mul = 2.0
        self.kernel_num = kernel_num
        self.fix_sigma  = fix_sigma

    def guassian_kernel(self, src_features, tar_features):
        n_samples = int(src_features.size()[0] + tar_features.size()[0])
        total = torch.cat([src_features, tar_features], dim=0)
        double_batch = int(total.size()[0])
        features_num = int(total.size()[1])
        total0 = total.unsqueeze(0).expand(double_batch, double_batch, features_num)
        total1 = total.unsqueeze(1).expand(double_batch, double_batch, features_num)
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if self.fix_sigma:
            bandwidth = self.fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul ** i) for i in range(self.kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_tmp) for bandwidth_tmp in bandwidth_list]
        return sum(kernel_val)

    def mmd(self, src_features, tar_features):
        batch_size = int(src_features.size()[0])
        kernel = self.guassian_kernel(src_features, tar_features)
        XX=kernel[:batch_size, :batch_size]
        YY=kernel[batch_size:, batch_size:]
        XY=kernel[:batch_size, batch_size:]
        YX=kernel[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return loss

    def mmd_linear(self, src_features, tar_features):
        batch_size = int(src_features.size()[0])
        kernel = self.guassian_kernel(src_features, tar_features)
        loss = 0.0
        for i in range(batch_size):
            s1, s2 = i, (i + 1) % batch_size
            t1, t2 = s1 + batch_size, s2 + batch_size
            loss += (kernel[s1, s2] + kernel[t1, t2])
            loss -= (kernel[s1, t2] + kernel[s2, t1])
        return loss / float(batch_size)

    def forward(self, src_features, tar_features):
        return self.mmd(src_features, tar_features)
        #return self.mmd_linear(src_features, tar_features)

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
            # lr scheduler
            scheduler.step()
            #inv_lr_scheduler(optimizer, _iter)
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
            entropy_loss = entropy_ciriterion(tar_outputs)
            MMD_loss = MMD_ciriterion(src_features, tar_features)

            loss_factor = 2.0 / (1.0 + math.exp(-10 * _iter / self.cfg.max_iter)) - 1.0
            #loss_factor = 1.0
            loss = classifier_loss + \
                   entropy_loss * self.cfg.entropy_loss_weight * loss_factor + \
                   MMD_loss * self.cfg.MMD_loss_weight * loss_factor

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
            print("Iter[{:02d}/{:03d}] Loss[M-Ave: {:.4f}\titer:{:.4f}\tCla:{:.4f}\tEnt:{:.4f}\tMMD:{:.4f}]".format(
                _iter, self.cfg.max_iter, move_average_loss, iter_loss, classifier_loss, entropy_loss, MMD_loss))

        time_pass = time.time() - start_time
        print("Train finish in {:.0f}m {:.0f}s".format(time_pass // 60, time_pass % 60))

    def eval(self, x):
        pass


def main():
    from configs.DAN_config import cfg
    print("configs:\n", cfg)
    dan = DAN(cfg)
    dan.train()
    print("configs:\n", cfg)


if __name__ == "__main__":
    main()

