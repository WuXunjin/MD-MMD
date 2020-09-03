#coding=utf-8

from network import extractor_dict, classifier_dict
import time
import math
import torch
from torch import nn, optim
from data_loader import data_loader_dict
from utils import one_vs_all_split
from loss import EntropyLoss, MMDLoss, ClusterLoss1
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available else "cpu")

class Net():
    def __init__(self, cfg):
        self.cfg = cfg
        self.features = extractor_dict[cfg.extractor]().to(DEVICE)
        self.classifier = classifier_dict[cfg.classifier](in_features=self.features.out_features(),
            num_class=cfg.num_class).to(DEVICE)

    def test(self, tar_test_loader):
        correct = 0
        for X, y in tar_test_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            outputs = self.eval(X)
            preds = outputs.argmax(dim=1)
            correct += (y == preds).double().sum().item()
        return correct / len(tar_test_loader.dataset)

    def eval(self, x):
        with torch.no_grad():
            x = self.classifier(self.features(x))
        return x

    def train(self):
        start_time = time.time()
        self.features.train()
        self.classifier.train()

        # prepare data
        src_loader, _, tar_test_loader = data_loader_dict[self.cfg.dataset](self.cfg, val=False)
        dataloaders = {"src": src_loader, "tar": tar_test_loader}
        print("data_size[src: {:.0f}, tar: {:.0f}]".format(len(src_loader.dataset), len(tar_test_loader.dataset)))

        #loss
        classifier_ciriterion = nn.CrossEntropyLoss()

        param_groups = self.features.get_param_groups(self.cfg.learning_rate)
        if self.cfg.classifier == "AlexNetClassifier":
            param_groups += self.classifier.get_param_groups(self.cfg.learning_rate, self.cfg.new_layer_learning_rate)
        else:
            param_groups += self.classifier.get_param_groups(self.cfg.new_layer_learning_rate)
        optimizer = optim.Adam(param_groups, weight_decay = 0.01)

        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.max_epoch)

        # train
        best_acc = {"src":0.0, "tar":0.0}
        best_test_acc = 0.0
        for epoch in range(self.cfg.max_epoch):
            epoch_loss = 0.0
            epoch_correct = 0
            sample_num = 0
            for phase in ["src", "tar"]:
                for X, y in dataloaders[phase]:
                    X, y = X.to(DEVICE), y.to(DEVICE)
                    sample_num += X.size(0)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == "src"):
                        outputs = self.classifier(self.features(X))
                        loss = classifier_ciriterion(outputs, y)
                    preds = outputs.argmax(dim=1)
                    if phase == "src":
                        loss.backward()
                        optimizer.step()
                        lr_scheduler.step()
                    epoch_loss += loss.item() * X.size(0)
                    epoch_correct += (preds == y).double().sum().item()
                epoch_loss /= sample_num
                epoch_acc = epoch_correct / sample_num
                if epoch_acc > best_acc[phase]:
                    best_acc[phase] = epoch_acc
                print("Epoch[{:02d}/{:03d}]---{} loss: {:.4f} acc: {:.4f} best acc: {:.4f}".format(
                    epoch, self.cfg.max_epoch, phase, loss, epoch_acc, best_acc[phase]))
            test_acc = self.test(tar_test_loader)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
            print("Epoch[{:02d}/{:03d}]---test loss: {:.4f} acc: {:.4f} best acc: {:.4f}".format(
                    epoch, self.cfg.max_epoch, 0.0, test_acc, best_test_acc))
            print()
        return best_acc["tar"]


def combine_exp():
    from configs.fine_tune_config import cfg
    print("configs:\n", cfg)
    best_acc_dict = {}
    ave = 0.0
    datasets_list = ["amazon", "dslr", "webcam"]
    for tar, src in one_vs_all_split(datasets_list):
        print("src: ", src, "tar: ", tar)
        cfg.src, cfg.tar = src, tar
        net = Net(cfg)
        best_tar_acc = net.train()
        best_acc_dict[" ".join(src)+"->"+" ".join(tar)] = best_tar_acc
    print("configs:\n", cfg)
    print("best acc " + "*" * 60)
    print(best_acc_dict)
    ave /= len(datasets_list)

def every_exp():
    from configs.fine_tune_config import cfg
    print("configs:\n", cfg)
    best_acc_dict = {}
    stop_acc_dict = {}
    ave = 0.0
    datasets_list = ["amazon", "dslr", "webcam"]
    for src in datasets_list:
        for tar in datasets_list:
            if src == tar:
                continue
            print("src: ", src, "tar: ", tar)
            cfg.src, cfg.tar = [src], [tar]
            net = Net(cfg)
            best_tar_acc = net.train()
            ave += best_tar_acc
            best_acc_dict[src+"->"+tar] = best_tar_acc
    print("configs:\n", cfg)
    print("best acc " + "*" * 60)
    print(best_acc_dict)
    ave /= ((len(datasets_list) * (len(datasets_list) - 1)) / 2.0)
    print("best ave acc", ave)

def main():
    combine_exp()
    #every_exp()
    return
    #from configs.MD_DAN_config import cfg
    #print("configs:\n", cfg)
    #dan = MD_DAN(cfg)
    #dan.train()
    #print("configs:\n", cfg)

if __name__ == "__main__":
    main()

