#coding=utf-8

import time
import torchvision
import torch
from torch import nn, optim
from network import extractor_dict, classifier_dict
from data_loader import data_loader_dict
from utils import inv_lr_scheduler
#from tensorboardX import SummaryWriter

DEVICE = torch.device("cuda" if torch.cuda.is_available else "cpu")

class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()
        self.cfg = cfg

        if self.cfg.network == "alexnet":
            self.features = extractor_dict.AlexNetExtractor()
            self.classifier = classifier_dict.AlexNetClassifier(in_features=self.features.out_features(), num_class=cfg.num_class)
        elif self.cfg.network == "resnet18":
            self.features = extractor_dict.ResNet18Extractor()
            self.classifier = classifier_dict.SingleClassifier(in_features=self.features.out_features(), num_class=cfg.num_class)
        elif self.cfg.network == "resnet34":
            self.features = extractor_dict.ResNet34Extractor()
            self.classifier = classifier_dict.SingleClassifier(in_features=self.features.out_features(), num_class=cfg.num_class)
        elif self.cfg.network == "resnet50":
            self.features = extractor_dict.ResNet50Extractor()
            self.classifier = classifier_dict.SingleClassifier(in_features=self.features.out_features(), num_class=cfg.num_class)
        elif self.cfg.network == "resnet101":
            self.features = extractor_dict.ResNet101Extractor()
            self.classifier = classifier_dict.SingleClassifier(in_features=self.features.out_features(), num_class=cfg.num_class)
        elif self.cfg.network == "resnet152":
            self.features = extractor_dict.ResNet152Extractor()
            self.classifier = classifier_dict.SingleClassifier(in_features=self.features.out_features(), num_class=cfg.num_class)
        elif self.cfg.network == "inception":
            self.features = extractor_dict.InceptionExtractor()
            self.classifier = classifier_dict.SingleClassifier(in_features=self.features.out_features(), num_class=cfg.num_class)


        self.param_groups = self.features.get_param_groups(cfg.learning_rate)
        if cfg.network == "alexnet":
            self.param_groups += self.classifier.get_param_groups(cfg.learning_rate, cfg.new_layer_learning_rate)
        else:
            self.param_groups += self.classifier.get_param_groups(cfg.new_layer_learning_rate)


    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_param_groups(self):
        return self.param_groups


def fine_tune(net, data_loaders, cfg):
    start_time = time.time()
    best_tar_acc, best_val_acc, tar_acc_when_best_val = 0.0, 0.0, 0.0
    classifier_ciriterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.get_param_groups(), momentum=cfg.momentum)
    for epoch in range(1, cfg.max_epoch + 1):
        inv_lr_scheduler(optimizer, epoch)
        acc_dict = {}
        for phase in ["src", "val", "tar"]:
            if phase == "src":
                net.train()
            else:
                net.eval()
            epoch_loss, epoch_correct = 0, 0
            for X, y in data_loaders[phase]:
                X, y = X.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "src"):
                    outputs = net(X)
                    loss = classifier_ciriterion(outputs, y)
                preds = outputs.argmax(dim=1)
                if phase == "src":
                    loss.backward()
                    optimizer.step()
                epoch_loss += loss.item() * X.size(0)
                epoch_correct += (preds == y).double().sum().item()
            epoch_loss /= len(data_loaders[phase].dataset)
            epoch_acc = epoch_correct / len(data_loaders[phase].dataset)
            acc_dict[phase] = epoch_acc
            print("Epoch[{:02d}/{:03d}]---{}, loss: {:.6f}, acc: {:.4f}".format(
                epoch, cfg.max_epoch, phase, epoch_loss, epoch_acc))
        if acc_dict["tar"] > best_tar_acc:
            best_tar_acc = acc_dict["tar"]
        if acc_dict["val"] > best_val_acc:
            best_val_acc = acc_dict["val"]
            tar_acc_when_best_val = acc_dict["tar"]
        print("Acc --- best tar: {:.4f}, best val: {:.4f}, tar when best val: {:.4f}".format(
            best_tar_acc, best_val_acc, tar_acc_when_best_val))
        print()
    time_pass = time.time() - start_time

    print("Train finish in {:.0f}m {:.0f}s".format(time_pass // 60, time_pass % 60))

def main():
    from configs.fine_tune_config import cfg
    print("config:\n", cfg)
    print("-" * 60)
    train_loader, val_loader, tar_loader = data_loader_dict[cfg.dataset](cfg)
    print("dataset:", cfg.dataset)
    print("src_train: ", len(train_loader.dataset), "src_val: ", len(val_loader.dataset), "tar: ", len(tar_loader.dataset))
    net = Net(cfg).to(DEVICE)
    #print("Network:\n", net)
    print("-" * 60)

    fine_tune(net, {"src": train_loader, "val": val_loader, "tar": tar_loader}, cfg)
    print("config:\n", cfg)

if __name__ == "__main__":
    main()
