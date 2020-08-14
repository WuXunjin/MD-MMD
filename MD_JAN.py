#coding=utf-8

from network import extractor_dict, classifier_dict
import time
import math
import torch
from torch import nn, optim
from data_loader import data_loader_dict
from utils import inv_lr_scheduler, cos_lr_scheduler
from loss import EntropyLoss, MMDLoss, JMMDLoss

DEVICE = torch.device("cuda" if torch.cuda.is_available else "cpu")

class MD_MMD_Layer(nn.Module):
    def __init__(self, in_features, out_features, latent_domain_num=2):
        super(MD_MMD_Layer, self).__init__()
        self.latent_domain_num = latent_domain_num
        self.in_features = in_features
        self.out_features = out_features
        self.aux_classifier = nn.Sequential(
            nn.Linear(self.in_features, self.latent_domain_num),
            nn.Softmax()
            )
        self.layers = []
        for i in range(self.latent_domain_num):
            self.layers.append(nn.Linear(self.in_features, self.out_features).to(DEVICE))
        self.MMD_ciriterion = MMDLoss()

    def inter_MMD(self, outputs_list):
        loss = 0.0
        for i in range(self.latent_domain_num):
            for j in range(i + 1, self.latent_domain_num):
                loss += self.MMD_ciriterion(outputs_list[i], outputs_list[j])
        return loss * 2.0 / (self.latent_domain_num ** 2 - self.latent_domain_num)

    def forward(self, x):
        #combine loss
        batch_size = int(x.size()[0])
        features_size = self.out_features
        latent_domain_label = self.aux_classifier(x)
        outputs = [layer(x) for layer in self.layers]
        stack_outputs = torch.stack(outputs, dim=2)
        expand_latent_domain_label = latent_domain_label.unsqueeze(1).expand(batch_size, features_size, self.latent_domain_num)
        combine_outputs = (expand_latent_domain_label * stack_outputs).sum(2)
        inter_MMD_loss = -self.inter_MMD(outputs)
        return combine_outputs, inter_MMD_loss

    def get_param_groups(self, learning_rate):
        param_groups = [{"params": self.aux_classifier.parameters(), "lr": learning_rate}]
        for layer in self.layers:
            param_groups.append({"params": layer.parameters(), "lr": learning_rate})
        return param_groups


class MD_JAN(): # based on ResNet 50
    def __init__(self, cfg):
        self.cfg = cfg

        self.features = extractor_dict.ResNet50Extractor().to(DEVICE)
        self.bottleneck_layer = nn.Linear(self.features.out_features(), 256).to(DEVICE)
        self.classifier = MD_MMD_Layer(
                in_features=256,
                out_features=cfg.num_class,
                latent_domain_num=self.cfg.latent_domain_num).to(DEVICE)
        self.softmax = nn.Softmax(dim=1).to(DEVICE)

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
        JMMD_ciriterion = JMMDLoss()

        optimizer = optim.Adam(
                self.features.get_param_groups(self.cfg.learning_rate) +
                [{"params": self.bottleneck_layer.parameters(), "lr": self.cfg.new_layer_learning_rate}] + \
                self.classifier.get_param_groups(self.cfg.new_layer_learning_rate),
                weight_decay = 0.01)

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

            if _iter % src_iter_len == 0:
                src_iter = iter(src_loader)

                epoch_src_acc = epoch_src_correct / (src_iter_len * self.cfg.batch_size)
                if epoch_src_acc > best_src_acc:
                    best_src_acc = epoch_src_acc

                print("\n" + "-" * 90)
                print("Iter[{:02d}/{:03d}] Acc[src: {:.4f}, tar: {:.4f}] Best Acc[src: {:.4f}, tar: {:.4f}] Src Update".format(
                    _iter, self.cfg.max_iter, epoch_src_acc, epoch_tar_acc, best_src_acc, best_tar_acc))
                print("-" * 90 + "\n")
                epoch_src_correct = 0.0

            if _iter % tar_iter_len == 0:
                tar_iter = iter(tar_loader)

                epoch_tar_acc = epoch_tar_correct / (tar_iter_len * self.cfg.batch_size)
                if epoch_tar_acc > best_tar_acc:
                    best_tar_acc = epoch_tar_acc

                print("\n" + "-" * 90)
                print("Iter[{:02d}/{:03d}] Acc[src: {:.4f}, tar: {:.4f}] Best Acc[src: {:.4f}, tar: {:.4f}] Tar Update".format(
                    _iter, self.cfg.max_iter, epoch_src_acc, epoch_tar_acc, best_src_acc, best_tar_acc))
                print("-" * 90 + "\n")
                epoch_tar_correct = 0.0


            X_src, y_src = src_iter.next()
            X_tar, y_tar = tar_iter.next()
            X_src, y_src = X_src.to(DEVICE), y_src.to(DEVICE)
            X_tar, y_tar = X_tar.to(DEVICE), y_tar.to(DEVICE)
            optimizer.zero_grad()

            # forward
            src_features = self.bottleneck_layer(self.features(X_src))
            src_outputs, src_inter_MMD_loss  = self.classifier(src_features)
            src_softmax = self.softmax(src_outputs)

            tar_features = self.bottleneck_layer(self.features(X_tar))
            tar_outputs, tar_inter_MMD_loss  = self.classifier(tar_features)
            tar_softmax = self.softmax(tar_outputs)

            # loss
            classifier_loss = classifier_ciriterion(src_outputs, y_src)
            entropy_loss = entropy_ciriterion(tar_outputs)
            intra_MMD_loss = JMMD_ciriterion([src_features, src_softmax], [tar_features, tar_softmax])

            #loss_factor = 2.0 / (1.0 + math.exp(-10 * _iter / self.cfg.max_iter)) - 1.0
            loss_factor = 1.0
            loss = classifier_loss + \
                   entropy_loss * self.cfg.entropy_loss_weight * loss_factor + \
                   intra_MMD_loss * self.cfg.intra_MMD_loss_weight * loss_factor + \
                   (src_inter_MMD_loss + tar_inter_MMD_loss) * self.cfg.inter_MMD_loss_weight * loss_factor

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
            print("Iter[{:02d}/{:03d}] Loss[M-Ave:{:.4f}\titer:{:.4f}\tCla:{:.4f}\tEnt:{:.4f}".format(
                _iter, self.cfg.max_iter, move_average_loss, iter_loss, classifier_loss, entropy_loss))
            print("Iter[{:02d}/{:03d}] MMD-Loss[intra:{:.4f}\tinter_src:{:.4f}\tinter_src:{:.4f}\n".format(
                _iter, self.cfg.max_iter, intra_MMD_loss, src_inter_MMD_loss, tar_inter_MMD_loss))

        time_pass = time.time() - start_time
        print("Train finish in {:.0f}m {:.0f}s".format(time_pass // 60, time_pass % 60))
        return best_tar_acc

    def eval(self, x):
        return self.classifier(self.features(x))


def combine_exp():
    from configs.MD_JAN_config import cfg
    print("configs:\n", cfg)
    acc_dict = {}
    ave = 0.0
    for src, tar in zip([["amazon", "webcam"], ["amazon", "dslr"], ["webcam", "dslr"]],[["dslr"],["webcam"], ["amazon"]]):
        print("src: ", src, "tar: ", tar)
        cfg.src, cfg.tar = src, tar
        jan = MD_JAN(cfg)
        best_tar_acc = jan.train()
        ave += best_tar_acc
        acc_dict[" ".join(src)+"->"+tar[0]] = best_tar_acc
    print("configs:\n", cfg)
    print(acc_dict)
    ave /= 3.0
    print("ave acc", ave)

def every_exp():
    from configs.MD_DAN_config import cfg
    print("configs:\n", cfg)
    acc_dict = {}
    ave = 0.0
    for src in ["amazon", "dslr", "webcam"]:
        for tar in ["amazon", "dslr", "webcam"]:
            if src == tar:
                continue
            print("src: ", src, "tar: ", tar)
            cfg.src, cfg.tar = [src], [tar]
            jan = MD_JAN(cfg)
            best_tar_acc = jan.train()
            ave += best_tar_acc
            acc_dict[src+"->"+tar] = best_tar_acc
    print("configs:\n", cfg)
    print(acc_dict)
    ave /= 6.0
    print("ave acc", ave)

def main():
    #combine_exp()
    #every_exp()
    from configs.MD_DAN_config import cfg
    print("configs:\n", cfg)
    jan = MD_JAN(cfg)
    jan.train()
    print("configs:\n", cfg)


if __name__ == "__main__":
    main()

