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
        entropy = -(nn.functional.softmax(x, dim=1) * nn.functional.log_softmax(x, dim=1)).mean()
        return entropy

class MMDLoss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5, fix_sigma=None, mmd_type="mmd"):
        super(MMDLoss, self).__init__()
        self.kernel_mul = 2.0
        self.kernel_num = kernel_num
        self.fix_sigma  = fix_sigma
        self.mmd_type = mmd_type

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
        if self.mmd_type == "mmd":
            return self.mmd(src_features, tar_features)
        elif self.mmd_type == "mmd_linear":
            return self.mmd_linear(src_features, tar_features)

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


class MD_DAN(): # based on ResNet 50
    def __init__(self, cfg):
        self.cfg = cfg

        self.features = extractor_dict.ResNet50Extractor().to(DEVICE)
        self.classifier = MD_MMD_Layer(
                in_features=self.features.out_features(),
                out_features=cfg.num_class,
                latent_domain_num=self.cfg.latent_domain_num).to(DEVICE)

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

        optimizer = optim.Adam(
                self.features.get_param_groups(self.cfg.learning_rate) +
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
            src_features = self.features(X_src)
            src_outputs, src_inter_MMD_loss  = self.classifier(src_features)
            tar_features = self.features(X_tar)
            tar_outputs, tar_inter_MMD_loss  = self.classifier(tar_features)

            # loss
            classifier_loss = classifier_ciriterion(src_outputs, y_src)
            entropy_loss = entropy_ciriterion(tar_outputs)
            intra_MMD_loss = MMD_ciriterion(src_features, tar_features)

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
        pass


def main():
    from configs.MD_DAN_config import cfg
    print("configs:\n", cfg)
    acc_dict = {}
    ave = 0.0
    for src in ["amazon", "webcam", "dslr"]:
        for tar in ["amazon", "webcam", "dslr"]:
            if src == tar:
                continue
            cfg.src, cfg.tar = [src], [tar]
            dan = MD_DAN(cfg)
            best_tar_acc = dan.train()
            ave += best_tar_acc
            acc_dict[src+"->"+"tar"] = best_tar_acc
    print("configs:\n", cfg)
    print(acc_dict)
    ave /= 6.0
    print("ave acc", ave)


if __name__ == "__main__":
    main()

