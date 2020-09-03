#coding=utf-8

from network import build_model
import time
import math
import torch
from torch import nn, optim
from data_loader import data_loader_dict
from utils import one_vs_all_split
from loss import EntropyLoss, MMDLoss, ClusterLoss1
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available else "cpu")

class MD_DAN(): # based on ResNet 50
    def __init__(self, cfg):
        self.cfg = cfg
        self.features, self.bottleneck_layer, self.classifier = build_model(self.cfg)

    def test(self, tar_test_loader):
        correct = 0
        sample_num = 0
        for X, y in tar_test_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            sample_num += X.size(0)
            outputs = self.eval(X)
            preds = outputs.argmax(dim=1)
            correct += (y == preds).double().sum().item()
        return correct / sample_num

    def eval(self, x):
        with torch.no_grad():
            x, _, _ = self.bottleneck_layer(self.features(x))
            x = self.classifier(x)
        return x

    def train(self):
        start_time = time.time()
        self.features.train()
        self.bottleneck_layer.train()
        self.classifier.train()

        # prepare data
        src_loader, tar_loader, tar_test_loader = data_loader_dict[self.cfg.dataset](self.cfg, val=False)
        src_iter_len, tar_iter_len = len(src_loader), len(tar_loader)
        print("data_size[src: {:.0f}, tar: {:.0f}]".format(len(src_loader.dataset), len(tar_loader.dataset)))

        #loss
        classifier_ciriterion = nn.CrossEntropyLoss()
        entropy_ciriterion = EntropyLoss()
        MMD_ciriterion = MMDLoss()

        optimizer = optim.Adam(
                self.features.get_param_groups(self.cfg.learning_rate) + \
                self.bottleneck_layer.get_param_groups(self.cfg.new_layer_learning_rate) + \
                self.classifier.get_param_groups(self.cfg.new_layer_learning_rate),
                weight_decay = 0.01)

        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.max_iter)

        # train
        best_src_acc, best_tar_acc, best_tar_test_acc = 0.0, 0.0, 0.0
        early_stop_acc = -1.0
        epoch_src_acc, epoch_tar_acc = 0.0, 0.0
        epoch_src_correct, epoch_tar_correct = 0.0, 0.0
        move_average_loss = 0.0
        move_factor = 0.9
        test_acc_list = []
        for _iter in range(self.cfg.max_iter):
            if _iter % src_iter_len == 0:
                src_iter = iter(src_loader)

                epoch_src_acc = epoch_src_correct / (src_iter_len * self.cfg.batch_size)
                if epoch_src_acc > best_src_acc:
                    best_src_acc = epoch_src_acc

                print("\n" + "-" * 100)
                print("Iter[{:02d}/{:03d}] Acc[src:{:.4f}, tar:{:.4f}] Best Acc[src:{:.4f}, tar:{:.4f}] Src Update".format(
                    _iter, self.cfg.max_iter, epoch_src_acc, epoch_tar_acc, best_src_acc, best_tar_acc))
                print("-" * 100 + "\n")
                epoch_src_correct = 0.0

            if _iter % tar_iter_len == 0 or _iter == self.cfg.max_iter - 1:
                tar_iter = iter(tar_loader)

                epoch_tar_acc = epoch_tar_correct / (tar_iter_len * self.cfg.batch_size)
                if epoch_tar_acc > best_tar_acc:
                    best_tar_acc = epoch_tar_acc

                test_tar_acc = self.test(tar_test_loader) # 每个batch结束后测试整个dataset，训练用的tar是drop last的，而且有random transform
                test_acc_list.append(test_tar_acc)
                if test_tar_acc > best_tar_test_acc:
                    best_tar_test_acc = test_tar_acc

                print("\n" + "-" * 100)
                print("Iter[{:02d}/{:03d}] Acc[src:{:.3f}, tar:{:.4f}, test:{:.4f}] Best Acc[src:{:.3f}, tar:{:.4f}, test:{:.4f}]".format(
                    _iter, self.cfg.max_iter, epoch_src_acc, epoch_tar_acc, test_tar_acc, best_src_acc, best_tar_acc, best_tar_test_acc))
                print("-" * 100 + "\n")
                epoch_tar_correct = 0.0
                if _iter > self.cfg.early_stop_iter:
                    if early_stop_acc <= 0.0:
                        early_stop_acc = test_tar_acc
                    if self.cfg.early_stop:
                        break

            X_src, y_src = src_iter.next()
            X_tar, y_tar = tar_iter.next()
            X_src, y_src = X_src.to(DEVICE), y_src.to(DEVICE)
            X_tar, y_tar = X_tar.to(DEVICE), y_tar.to(DEVICE)
            optimizer.zero_grad()

            # forward
            src_features, src_cluster_loss, src_aux_entropy_loss = self.bottleneck_layer(self.features(X_src))
            src_outputs = self.classifier(src_features)

            tar_features, tar_cluster_loss, tar_aux_entropy_loss = self.bottleneck_layer(self.features(X_tar))
            tar_outputs = self.classifier(tar_features)

            # loss
            classifier_loss = classifier_ciriterion(src_outputs, y_src)
            entropy_loss = entropy_ciriterion(tar_outputs)
            inter_MMD_loss = MMD_ciriterion(src_features, tar_features)

            loss_factor = 2.0 / (1.0 + math.exp(-10 * _iter / self.cfg.max_iter)) - 1.0
            loss = classifier_loss + \
                   entropy_loss * self.cfg.entropy_loss_weight * loss_factor + \
                   inter_MMD_loss * self.cfg.inter_MMD_loss_weight * loss_factor + \
                   (src_aux_entropy_loss + tar_aux_entropy_loss) * self.cfg.aux_entropy_loss_weight * loss_factor + \
                   (src_cluster_loss + tar_cluster_loss) * self.cfg.cluster_loss_weight * loss_factor

            # optimize
            loss.backward()
            optimizer.step()

            #lr_scheduler
            lr_scheduler.step()

            # stat
            iter_loss = loss.item()
            move_average_loss = move_average_loss * move_factor + iter_loss * (1.0 - move_factor)

            pred_src = src_outputs.argmax(dim=1)
            pred_tar = tar_outputs.argmax(dim=1)
            epoch_src_correct += (y_src == pred_src).double().sum().item()
            epoch_tar_correct += (y_tar == pred_tar).double().sum().item()
            print("Iter[{:02d}/{:03d}] Loss[M-Ave:{:.4f}\titer:{:.4f}\tCla:{:.4f}\tMMD:{:.4f}".format(
                _iter, self.cfg.max_iter, move_average_loss, iter_loss, classifier_loss,  inter_MMD_loss))
            print("Iter[{:02d}/{:03d}] Ent-Loss[aux_src:{:.4f}\taux_tar:{:.4f}\tClaEnt{:.4f}".format(
                _iter, self.cfg.max_iter, src_aux_entropy_loss, tar_aux_entropy_loss, entropy_loss))
            print("Iter[{:02d}/{:03d}] Cluster-Loss[src:{:.4f}\ttar:{:.4f}]\n".format(
                _iter, self.cfg.max_iter, src_cluster_loss, tar_cluster_loss))
        time_pass = time.time() - start_time
        print("Train finish in {:.0f}m {:.0f}s".format(time_pass // 60, time_pass % 60))
        return best_tar_test_acc, early_stop_acc, test_acc_list

def combine_exp():
    from configs.MDCL_DAN_config import cfg
    print("configs:\n", cfg)
    best_acc_dict = {}
    stop_acc_dict = {}
    test_acc_list_dict = {}
    ave = 0.0
    datasets_list = ["amazon", "dslr", "webcam"]
    for tar, src in one_vs_all_split(datasets_list):
        print("src: ", src, "tar: ", tar)
        cfg.src, cfg.tar = src, tar
        dan = MD_DAN(cfg)
        best_tar_acc, stop_tar_acc, test_acc_list = dan.train()
        test_acc_list = np.array(test_acc_list)
        ave += best_tar_acc
        best_acc_dict[" ".join(src)+"->"+" ".join(tar)] = best_tar_acc
        stop_acc_dict[" ".join(src)+"->"+" ".join(tar)] = stop_tar_acc
        test_acc_list_dict[" ".join(src)+"->"+" ".join(tar)] = test_acc_list
    print("configs:\n", cfg)
    print("best acc " + "*" * 60)
    print(best_acc_dict)
    print("stop acc " + "*" * 60)
    print(stop_acc_dict)
    ave /= len(datasets_list)
    print("best ave acc", ave)
    return test_acc_list_dict

def every_exp():
    from configs.MD_DAN_config import cfg
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
            dan = MD_DAN(cfg)
            best_tar_acc, stop_tar_acc = dan.train()
            ave += best_tar_acc
            best_acc_dict[src+"->"+tar] = best_tar_acc
            stop_acc_dict[" ".join(src)+"->"+" ".join(tar)] = stop_tar_acc
    print("configs:\n", cfg)
    print("best acc " + "*" * 60)
    print(best_acc_dict)
    print("stop acc " + "*" * 60)
    print(stop_acc_dict)
    ave /= ((len(datasets_list) * (len(datasets_list) - 1)) / 2.0)
    print("best ave acc", ave)

def main():
    #torch.manual_seed(0)
    #torch.cuda.manual_seed(0)
    combine_exp()
    #every_exp()
    return
    n = 10
    sum_dict = {}
    for i in range(n):
        print("\n" * 5, "#" * 100, "\n", i, "\n" * 5)
        test_acc_dict = combine_exp()
        for k, v in test_acc_dict.items():
            if k in sum_dict.keys():
                sum_dict[k] += v
            else:
                sum_dict[k] = v
    import matplotlib.pyplot as plt
    colors = ["r", "g", "b"]
    idx = 0
    for k, v in sum_dict.items():
        print(k, colors[idx])
        v /= n
        plt.plot(v, colors[idx])
        idx += 1
    plt.show()
    import ujson as json
    for k, v in sum_dict.items():
        sum_dict[k] = str(v)
    with open("./outputs/test_acc.txt", "w") as fin:
        fin.write(json.dumps(str(sum_dict)))
    plt.savefig("./outputs/iter_acc.png")
    return

if __name__ == "__main__":
    main()

