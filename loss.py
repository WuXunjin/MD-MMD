#zcoding=utf-8

import numpy as np
import torch
from torch import nn

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
        self.kernel_mul = kernel_mul
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

class JMMDLoss(nn.Module):
    def __init__(self, kernel_mul=[2.0, 2.0], kernel_num=[5, 1], fix_sigma=[None, 1.68], mmd_type="jmmd"):
        super(JMMDLoss, self).__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma  = fix_sigma
        self.mmd_type = mmd_type

    def guassian_kernel(self, src_features, tar_features, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(src_features.size()[0] + tar_features.size()[0])
        total = torch.cat([src_features, tar_features], dim=0)
        double_batch = int(total.size()[0])
        features_num = int(total.size()[1])
        total0 = total.unsqueeze(0).expand(double_batch, double_batch, features_num)
        total1 = total.unsqueeze(1).expand(double_batch, double_batch, features_num)
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_tmp) for bandwidth_tmp in bandwidth_list]
        return sum(kernel_val)

    def joint_kernel(self, src_features_list, tar_features_list):
        layer_num = len(src_features_list)
        joint_kernel_val = None
        for i in range(layer_num):
            kernel_val = self.guassian_kernel(src_features_list[i], tar_features_list[i],
                    self.kernel_mul[i], self.kernel_num[i], self.fix_sigma[i])
            if joint_kernel_val is None:
                joint_kernel_val = kernel_val
            else:
                joint_kernel_val *= kernel_val
        return joint_kernel_val


    def jmmd(self, src_features_list, tar_features_list):
        batch_size = int(src_features_list[0].size()[0])
        kernel = self.joint_kernel(src_features_list, tar_features_list)
        XX=kernel[:batch_size, :batch_size]
        YY=kernel[batch_size:, batch_size:]
        XY=kernel[:batch_size, batch_size:]
        YX=kernel[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return loss

    def jmmd_linear(self, src_features_list, tar_features_list):
        batch_size = int(src_features_list[0].size()[0])
        kernel = self.joint_kernel(src_features_list, tar_features_list)
        loss = 0.0
        for i in range(batch_size):
            s1, s2 = i, (i + 1) % batch_size
            t1, t2 = s1 + batch_size, s2 + batch_size
            loss += (kernel[s1, s2] + kernel[t1, t2])
            loss -= (kernel[s1, t2] + kernel[s2, t1])
        return loss / float(batch_size)

    def forward(self, src_features_list, tar_features_list):
        if self.mmd_type == "jmmd":
            return self.jmmd(src_features_list, tar_features_list)
        elif self.mmd_type == "jmmd_linear":
            return self.jmmd_linear(src_features_list, tar_features_list)


class ClusterLoss1(nn.Module): # 直接对输出的features进行聚类
    def __init__(self, alpha=1.0):
        super(ClusterLoss1, self).__init__()
        self.alpha = alpha

    def forward(self, stack_outputs, latent_domain_label):
        (batch_size, features_num, domain_num) = stack_outputs.size()
        center = (stack_outputs * latent_domain_label.unsqueeze(1).expand(
                batch_size, features_num, domain_num)).mean(dim=0)
        cluster_loss_matrix = ((stack_outputs.unsqueeze(3).expand(
            batch_size, features_num, domain_num, domain_num) - \
            (center.unsqueeze(0).expand(batch_size, features_num, domain_num)).unsqueeze(
                2).expand(batch_size, features_num, domain_num, domain_num)) ** 2.0).sum(1)
        pro_sum = ((cluster_loss_matrix / self.alpha + 1.0) ** ((self.alpha + 1.0) / 2.0)).sum(2)
        pro_belong = (cluster_loss_matrix / self.alpha + 1.0) ** ((self.alpha + 1.0) / 2.0)
        pro_belong /= pro_sum.unsqueeze(2).expand(batch_size, domain_num, domain_num)
        cluster_loss = torch.log(pro_belong).mean(0)
        sig_matrix = torch.Tensor(np.diag([-2.0] * domain_num) + np.ones((domain_num, domain_num))).to(DEVICE)
        cluster_loss = (cluster_loss_matrix * sig_matrix).mean()
        return cluster_loss


