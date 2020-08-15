#coding=utf-8

import math

PI = math.acos(-1.0)

def inv_lr_scheduler(optimizer, iter_num, gamma=0.1, power=0.9):
    lr_factor = (1.0 + gamma * iter_num) ** (-power)
    for param_group in optimizer.param_groups:
        param_group["lr"] *= lr_factor

def cos_lr_scheduler(optimizer, iter_num, max_iter, eps=1e-4):
    lr_factor = (math.cos(iter_num * PI / max_iter) * (1.0 - eps ) / 2.0) + \
                (1.0 + eps) / 2.0
    for param_group in optimizer.param_groups:
        param_group["lr"] *= lr_factor

def one_vs_all_split(dataset_list):
    tar_list = []
    src_list = []
    for i in range(len(dataset_list)):
        src, tar = [], []
        for j in range(len(dataset_list)):
            if j == i:
                tar.append(dataset_list[j])
            else:
                src.append(dataset_list[j])
        tar_list.append(tar)
        src_list.append(src)
    return zip(tar_list, src_list)
