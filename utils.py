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
