#coding=utf-8

def inv_lr_scheduler(optimizer, iter_num, gamma=0.1, power=0.9):
    lr_factor = (1.0 + gamma * iter_num) ** (-power)
    for param_group in optimizer.param_groups:
        param_group["lr"] *= lr_factor

