#coding=utf-8

from torchvision import datasets, transforms
import torch
from torch.utils.data import ConcatDataset, DataLoader, random_split

def office31_loader(cfg, val=True):
    img_size = 299 if cfg.network == "inception" else 224
    src_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
            )
        ])
    tar_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
            )
        ])

    datasets_src, datasets_tar = [], []
    for src in cfg.src:
        datasets_src.append(datasets.ImageFolder(root=cfg.data_root + src, transform=src_transform))

    for tar in cfg.tar:
        datasets_tar.append(datasets.ImageFolder(root=cfg.data_root + tar, transform=tar_transform))

    data_src = ConcatDataset(datasets_src)
    data_tar = ConcatDataset(datasets_tar)

    tar_loader = DataLoader(data_tar, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    if val:
        train_size = int(cfg.src_val_ratio*len(data_src))
        val_size   = len(data_src) - train_size
        data_train, data_val = random_split(data_src, [train_size, val_size])
        train_loader = DataLoader(data_train, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
        val_loader   = DataLoader(data_val, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
        return train_loader, val_loader, tar_loader
    else:
        train_loader = DataLoader(data_src, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
        return train_loader, tar_loader


def digits_loader(cfg, val=True):
    pass


data_loader_dict = {"office31": office31_loader}
