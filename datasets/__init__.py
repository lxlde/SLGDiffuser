import os
import torch
import numbers
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.datasets import CIFAR10
from datasets.celeba import CelebA
from datasets.ffhq import FFHQ
from datasets.lsun import LSUN
from datasets.word import Word, Word_Test, Word_Test1, Word_Test2, Word_Test4
from torch.utils.data import Subset
import numpy as np


class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )


def get_dataset(args, config):
    if config.data.random_flip is False:
        tran_transform = test_transform = transforms.Compose(
            [transforms.Resize(config.data.image_size), transforms.ToTensor()]
        )
    else:
        tran_transform = transforms.Compose(
            [
                transforms.Resize(config.data.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.Compose(
            [transforms.Resize(config.data.image_size), transforms.ToTensor()]
        )

    if config.data.dataset == "CIFAR10":
        dataset = CIFAR10(
            os.path.join(args.exp, "datasets", "cifar10"),
            train=True,
            download=True,
            transform=tran_transform,
        )
        test_dataset = CIFAR10(
            os.path.join(args.exp, "datasets", "cifar10_test"),
            train=False,
            download=True,
            transform=test_transform,
        )
    elif config.data.dataset == "WORD" and args.sample == False:
        dataset = Word(
            root_content = '../../../data/lxl/srnet_datagen1/srnet_data_new6',
            transform=transforms.Compose([
                transforms.Resize(config.data.image_size),
                transforms.ToTensor()]
            )
        )
        return dataset
    elif config.data.dataset == "WORD" and args.sample == True:
        dataset = Word_Test(
            root_content = r'../ddim_mask_img/exp/datasets',
            transform=transforms.Compose([
                transforms.Resize(config.data.image_size),
                transforms.ToTensor()]
            )
        )
        return dataset
    elif config.data.dataset == "WORD1":
        dataset = Word_Test1(
            root_content=r'../ddim_mask_img/exp/datasets/final_test1',
            transform=transforms.Compose([
                transforms.Resize(config.data.image_size),
                transforms.ToTensor()]
            )
        )
        return dataset
    elif config.data.dataset == "WORD4":
        dataset = Word_Test4(
            root_content=r'../ddim_mask_img/exp/datasets/final_test1',
            transform=transforms.Compose([
                transforms.Resize(config.data.image_size),
                transforms.ToTensor()]
            )
        )
        return dataset
    else:
        dataset = Word_Test2(
            root_content=r'../ddim_mask_img/exp/datasets/效果展示_t',
            transform=transforms.Compose([
                transforms.Resize(config.data.image_size),
                transforms.ToTensor()]
            )
        )
        return dataset

    return dataset, test_dataset


def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


def data_transform(config, X):
    if config.data.uniform_dequantization:
        X = X / 256.0 * 255.0 + torch.rand_like(X) / 256.0
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if config.data.rescaled:
        X = 2 * X - 1.0
    elif config.data.logit_transform:
        X = logit_transform(X)

    if hasattr(config, "image_mean"):
        return X - config.image_mean.to(X.device)[None, ...]

    return X


def inverse_data_transform(config, X):
    if hasattr(config, "image_mean"):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.0) / 2.0

    return torch.clamp(X, 0.0, 1.0)
