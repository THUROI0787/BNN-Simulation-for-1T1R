"""
Dataset builders for CIFAR-10, CIFAR-100, and MNIST.
"""

import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
import warnings

warnings.filterwarnings('ignore')


def build_cifar(cutout=False, use_cifar10=True, download=True, data_root='./data'):
    """Build CIFAR-10 or CIFAR-100 train/val datasets."""
    aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    aug.append(transforms.ToTensor())

    if cutout:
        aug.append(transforms.RandomErasing(p=1.0, scale=(0.02, 0.1)))

    if use_cifar10:
        aug.append(
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        )
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = CIFAR10(root=data_root, train=True, download=download,
                                transform=transform_train)
        val_dataset = CIFAR10(root=data_root, train=False, download=download,
                              transform=transform_test)
    else:
        aug.append(
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        )
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        train_dataset = CIFAR100(root=data_root, train=True, download=download,
                                 transform=transform_train)
        val_dataset = CIFAR100(root=data_root, train=False, download=download,
                               transform=transform_test)

    return train_dataset, val_dataset


def build_mnist(download=True, data_root='./data'):
    """Build MNIST train/val datasets."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_dataset = MNIST(root=data_root, train=True, download=download,
                          transform=transform)
    val_dataset = MNIST(root=data_root, train=False, download=download,
                        transform=transform)
    return train_dataset, val_dataset
