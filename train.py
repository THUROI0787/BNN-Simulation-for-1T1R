"""
Unified training script for Binary Neural Networks (BNN) on CIFAR-10 and MNIST.

Supports:
  - CIFAR-10: Net_00, Net_01, Net_11, Net_small, Net_small2, Net_mid
  - CIFAR-10 + 1T1R FC: Net_mid2 (2-stage), Net_mid3 (2-stage)
  - MNIST: Net_00_MNIST, Net_01_MNIST, Net_11_MNIST

Examples:
  # Train Net_01 on CIFAR-10
  python train.py --dataset cifar10 --arch Net_01 --epochs 200 --lr 0.01 --log my_exp

  # Train Net_mid2 with 2-stage 1T1R training
  python train.py --dataset cifar10 --arch Net_mid2 --epochs 200 --epochs_finetune 50 --log mid2_exp

  # Train Net_01_MNIST on MNIST
  python train.py --dataset mnist --arch Net_01_MNIST --epochs 20 --lr 0.001 --log mnist_exp

  # Skip Stage 1 and only finetune FC (requires --pretrained)
  python train.py --dataset cifar10 --arch Net_mid2 --skip_stage1 --pretrained ckpt/mid2_stage1_best.pth --log mid2_s2
"""

from __future__ import absolute_import, division, print_function

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from collections import OrderedDict

from utils import build_cifar, build_mnist, BinOp
from models import (
    Net_00, Net_01, Net_11,
    Net_small, Net_small2, Net_mid,
    Net_mid2, Net_mid3,
    Net_00_MNIST, Net_01_MNIST, Net_11_MNIST,
)


# ============================================================
#  Logger
# ============================================================

class Logger:
    def __init__(self, fpath):
        self.console = sys.stdout
        self.file = open(fpath, 'w')

    def write(self, msg):
        self.console.write(msg)
        self.file.write(msg)
        self.file.flush()

    def flush(self):
        self.console.flush()
        self.file.flush()


# ============================================================
#  Model factory
# ============================================================

MODEL_REGISTRY = {
    'Net_00': Net_00,
    'Net_01': Net_01,
    'Net_11': Net_11,
    'Net_small': Net_small,
    'Net_small2': Net_small2,
    'Net_mid': Net_mid,
    'Net_mid2': Net_mid2,
    'Net_mid3': Net_mid3,
    'Net_00_MNIST': Net_00_MNIST,
    'Net_01_MNIST': Net_01_MNIST,
    'Net_11_MNIST': Net_11_MNIST,
}

TWO_STAGE_ARCHS = {'Net_mid2', 'Net_mid3'}


def build_model(arch, dataset):
    if arch not in MODEL_REGISTRY:
        raise ValueError(f"Unknown architecture: {arch}. "
                         f"Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[arch]()


# ============================================================
#  Training & Testing
# ============================================================

def train_epoch(epoch, model, trainloader, optimizer, criterion, bin_op, device,
                stage_name=None):
    model_ref = model.module if isinstance(model, nn.DataParallel) else model

    if stage_name == 'S2-FC':
        model.eval()
    else:
        model.train()

    for batch_idx, (data, target) in enumerate(trainloader):
        bin_op.binarization()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        bin_op.restore()

        if stage_name != 'S2-FC':
            bin_op.updateBinaryGradWeight()

        optimizer.step()

        if hasattr(model_ref, 'memristor_fc') and hasattr(model_ref.memristor_fc, 'clamp_weights'):
            model_ref.memristor_fc.clamp_weights()

        if batch_idx % 100 == 0:
            tag = f'[{stage_name}] ' if stage_name else ''
            print(f'{tag}Epoch: {epoch} [{batch_idx * len(data)}/{len(trainloader.dataset)} '
                  f'({100. * batch_idx / len(trainloader):.0f}%)]\t'
                  f'Loss: {loss.item():.6f}\tLR: {optimizer.param_groups[0]["lr"]:.6f}')


def test(model, testloader, criterion, bin_op, device):
    model.eval()
    test_loss = 0
    correct = 0

    bin_op.binarization()
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    bin_op.restore()

    test_loss /= len(testloader)
    acc = 100. * correct / len(testloader.dataset)
    return test_loss, acc


def adjust_learning_rate(optimizer, epoch, lr_schedule, base_lr):
    lr = base_lr
    for milestone in lr_schedule:
        if epoch >= milestone:
            lr *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def load_checkpoint(model, ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)
    best_acc = checkpoint.get('acc', checkpoint.get('best_acc', 0.0))
    return best_acc


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='BNN Training')

    # Dataset & Model
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'mnist'],
                        help='Dataset to use')
    parser.add_argument('--arch', type=str, default='Net_01',
                        help='Model architecture')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root directory for datasets')

    # Training hyperparams
    parser.add_argument('--epochs', type=int, default=200,
                        help='Training epochs (Stage 1 for 2-stage models)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=4)

    # 2-stage (1T1R) specific
    parser.add_argument('--skip_stage1', action='store_true',
                        help='Skip Stage 1 (CNN) training')
    parser.add_argument('--epochs_finetune', type=int, default=50,
                        help='Stage 2 (FC) finetuning epochs')
    parser.add_argument('--lr_finetune', type=float, default=0.01,
                        help='Stage 2 learning rate')

    # Misc
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained checkpoint')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log', type=str, default='default_exp',
                        help='Experiment name (used for ckpt and log naming)')
    parser.add_argument('--save_dir', type=str, default='ckpt',
                        help='Directory to save checkpoints')
    parser.add_argument('--cpu', action='store_true',
                        help='Use CPU only')

    args = parser.parse_args()

    # --- Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    log_path = os.path.join(args.save_dir, f'{args.log}.txt')
    sys.stdout = Logger(log_path)

    print(f'==> Options: {args}')
    print(f'==> Device: {device}')

    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)

    # --- Data ---
    print('==> Preparing data...')
    if args.dataset == 'cifar10':
        train_dataset, val_dataset = build_cifar(data_root=args.data_root)
    else:
        train_dataset, val_dataset = build_mnist(data_root=args.data_root)

    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))
    testloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.test_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))

    # --- Model ---
    print(f'==> Building model: {args.arch}')
    model = build_model(args.arch, args.dataset)

    best_acc = 0.0
    if args.pretrained:
        print(f'==> Loading pretrained model from {args.pretrained}')
        best_acc = load_checkpoint(model, args.pretrained, device)
        print(f'    Inherited best acc: {best_acc:.2f}%')
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    model.to(device)
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    bin_op = BinOp(model)
    criterion = nn.CrossEntropyLoss()

    is_two_stage = args.arch in TWO_STAGE_ARCHS

    # ================================================================
    #  Stage 1: Train CNN (or full model for single-stage archs)
    # ================================================================
    if not args.skip_stage1:
        print('\n' + '=' * 50)
        print(f' STAGE 1: Training for {args.epochs} epochs')
        if is_two_stage:
            print(' FC layer is FROZEN (Identity mode)')
        print('=' * 50)

        if is_two_stage:
            model_ref = model.module if isinstance(model, nn.DataParallel) else model
            model_ref.set_stage('train_cnn')

        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, weight_decay=args.weight_decay)
        lr_schedule = [int(args.epochs * 0.6), int(args.epochs * 0.8), int(args.epochs * 0.9)]

        for epoch in range(1, args.epochs + 1):
            adjust_learning_rate(optimizer, epoch, lr_schedule, args.lr)
            stage_tag = 'S1-CNN' if is_two_stage else None
            train_epoch(epoch, model, trainloader, optimizer, criterion, bin_op, device,
                        stage_name=stage_tag)
            loss, acc = test(model, testloader, criterion, bin_op, device)
            print(f'Stage 1 | Epoch {epoch} | Test Acc: {acc:.2f}%')

            if acc > best_acc:
                best_acc = acc
                ckpt_path = os.path.join(args.save_dir, f'{args.log}_stage1_best.pth')
                torch.save({'state_dict': model.state_dict(), 'acc': acc, 'epoch': epoch},
                           ckpt_path)
                print(f'==> Saved best Stage 1 model: {acc:.2f}%')
    else:
        print('\n' + '=' * 50)
        print(' SKIPPING STAGE 1 (using loaded weights)')
        print('=' * 50)

    # ================================================================
    #  Stage 2: Finetune 1T1R FC (only for 2-stage architectures)
    # ================================================================
    if is_two_stage:
        print('\n' + '=' * 50)
        print(f' STAGE 2: Finetuning 1T1R FC for {args.epochs_finetune} epochs')
        print(' CNN layers are FROZEN')
        print('=' * 50)

        model_ref = model.module if isinstance(model, nn.DataParallel) else model
        model_ref.set_stage('train_fc')

        # Sanity check
        baseline_loss, baseline_acc = test(model, testloader, criterion, bin_op, device)
        print(f'==> Baseline accuracy (before Stage 2): {baseline_acc:.2f}%')
        best_acc = baseline_acc

        # Verify frozen state
        trainable_params = []
        for name, param in model.named_parameters():
            if 'memristor_fc' in name:
                param.requires_grad = True
                trainable_params.append(name)
            else:
                param.requires_grad = False
        print(f'  -> Trainable params: {len(trainable_params)} (should be FC only)')

        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr_finetune, weight_decay=0)
        lr_schedule_s2 = [int(args.epochs_finetune * 0.5)]

        best_acc_s2 = 0.0
        for i in range(1, args.epochs_finetune + 1):
            epoch_disp = (args.epochs if not args.skip_stage1 else 0) + i
            adjust_learning_rate(optimizer, i, lr_schedule_s2, args.lr_finetune)
            train_epoch(epoch_disp, model, trainloader, optimizer, criterion, bin_op, device,
                        stage_name='S2-FC')

            if i % 10 == 1 or i == args.epochs_finetune:
                with torch.no_grad():
                    w_prob = torch.sigmoid(model_ref.memristor_fc.weight_latent)
                    print(f'\n[Debug] FC mapping (round):\n{w_prob.round().cpu().numpy()}')

            loss, acc = test(model, testloader, criterion, bin_op, device)
            print(f'Stage 2 | Epoch {epoch_disp} | Test Acc: {acc:.2f}%')

            if acc > best_acc_s2:
                best_acc_s2 = acc
                ckpt_path = os.path.join(args.save_dir, f'{args.log}_stage2_best.pth')
                torch.save({'state_dict': model.state_dict(), 'acc': acc, 'config': str(args)},
                           ckpt_path)
                print(f'==> Saved best Stage 2 model: {acc:.2f}%')

        print(f'\nFinal: Stage 1 best = {best_acc:.2f}%, Stage 2 best = {best_acc_s2:.2f}%')

    print('\n==> Training complete.')


if __name__ == '__main__':
    main()
