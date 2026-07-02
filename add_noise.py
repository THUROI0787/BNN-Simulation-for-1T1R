"""
Generate noisy checkpoints for 1T1R robustness experiments.

Applies bit-flip, stuck-at-0 (drop), or stuck-at-1 noise to binarized
convolutional weights at specified layers and saves noisy checkpoints.

Example:
  python add_noise.py --base_ckpt ckpt/mid2_exp1_stage1_best.pth --save_dir ckpt/noisy_models
"""

from __future__ import absolute_import, division, print_function

import argparse
import os
import copy
import torch
import torch.nn as nn
from collections import OrderedDict

from utils import build_cifar, BinOp
from models import Net_mid2


def test(model, testloader, device, bin_op):
    model.eval()
    correct = 0
    total = 0
    bin_op.binarization()
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)
    bin_op.restore()
    return 100. * correct / total


def apply_noise(model, noise_type, probability, target_layers='all'):
    print(f'Applying Noise -> Type: {noise_type}, Prob: {probability}, Target: {target_layers}')
    conv_count = 0
    affected = 0

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            conv_count += 1
            if conv_count == 1:
                continue  # skip first FP conv

            is_target = False
            if target_layers == 'all':
                is_target = True
            elif target_layers == 'shallow' and conv_count in [2, 3]:
                is_target = True
            elif target_layers == 'mid' and conv_count in [4, 5]:
                is_target = True
            elif target_layers == 'deep' and conv_count in [6, 7]:
                is_target = True

            if not is_target:
                continue

            affected += 1
            weight = m.weight.data
            mask = torch.rand_like(weight) < probability

            if noise_type == 'flip':
                weight[mask] = weight[mask] * -1
            elif noise_type == 'drop':
                weight[mask] = 0
            elif noise_type == 'stuck':
                weight[mask] = weight[mask].abs() + 0.1

    print(f'  -> Noise injected into {affected} layers.')
    return model


def main():
    parser = argparse.ArgumentParser(description='Generate Noisy Checkpoints')
    parser.add_argument('--base_ckpt', type=str, required=True,
                        help='Path to clean checkpoint')
    parser.add_argument('--save_dir', type=str, default='ckpt/noisy_models',
                        help='Directory to save noisy checkpoints')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Data
    print('Loading Dataset...')
    _, val_dataset = build_cifar()
    testloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers)

    # Model
    print('Building Model...')
    base_model = Net_mid2()

    print(f'Loading Base Checkpoint: {args.base_ckpt}')
    checkpoint = torch.load(args.base_ckpt, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    base_model.load_state_dict(new_state_dict)
    base_model.to(device)

    bin_op = BinOp(base_model)
    base_acc = test(base_model, testloader, device, bin_op)
    print(f'Baseline Accuracy: {base_acc:.2f}%')

    configs = [
        ('mid_flip_10', 'flip', 0.10, 'mid'),
    ]

    print('\n' + '=' * 50)
    print(' Starting Noise Generation Loop ')
    print('=' * 50)

    for name, n_type, prob, target in configs:
        noisy_model = copy.deepcopy(base_model)
        noisy_model = apply_noise(noisy_model, n_type, prob, target)
        current_bin_op = BinOp(noisy_model)
        acc = test(noisy_model, testloader, device, current_bin_op)
        print(f'[{name}] Acc: {acc:.2f}% (Delta: {acc - base_acc:.2f}%)')

        save_name = f'mid2_noise_{name}_acc{acc:.1f}.pth'
        save_path = os.path.join(args.save_dir, save_name)
        torch.save({
            'state_dict': noisy_model.state_dict(),
            'noise_config': {'type': n_type, 'prob': prob, 'target': target},
            'acc': acc, 'base_acc': base_acc,
        }, save_path)
        print(f'Saved to {save_path}\n' + '-' * 30)

    print('Done! All noisy checkpoints generated.')


if __name__ == '__main__':
    main()
