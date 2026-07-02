"""
Unified testing/evaluation script for Binary Neural Networks.

Supports multiple evaluation modes:
  - standard:    basic accuracy evaluation
  - noise:       add Gaussian noise to binarized weights, sweep std values
  - quant:       quantize first/last Conv layers to N-bit
  - sparsity:    measure sparsity (+1 ratio) of binarized weights
  - fc:          evaluate 1T1R FC models (Net_mid2/Net_mid3), with optional identity mode
  - confusion:   generate confusion matrix (CSV)

Examples:
  # Standard evaluation
  python test.py --dataset cifar10 --arch Net_01 --ckpt ckpt/my_model.pth

  # Noise robustness sweep
  python test.py --dataset cifar10 --arch Net_01 --ckpt ckpt/my_model.pth --mode noise --noise_std_max 0.2 --noise_steps 25

  # Quantization evaluation
  python test.py --dataset cifar10 --arch Net_01 --ckpt ckpt/my_model.pth --mode quant --n_bits 4

  # Sparsity measurement
  python test.py --dataset cifar10 --arch Net_01 --ckpt ckpt/my_model.pth --mode sparsity

  # 1T1R FC evaluation (with identity baseline)
  python test.py --dataset cifar10 --arch Net_mid2 --ckpt ckpt/mid2_stage2_best.pth --mode fc --force_identity

  # Confusion matrix
  python test.py --dataset cifar10 --arch Net_mid2 --ckpt ckpt/mid2_stage2_best.pth --mode confusion
"""

from __future__ import absolute_import, division, print_function

import argparse
import os
import csv
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

from utils import build_cifar, build_mnist, BinOp
from models import (
    Net_00, Net_01, Net_11,
    Net_small, Net_small2, Net_mid,
    Net_mid2, Net_mid3,
    Net_00_MNIST, Net_01_MNIST, Net_11_MNIST,
    BinConv2d,
)


MODEL_REGISTRY = {
    'Net_00': Net_00, 'Net_01': Net_01, 'Net_11': Net_11,
    'Net_small': Net_small, 'Net_small2': Net_small2, 'Net_mid': Net_mid,
    'Net_mid2': Net_mid2, 'Net_mid3': Net_mid3,
    'Net_00_MNIST': Net_00_MNIST, 'Net_01_MNIST': Net_01_MNIST,
    'Net_11_MNIST': Net_11_MNIST,
}


# ============================================================
#  Helpers
# ============================================================

def load_model(arch, ckpt_path, device):
    if arch not in MODEL_REGISTRY:
        raise ValueError(f"Unknown architecture: {arch}. "
                         f"Available: {list(MODEL_REGISTRY.keys())}")
    model = MODEL_REGISTRY[arch]()
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)

    info = {}
    if 'acc' in checkpoint:
        info['ckpt_acc'] = checkpoint['acc']
    if 'best_acc' in checkpoint:
        info['ckpt_best_acc'] = checkpoint['best_acc']
    if 'config' in checkpoint:
        info['config'] = checkpoint['config']
    return model, info


def evaluate(model, testloader, criterion, bin_op, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    if bin_op:
        bin_op.binarization()

    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)

    if bin_op:
        bin_op.restore()

    acc = 100. * correct / total
    avg_loss = test_loss / total
    return acc, avg_loss


def evaluate_with_confusion(model, testloader, criterion, bin_op, device, num_classes=10):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)

    if bin_op:
        bin_op.binarization()

    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)
            for t, p in zip(target.view(-1), pred.view(-1)):
                cm[t.long(), p.long()] += 1

    if bin_op:
        bin_op.restore()

    acc = 100. * correct / total
    avg_loss = test_loss / total
    return acc, avg_loss, cm.cpu().numpy()


def add_gaussian_noise_to_binconv(model, std_dev):
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, BinConv2d):
                m.conv.weight.data.add_(torch.randn_like(m.conv.weight.data) * std_dev)


def quantize_tensor_asymmetric(tensor_data, num_bits=4, clip_percentile=None):
    if tensor_data is None:
        return None
    original_dtype = tensor_data.dtype
    t = tensor_data.float()

    if clip_percentile is not None and 0 < clip_percentile < 50:
        flat = t.flatten()
        lo = torch.quantile(flat, clip_percentile / 100.0)
        hi = torch.quantile(flat, 1.0 - clip_percentile / 100.0)
    else:
        lo, hi = t.min(), t.max()

    if lo.item() == hi.item():
        return torch.full_like(tensor_data, lo.item(), dtype=original_dtype)

    num_levels = 2 ** num_bits
    scale = (hi - lo) / (num_levels - 1)
    if scale.item() < 1e-9:
        return torch.full_like(tensor_data, ((lo + hi) / 2).item(), dtype=original_dtype)

    q = torch.round((t - lo) / scale)
    q = torch.clamp(q, 0, num_levels - 1)
    return (lo + q * scale).to(original_dtype)


def quantize_first_conv(model, num_bits, clip_percentile):
    """Quantize the first Conv2d layer (index 0 in features/xnor)."""
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                if m.weight is not None:
                    m.weight.data.copy_(
                        quantize_tensor_asymmetric(m.weight.data, num_bits, clip_percentile))
                break  # only first conv


def print_sparsity(model):
    print('\n==> Sparsity of BinConv2d layers (+1 ratio):')
    with torch.no_grad():
        for name, m in model.named_modules():
            if isinstance(m, BinConv2d):
                w = m.conv.weight.data
                total = w.nelement()
                pos = (w > 0).sum().item()
                sp = 100. * pos / total if total > 0 else 0
                print(f'  {name}: {sp:.2f}% ({pos}/{total})')


def print_fc_mapping(model):
    if hasattr(model, 'memristor_fc'):
        fc = model.memristor_fc
        w_bin = torch.sigmoid(fc.weight_latent.data).round().cpu().numpy()
        print('\n=== 1T1R FC Mapping (0/1) ===')
        print(w_bin)


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='BNN Evaluation')

    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'mnist'])
    parser.add_argument('--arch', type=str, required=True, help='Model architecture')
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--data_root', type=str, default='./data')

    parser.add_argument('--mode', type=str, default='standard',
                        choices=['standard', 'noise', 'quant', 'sparsity', 'fc', 'confusion'],
                        help='Evaluation mode')

    # Noise mode
    parser.add_argument('--noise_std_max', type=float, default=0.2,
                        help='Max noise std for sweep')
    parser.add_argument('--noise_steps', type=int, default=25,
                        help='Number of noise steps')

    # Quant mode
    parser.add_argument('--n_bits', type=int, default=4, help='Quantization bits')
    parser.add_argument('--clip_percentile', type=float, default=None,
                        help='Percentile for clipping (e.g. 0.25)')

    # FC mode
    parser.add_argument('--force_identity', action='store_true',
                        help='Force FC to identity mode (baseline)')

    # Common
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')

    args = parser.parse_args()
    print('==> Options:', args)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f'==> Device: {device}')

    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)

    # --- Data ---
    print('==> Preparing data...')
    if args.dataset == 'cifar10':
        _, val_dataset = build_cifar(data_root=args.data_root)
    else:
        _, val_dataset = build_mnist(data_root=args.data_root)

    testloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))

    # --- Model ---
    print(f'==> Loading model: {args.arch} from {args.ckpt}')
    model, info = load_model(args.arch, args.ckpt, device)
    if info:
        for k, v in info.items():
            print(f'    {k}: {v}')

    # FC mode: optionally force identity
    if args.mode == 'fc' and args.force_identity:
        if hasattr(model, 'memristor_fc'):
            model.memristor_fc.mode = 'identity'
            print('\n[WARNING] Force Identity mode: 1T1R weights IGNORED.')
        else:
            print('\n[WARNING] Model has no memristor_fc layer.')

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    bin_op = BinOp(model) if any(isinstance(m, BinConv2d) for m in model.modules()) else None

    # ================================================================
    #  Mode: standard
    # ================================================================
    if args.mode == 'standard':
        acc, loss = evaluate(model, testloader, criterion, bin_op, device)
        print(f'\nTest Results: Acc = {acc:.2f}%, Loss = {loss:.4f}')

    # ================================================================
    #  Mode: noise
    # ================================================================
    elif args.mode == 'noise':
        acc0, loss0 = evaluate(model, testloader, criterion, bin_op, device)
        print(f'\nBaseline (no noise): Acc = {acc0:.2f}%')

        std_values = np.linspace(0.001, args.noise_std_max, args.noise_steps)
        accs, losses = [], []

        for std in std_values:
            # Reload model each time to avoid accumulated noise
            model_i, _ = load_model(args.arch, args.ckpt, device)
            model_i.to(device)
            bin_op_i = BinOp(model_i)
            bin_op_i.binarization()
            add_gaussian_noise_to_binconv(model_i, std)
            acc, loss = evaluate(model_i, testloader, criterion, None, device)
            # Note: evaluate will binarize again, but noise is already on the weights
            accs.append(acc)
            losses.append(loss)
            bin_op_i.restore()

        print('\n=== Noise Sweep Results ===')
        print('std:', ','.join([f'{s:.4f}' for s in std_values]))
        print('acc:', ','.join([f'{a:.2f}' for a in accs]))
        print(f'Baseline acc: {acc0:.2f}')

    # ================================================================
    #  Mode: quant
    # ================================================================
    elif args.mode == 'quant':
        acc0, loss0 = evaluate(model, testloader, criterion, bin_op, device)
        print(f'\nBaseline (no quant): Acc = {acc0:.2f}%')

        quantize_first_conv(model, args.n_bits, args.clip_percentile)
        acc_q, loss_q = evaluate(model, testloader, criterion, bin_op, device)
        print(f'Quantized ({args.n_bits}-bit): Acc = {acc_q:.2f}%')
        print(f'Delta: {acc_q - acc0:.2f}%')

    # ================================================================
    #  Mode: sparsity
    # ================================================================
    elif args.mode == 'sparsity':
        if bin_op:
            bin_op.binarization()
        print_sparsity(model)
        if bin_op:
            bin_op.restore()
        acc, loss = evaluate(model, testloader, criterion, bin_op, device)
        print(f'\nTest Acc = {acc:.2f}%')

    # ================================================================
    #  Mode: fc
    # ================================================================
    elif args.mode == 'fc':
        print_fc_mapping(model)
        acc, loss = evaluate(model, testloader, criterion, bin_op, device)
        print(f'\nTest Acc = {acc:.2f}%')

    # ================================================================
    #  Mode: confusion
    # ================================================================
    elif args.mode == 'confusion':
        print_fc_mapping(model)
        acc, loss, cm = evaluate_with_confusion(model, testloader, criterion, bin_op, device)

        # Normalize
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm_prob = cm / row_sums

        print(f'\nTest Acc = {acc:.2f}%')
        print('\nConfusion Matrix (Row=Actual, Col=Predicted):')
        print('      ' + '  '.join([f'P{i}' for i in range(10)]))
        for i in range(10):
            row_str = '  '.join([f'{v:.2f}' for v in cm_prob[i]])
            print(f'Act{i}: {row_str}')

        ckpt_name = os.path.basename(args.ckpt).replace('.pth', '')
        csv_path = f'confusion_matrix_{ckpt_name}.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Actual'] + [f'Pred_{i}' for i in range(10)])
            for i in range(10):
                writer.writerow([f'Class_{i}'] + [f'{v:.4f}' for v in cm_prob[i]])
        print(f'\nConfusion matrix saved to: {csv_path}')

    print('\n==> Evaluation finished.')


if __name__ == '__main__':
    main()
