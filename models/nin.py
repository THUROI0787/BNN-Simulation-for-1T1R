"""
Binary Neural Network (BNN) models based on Network-in-Network (NIN) backbone.

Supports:
  - CIFAR-10: Net_00, Net_01, Net_11, Net_small, Net_small2, Net_mid
  - CIFAR-10 + 1T1R memristor FC: Net_mid2 (10x10), Net_mid3 (192x10)
  - MNIST: Net_00_MNIST, Net_01_MNIST, Net_11_MNIST
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
#  Activation Binarization (STE)
# ============================================================

class BinActive(torch.autograd.Function):
    """Binarize activations via sign() with Straight-Through Estimator."""

    @staticmethod
    def forward(ctx, input_tensor):
        ctx.save_for_backward(input_tensor)
        return input_tensor.sign()

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input_tensor.ge(1)] = 0
        grad_input[input_tensor.le(-1)] = 0
        return grad_input


# ============================================================
#  Binary Convolution Layer
# ============================================================

class BinConv2d(nn.Module):
    """BN → BinActive → Dropout → Conv2d → ReLU."""

    def __init__(self, input_channels, output_channels,
                 kernel_size, stride=1, padding=0, dropout=0, has_bn=True):
        super(BinConv2d, self).__init__()
        self.has_bn = has_bn
        self.dropout_ratio = dropout

        if self.has_bn:
            self.bn = nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True)
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)

        self.conv = nn.Conv2d(input_channels, output_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.has_bn:
            x = self.bn(x)
        x = BinActive.apply(x)
        if self.dropout_ratio > 0:
            x = self.dropout(x)
        x = self.conv(x)
        x = self.relu(x)
        return x


# ============================================================
#  0/1 Linear Layer for 1T1R Memristor Simulation
# ============================================================

class STE_SigmoidRound(torch.autograd.Function):
    """
    Forward:  round(sigmoid(x))  → {0, 1}
    Backward: sigmoid'(x) * grad_output  (STE variant with soft gradient)
    """

    @staticmethod
    def forward(ctx, input_tensor):
        prob = torch.sigmoid(input_tensor)
        ctx.save_for_backward(prob)
        return prob.round()

    @staticmethod
    def backward(ctx, grad_output):
        prob, = ctx.saved_tensors
        return grad_output * prob * (1 - prob)


class Linear01_Sigmoid(nn.Module):
    """
    A linear layer whose weights are constrained to {0, 1} via sigmoid + rounding.
    Supports two modes:
      - 'identity': acts as identity matrix (for Stage 1 CNN-only training)
      - 'learn':    uses learned 0/1 weights (for Stage 2 FC finetuning)
    """

    def __init__(self, in_features, out_features):
        super(Linear01_Sigmoid, self).__init__()
        self.weight_latent = nn.Parameter(torch.zeros(out_features, in_features))
        self.mode = 'identity'

    def forward(self, x):
        if self.mode == 'identity':
            return x
        w_bin = STE_SigmoidRound.apply(self.weight_latent)
        return F.linear(x, w_bin)


# ============================================================
#  Helper: BN weight clamping (standard BNN practice)
# ============================================================

def _clamp_bn_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if hasattr(m, 'weight') and m.weight is not None:
                m.weight.data.clamp_(min=0.01)


# ============================================================
#  CIFAR-10 NIN Architectures
# ============================================================

class Net_00(nn.Module):
    """First and last layers are full-precision (FP)."""

    def __init__(self, num_classes=10):
        super(Net_00, self).__init__()
        self.xnor_00 = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            BinConv2d(192, 160, kernel_size=1, stride=1, padding=0),
            BinConv2d(160, 96, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            BinConv2d(96, 192, kernel_size=5, stride=1, padding=2, dropout=0.5),
            BinConv2d(192, 192, kernel_size=1, stride=1, padding=0),
            BinConv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),

            BinConv2d(192, 192, kernel_size=3, stride=1, padding=1, dropout=0.5),
            BinConv2d(192, 192, kernel_size=1, stride=1, padding=0),

            nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=True),
            nn.Conv2d(192, num_classes, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=8, stride=1, padding=0),
        )

    def forward(self, x):
        _clamp_bn_weights(self)
        x = self.xnor_00(x)
        return x.view(x.size(0), -1)


class Net_01(nn.Module):
    """First layer FP, last layer binarized."""

    def __init__(self, num_classes=10):
        super(Net_01, self).__init__()
        self.xnor_01 = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            BinConv2d(192, 160, kernel_size=1, stride=1, padding=0),
            BinConv2d(160, 96, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            BinConv2d(96, 192, kernel_size=5, stride=1, padding=2, dropout=0.5),
            BinConv2d(192, 192, kernel_size=1, stride=1, padding=0),
            BinConv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),

            BinConv2d(192, 192, kernel_size=3, stride=1, padding=1, dropout=0.5),
            BinConv2d(192, 192, kernel_size=1, stride=1, padding=0),

            BinConv2d(192, num_classes, kernel_size=1, stride=1, padding=0, dropout=0),
            nn.AvgPool2d(kernel_size=8, stride=1, padding=0),
        )

    def forward(self, x):
        _clamp_bn_weights(self)
        x = self.xnor_01(x)
        return x.view(x.size(0), -1)


class Net_11(nn.Module):
    """First and last layers are binarized."""

    def __init__(self, num_classes=10):
        super(Net_11, self).__init__()
        self.xnor_11 = nn.Sequential(
            BinConv2d(3, 192, kernel_size=5, stride=1, padding=2),

            BinConv2d(192, 160, kernel_size=1, stride=1, padding=0),
            BinConv2d(160, 96, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            BinConv2d(96, 192, kernel_size=5, stride=1, padding=2, dropout=0.5),
            BinConv2d(192, 192, kernel_size=1, stride=1, padding=0),
            BinConv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),

            BinConv2d(192, 192, kernel_size=3, stride=1, padding=1, dropout=0.5),
            BinConv2d(192, 192, kernel_size=1, stride=1, padding=0),

            BinConv2d(192, num_classes, kernel_size=1, stride=1, padding=0, dropout=0),
            nn.AvgPool2d(kernel_size=8, stride=1, padding=0),
        )

    def forward(self, x):
        _clamp_bn_weights(self)
        x = self.xnor_11(x)
        return x.view(x.size(0), -1)


class Net_small(nn.Module):
    """Smaller VGG-style BNN for CIFAR-10 (~0.5M params)."""

    def __init__(self, num_classes=10):
        super(Net_small, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            BinConv2d(128, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            BinConv2d(128, 256, kernel_size=3, padding=1),
            BinConv2d(256, 256, kernel_size=3, padding=1, dropout=0.5),
            nn.MaxPool2d(kernel_size=2, stride=2),

            BinConv2d(256, 512, kernel_size=3, padding=1),
            BinConv2d(512, 512, kernel_size=3, padding=1, dropout=0.5),

            BinConv2d(512, num_classes, kernel_size=1, padding=0, has_bn=False),
            nn.AvgPool2d(kernel_size=8),
        )

    def forward(self, x):
        _clamp_bn_weights(self)
        x = self.features(x)
        return x.view(x.size(0), -1)


class Net_small2(nn.Module):
    """Ultra-lightweight BNN for CIFAR-10 (~0.15M params)."""

    def __init__(self, num_classes=10):
        super(Net_small2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            BinConv2d(64, 64, kernel_size=3, padding=1, dropout=0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),

            BinConv2d(64, 128, kernel_size=3, padding=1),
            BinConv2d(128, 128, kernel_size=3, padding=1, dropout=0.5),
            nn.MaxPool2d(kernel_size=2, stride=2),

            BinConv2d(128, num_classes, kernel_size=1, padding=0),
            nn.AvgPool2d(kernel_size=8),
        )

    def forward(self, x):
        _clamp_bn_weights(self)
        x = self.features(x)
        return x.view(x.size(0), -1)


class Net_mid(nn.Module):
    """Medium BNN for CIFAR-10 (~0.33M params)."""

    def __init__(self, num_classes=10):
        super(Net_mid, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(96, eps=1e-4, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            BinConv2d(96, 96, kernel_size=3, padding=1, dropout=0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),

            BinConv2d(96, 192, kernel_size=3, padding=1),
            BinConv2d(192, 192, kernel_size=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),

            BinConv2d(192, 192, kernel_size=3, padding=1, dropout=0.5),
            BinConv2d(192, 192, kernel_size=1, padding=0),

            BinConv2d(192, num_classes, kernel_size=1, padding=0),
            nn.AvgPool2d(kernel_size=8),
        )

    def forward(self, x):
        _clamp_bn_weights(self)
        x = self.features(x)
        return x.view(x.size(0), -1)


# ============================================================
#  1T1R Memristor FC Architectures (CIFAR-10)
# ============================================================

class Net_mid2(nn.Module):
    """
    NIN backbone + 10×10 1T1R memristor FC layer.
    Two-stage training:
      Stage 1 (train_cnn): FC = Identity, train CNN only
      Stage 2 (train_fc):   CNN frozen, train 10×10 0/1 FC
    """

    def __init__(self, num_classes=10):
        super(Net_mid2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(96, eps=1e-4, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            BinConv2d(96, 96, kernel_size=3, padding=1, dropout=0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),

            BinConv2d(96, 192, kernel_size=3, padding=1),
            BinConv2d(192, 192, kernel_size=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),

            BinConv2d(192, 192, kernel_size=3, padding=1, dropout=0.5),
            BinConv2d(192, 192, kernel_size=1, padding=0),

            BinConv2d(192, num_classes, kernel_size=1, padding=0),
            nn.AvgPool2d(kernel_size=8),
        )
        self.memristor_fc = Linear01_Sigmoid(num_classes, num_classes)

    def forward(self, x):
        _clamp_bn_weights(self)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.memristor_fc(x)
        return x

    def set_stage(self, stage):
        if stage == 'train_cnn':
            self.memristor_fc.mode = 'identity'
            for p in self.memristor_fc.parameters():
                p.requires_grad = False
            for p in self.features.parameters():
                p.requires_grad = True
        elif stage == 'train_fc':
            self.memristor_fc.mode = 'learn'
            nn.init.zeros_(self.memristor_fc.weight_latent)
            for p in self.memristor_fc.parameters():
                p.requires_grad = True
            for p in self.features.parameters():
                p.requires_grad = False


class Net_mid3(nn.Module):
    """
    NIN backbone (outputs 192-dim features) + 192×10 1T1R memristor FC.
    The FC layer acts as the sole classifier.
    """

    def __init__(self, num_classes=10):
        super(Net_mid3, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(96, eps=1e-4, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            BinConv2d(96, 96, kernel_size=3, padding=1, dropout=0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),

            BinConv2d(96, 192, kernel_size=3, padding=1),
            BinConv2d(192, 192, kernel_size=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),

            BinConv2d(192, 192, kernel_size=3, padding=1, dropout=0.5),
            BinConv2d(192, 192, kernel_size=1, padding=0),

            nn.AvgPool2d(kernel_size=8),
        )
        self.memristor_fc = Linear01_Sigmoid(192, num_classes)

    def forward(self, x):
        _clamp_bn_weights(self)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.memristor_fc(x)
        return x

    def set_stage(self, stage):
        if stage == 'train_cnn':
            self.memristor_fc.mode = 'learn'
            for p in self.parameters():
                p.requires_grad = True
        elif stage == 'train_fc':
            self.memristor_fc.mode = 'learn'
            for p in self.features.parameters():
                p.requires_grad = False
            for p in self.memristor_fc.parameters():
                p.requires_grad = True


# ============================================================
#  MNIST NIN Architectures (1 input channel, 28×28)
# ============================================================

class Net_00_MNIST(nn.Module):
    """MNIST: First and last layers FP."""

    def __init__(self, num_classes=10, input_channels=1):
        super(Net_00_MNIST, self).__init__()
        self.xnor_00 = nn.Sequential(
            nn.Conv2d(input_channels, 192, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            BinConv2d(192, 160, kernel_size=1, stride=1, padding=0),
            BinConv2d(160, 96, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            BinConv2d(96, 192, kernel_size=5, stride=1, padding=2, dropout=0.4),
            BinConv2d(192, 192, kernel_size=1, stride=1, padding=0),
            BinConv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),

            BinConv2d(192, 192, kernel_size=3, stride=1, padding=1, dropout=0.4),
            BinConv2d(192, 192, kernel_size=1, stride=1, padding=0),

            nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=True),
            nn.Conv2d(192, num_classes, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=7, stride=1, padding=0),
        )

    def forward(self, x):
        _clamp_bn_weights(self)
        x = self.xnor_00(x)
        return x.view(x.size(0), -1)


class Net_01_MNIST(nn.Module):
    """MNIST: First layer FP, last layer binarized."""

    def __init__(self, num_classes=10, input_channels=1):
        super(Net_01_MNIST, self).__init__()
        self.xnor_01 = nn.Sequential(
            nn.Conv2d(input_channels, 192, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            BinConv2d(192, 160, kernel_size=1, stride=1, padding=0),
            BinConv2d(160, 96, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            BinConv2d(96, 192, kernel_size=5, stride=1, padding=2, dropout=0.4),
            BinConv2d(192, 192, kernel_size=1, stride=1, padding=0),
            BinConv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),

            BinConv2d(192, 192, kernel_size=3, stride=1, padding=1, dropout=0.4),
            BinConv2d(192, 192, kernel_size=1, stride=1, padding=0),

            BinConv2d(192, num_classes, kernel_size=1, stride=1, padding=0, dropout=0),
            nn.AvgPool2d(kernel_size=7, stride=1, padding=0),
        )

    def forward(self, x):
        _clamp_bn_weights(self)
        x = self.xnor_01(x)
        return x.view(x.size(0), -1)


class Net_11_MNIST(nn.Module):
    """MNIST: First and last layers binarized."""

    def __init__(self, num_classes=10, input_channels=1):
        super(Net_11_MNIST, self).__init__()
        self.xnor_11 = nn.Sequential(
            BinConv2d(input_channels, 192, kernel_size=5, stride=1, padding=2, first_layer=True),

            BinConv2d(192, 160, kernel_size=1, stride=1, padding=0),
            BinConv2d(160, 96, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            BinConv2d(96, 192, kernel_size=5, stride=1, padding=2, dropout=0.4),
            BinConv2d(192, 192, kernel_size=1, stride=1, padding=0),
            BinConv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),

            BinConv2d(192, 192, kernel_size=3, stride=1, padding=1, dropout=0.4),
            BinConv2d(192, 192, kernel_size=1, stride=1, padding=0),

            BinConv2d(192, num_classes, kernel_size=1, stride=1, padding=0, dropout=0),
            nn.AvgPool2d(kernel_size=7, stride=1, padding=0),
        )

    def forward(self, x):
        _clamp_bn_weights(self)
        x = self.xnor_11(x)
        return x.view(x.size(0), -1)
