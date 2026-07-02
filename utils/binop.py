"""
Binary Operation (BinOp) for BNN training.

Handles weight binarization, restoration, and custom gradient updates
following the XNOR-Net style training algorithm.
"""

import torch
import torch.nn as nn
import numpy as np


class BinOp:
    def __init__(self, model):
        count_Conv2d = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                count_Conv2d += 1

        start_range = 1
        if count_Conv2d <= 2:
            self.bin_range = []
        else:
            end_idx = count_Conv2d - 2
            self.bin_range = np.linspace(
                start_range, end_idx,
                end_idx - start_range + 1
            ).astype('int').tolist() if end_idx >= start_range else []

        self.num_of_params = len(self.bin_range)
        self.saved_params = []
        self.target_modules = []

        current_conv_idx = -1
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                current_conv_idx += 1
                if current_conv_idx in self.bin_range:
                    self.saved_params.append(m.weight.data.clone())
                    self.target_modules.append(m.weight)

    def binarization(self):
        self.meancenterConvParams()
        self.clampConvParams()
        self.save_params()
        self.binarizeConvParams()

    def meancenterConvParams(self):
        for i in range(self.num_of_params):
            w = self.target_modules[i].data
            mean_val = w.mean(dim=1, keepdim=True)
            w.add_(mean_val.mul(-1).expand_as(w))

    def clampConvParams(self):
        for i in range(self.num_of_params):
            self.target_modules[i].data.clamp_(-1.0, 1.0)

    def save_params(self):
        for i in range(self.num_of_params):
            self.saved_params[i].copy_(self.target_modules[i].data)

    def binarizeConvParams(self):
        for i in range(self.num_of_params):
            w = self.target_modules[i].data
            s = w.size()
            n = s[1] * s[2] * s[3] if len(s) == 4 else w[0].nelement()
            if n == 0:
                n = 1
            m = w.abs().sum(dim=(1, 2, 3), keepdim=True).div_(n)
            self.target_modules[i].data.copy_(w.sign().mul_(m.expand(s)))

    def restore(self):
        for i in range(self.num_of_params):
            self.target_modules[i].data.copy_(self.saved_params[i])

    def updateBinaryGradWeight(self):
        for i in range(self.num_of_params):
            w_param = self.target_modules[i]
            w_data = w_param.data
            if w_param.grad is None:
                continue
            grad_data = w_param.grad.data
            s = w_data.size()
            n = s[1] * s[2] * s[3] if len(s) == 4 else w_data[0].nelement()
            if n == 0:
                n = 1

            m_scale = w_data.abs().sum(dim=(1, 2, 3), keepdim=True).div_(n).expand(s).clone()
            m_scale[w_data.lt(-1.0)] = 0
            m_scale[w_data.gt(1.0)] = 0
            term1 = grad_data.mul(m_scale)

            term2_sum = grad_data.mul(w_data.sign()).sum(dim=(1, 2, 3), keepdim=True).div_(n)
            term2 = term2_sum.expand(s).mul(w_data.sign())

            final_scale = (1.0 - 1.0 / s[1]) * n if s[1] > 1 else n
            new_grad = term1.add(term2).mul_(final_scale)
            w_param.grad.data.copy_(new_grad)
