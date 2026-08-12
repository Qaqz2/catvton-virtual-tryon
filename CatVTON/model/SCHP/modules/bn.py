"""
Simplified bn.py — uses standard PyTorch BatchNorm2d instead of
the compiled InPlaceABN/CUDA extension.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .functions import ACT_RELU, ACT_LEAKY_RELU, ACT_ELU, ACT_NONE


class ABN(nn.BatchNorm2d):
    """Activated Batch Normalization using standard PyTorch BatchNorm2d.

    直接继承 nn.BatchNorm2d，这样 state_dict 的 key 与原始权重
    （bn1.weight / bn1.running_mean 等）保持一致，避免 strict=False
    加载时把所有 BN 参数跳过。
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 activation="leaky_relu", slope=0.01):
        super().__init__(num_features, eps=eps, momentum=momentum, affine=affine)
        self.activation = activation
        self.slope = slope

    def forward(self, x):
        x = super().forward(x)
        if self.activation == ACT_RELU:
            return F.relu(x, inplace=True)
        elif self.activation == ACT_LEAKY_RELU:
            return F.leaky_relu(x, negative_slope=self.slope, inplace=True)
        elif self.activation == ACT_ELU:
            return F.elu(x, inplace=True)
        else:
            return x


class InPlaceABN(ABN):
    """InPlace variant — identical to ABN with standard BN."""
    pass


class InPlaceABNSync(ABN):
    """Sync variant — uses regular BN for single-GPU inference."""
    pass
