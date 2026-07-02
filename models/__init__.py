from .nin import (
    # CIFAR-10 NIN variants
    Net_00, Net_01, Net_11,
    Net_small, Net_small2, Net_mid,
    # 1T1R variants (CIFAR-10)
    Net_mid2, Net_mid3,
    # MNIST NIN variants
    Net_00_MNIST, Net_01_MNIST, Net_11_MNIST,
    # Building blocks
    BinActive, BinConv2d,
    Linear01_Sigmoid, STE_SigmoidRound,
)
