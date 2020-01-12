import torch
from torch import nn
from ..registry import DECODERS


@DECODERS.register_module
class WeightedDecoder(nn.Module):
    def __init__(self, cfg):
        super(WeightedDecoder, self).__init__()

    def forward(self, x):
        pass


class InterpolationDecoder(nn.Module):
    def __init__(self, cfg):
        super(InterpolationDecoder, self).__init__()
