import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, mode="train"):
        super(DiceLoss, self).__init__()
        self.mode = mode

    def forward(self, inputs, targets, smooth=1):
        
        inputs = (inputs > 0.5) * 1.

        inputs = inputs.view(-1)
        targets = targets.view(-1)
                
        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


def Dice(inputs, targets, smooth=1):

    inputs = inputs.reshape(-1)
    inputs = inputs / 255
    targets= targets.reshape(-1)
    targets= targets/ 255
    intersection = (inputs * targets).sum()
    dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

    return dice
