'''
Architectures for segmentation and SFTGAN models
'''
import torch.nn as nn
import torch.nn.functional as F


#############################
# SFTGAN (pytorch version)
#############################


class SFTLayer(nn.Module):
    def __init__(self):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(1024, 1024, 1)
        self.SFT_scale_conv1 = nn.Conv2d(1024, 1024, 1)
        self.SFT_shift_conv0 = nn.Conv2d(1024, 1024, 1)
        self.SFT_shift_conv1 = nn.Conv2d(1024, 1024, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x[1]), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x[1]), 0.1, inplace=True))
        return x[0] * (scale + 1) + shift


class ResBlock_SFT(nn.Module):
    def __init__(self):
        super(ResBlock_SFT, self).__init__()
        self.sft0 = SFTLayer()
        self.conv0 = nn.Conv2d(1024, 1024, 3, 1, 1)
        self.sft1 = SFTLayer()
        self.conv1 = nn.Conv2d(1024, 1024, 3, 1, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        fea = self.sft0(x)
        fea = F.relu(self.conv0(fea), inplace=True)
        fea = self.sft1((fea, x[1]))
        fea = self.conv1(fea)
        return x[0] + fea, x[1]


