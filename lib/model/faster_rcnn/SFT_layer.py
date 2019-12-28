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
        self.SFT_scale_conv0 = nn.Linear(300, 1024)
        self.SFT_scale_conv1 = nn.Linear(1024, 1024)
        self.SFT_shift_conv0 = nn.Linear(300, 1024)
        self.SFT_shift_conv1 = nn.Linear(1024, 1024)

    def forward(self, feat, objvec):
        # x[0]: fea; x[1]: cond
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(objvec), 0.1, inplace=True))
        scale = scale.view((-1, 1024, 1, 1))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(objvec), 0.1, inplace=True))
        shift = shift.view((-1, 1024, 1, 1))
        return (feat * scale + shift) + feat


class ResBlock_SFT(nn.Module):
    def __init__(self, channel=1024):
        super(ResBlock_SFT, self).__init__()
        self.sft0 = SFTLayer()
        self.conv0 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.sft1 = SFTLayer()
        self.conv1 = nn.Conv2d(channel, channel, 3, 1, 1)

    def forward(self, feat, cond):
        # x[0]: fea; x[1]: cond
        feat_sft = self.sft0(feat, cond)
        feat_sft = F.relu(self.conv0(feat_sft), inplace=True)
        feat_sft = self.sft1(feat_sft, cond)
        feat_sft = self.conv1(feat_sft)
        return feat + feat_sft