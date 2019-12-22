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
        self.SFT_scale_conv1 = nn.Linear(1024, 2048)
        self.SFT_shift_conv0 = nn.Linear(300, 1024)
        self.SFT_shift_conv1 = nn.Linear(1024, 2048)

    def forward(self, feat, objvec):
        # x[0]: fea; x[1]: cond
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(objvec), 0.1, inplace=True))
        scale = scale.view((-1, 1, 1, 2048))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(objvec), 0.1, inplace=True))
        shift = shift.view((-1, 1, 1, 2048))
        return (feat * scale + shift) + feat



