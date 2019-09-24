import torch
import torch.nn as nn
import torch.nn.functional as F


class SpaLan(nn.Module):

    def __str__(self):
        return 'SpaLan'

    def __init__(self, in_feat_dim, class_num):
        super(SpaLan, self).__init__()
        self.hidden_layer = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_feat_dim, 1024))

        self.classifier = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, class_num))

        self.proposal = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 2))

    def forward(self, input, hoi_cates=None, bin_cates=None, pos_mask=None):

        hidden = self.hidden_layer(input)
        bin_scores = self.proposal(hidden)
        hoi_scores = self.classifier(hidden)

        bin_prob = F.softmax(bin_scores, dim=1)
        hoi_prob = F.sigmoid(hoi_scores)

        bin_pred = torch.argmax(bin_prob, dim=1)
        hoi_pred = (hoi_prob > 0.5).float()

        bin_error = -1
        hoi_error = -1

        loss_cls = -1
        loss_bin = -1

        if self.training:
            bin_error = torch.abs(bin_pred - bin_cates).sum()
            hoi_error = torch.abs(hoi_pred[pos_mask] - hoi_cates[pos_mask]).sum()
            loss_bin = F.cross_entropy(bin_scores, bin_cates, size_average=False)
            loss_cls = F.binary_cross_entropy(hoi_prob[pos_mask], hoi_cates[pos_mask], size_average=False)    # multi-label classification

        return bin_prob, hoi_prob, loss_bin, loss_cls, bin_error, hoi_error




