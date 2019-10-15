import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpaConv(nn.Module):
    def __init__(self):
        super(SpaConv, self).__init__()
        # (batch,64,64,2)->(batch,60,60,64)
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=5)
        # (batch,60,60,64)->(batch,30,30,64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # (batch,30,30,64)->(batch,26,26,32)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5)
        # (batch,26,26,32)->(batch,13,13,32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, spa_map):
        conv1 = self.conv1(spa_map)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        return pool2.view(spa_map.shape[0], -1)


class SpaLan(nn.Module):

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def __str__(self):
        return 'Spa+Obj+Pose'

    def __init__(self, spa_feat_dim, num_hoi_class, num_obj_class, num_key_point):
        super(SpaLan, self).__init__()

        self.spa_conv = SpaConv()

        self.spa_hidden_layer = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(spa_feat_dim, 1024))

        self.spa_classifier = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_hoi_class))

        # self.spa_proposal = nn.Sequential(
        #     nn.LeakyReLU(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(1024, 2))

        self.obj_hidden_layer = nn.Sequential(
            nn.Linear(num_obj_class, 1024))

        self.obj_classifier = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_hoi_class))

        # self.obj_proposal = nn.Sequential(
        #     nn.LeakyReLU(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(1024, 2))

        self.pose_hidden_layer = nn.Sequential(
            nn.Linear(num_key_point, 1024))

        self.pose_classifier = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_hoi_class))

        # self.pose_proposal = nn.Sequential(
        #     nn.LeakyReLU(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(1024, 2))

        # self._initialize_weights()

    def forward(self, spa_map, obj_vec, pose_feat, hoi_cates=None, bin_cates=None, pos_mask=None):
        num_ins = spa_map.shape[0]

        spa_vec = self.spa_conv(spa_map)
        spa_hidden = self.spa_hidden_layer(spa_vec)
        spa_hoi_scores = self.spa_classifier(spa_hidden)
        spa_hoi_prob = F.sigmoid(spa_hoi_scores)
        # spa_bin_scores = self.spa_proposal(spa_hidden)

        obj_hidden = self.obj_hidden_layer(obj_vec)
        obj_hoi_scores = self.spa_classifier(obj_hidden)
        obj_hoi_prob = F.sigmoid(obj_hoi_scores)
        # obj_bin_scores = self.obj_proposal(obj_hidden)

        pose_hidden = self.pose_hidden_layer(pose_feat)
        pose_hoi_scores = self.pose_classifier(pose_hidden)
        pose_hoi_prob = F.sigmoid(pose_hoi_scores)
        # pose_bin_scores = self.pose_proposal(pose_hidden)

        # bin_scores = (spa_bin_scores + obj_bin_scores + pose_bin_scores) / 3
        # hoi_scores = (spa_hoi_scores + obj_hoi_scores + pose_hoi_scores) / 3

        # bin_prob = F.softmax(bin_scores, dim=1)
        bin_prob = torch.zeros(num_ins)
        hoi_prob = spa_hoi_prob * obj_hoi_prob * pose_hoi_prob

        # bin_pred = torch.argmax(bin_prob, dim=1)
        hoi_pred = (hoi_prob > 0.5).float()

        bin_error = torch.zeros(num_ins)
        hoi_error = torch.zeros(num_ins)

        loss_cls = torch.zeros(num_ins)
        loss_bin = torch.ones(num_ins)

        if hoi_cates is not None and bin_cates is not None:
            # bin_error = torch.abs(bin_pred - bin_cates).sum().float() / num_ins
            hoi_error = torch.abs(hoi_pred[pos_mask] - hoi_cates[pos_mask]).sum() * 1.0 / pos_mask.sum().item()
            # loss_bin = F.cross_entropy(bin_scores, bin_cates, size_average=False)
            spa_loss_cls = F.binary_cross_entropy(spa_hoi_prob[pos_mask], hoi_cates[pos_mask], size_average=False)    # multi-label classification
            obj_loss_cls = F.binary_cross_entropy(obj_hoi_prob[pos_mask], hoi_cates[pos_mask], size_average=False)
            pos_loss_cls = F.binary_cross_entropy(pose_hoi_prob[pos_mask], hoi_cates[pos_mask], size_average=False)
            loss_cls = spa_loss_cls + obj_loss_cls + pos_loss_cls
        return bin_prob, hoi_prob, loss_bin, loss_cls, bin_error, hoi_error




