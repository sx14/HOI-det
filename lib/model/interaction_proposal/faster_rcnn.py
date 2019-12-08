import random
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta


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


class _fasterRCNN(nn.Module):
    """ faster RCNN """

    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic

        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0 / 16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0 / 16.0)

        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.RCNN_roi_crop = _RoICrop()
        self.spaCNN = SpaConv()

        self.spa_feat = nn.Linear(5408, 1024)

        self.obj_feat = nn.Linear(300, 1024)

        self.classifier = nn.Sequential(
              nn.Linear(1024 * 3, 1024 * 3),
              nn.LeakyReLU(),
              nn.Dropout(p=0.5),
              nn.Linear(1024 * 3, 1))

    def forward(self, im_data, im_info,
                hboxes, oboxes, iboxes,
                hoi_classes, bin_classes,
                hoi_masks, spa_maps,
                obj_vecs, num_hois):

        batch_size = im_data.size(0)
        iboxes = iboxes.data

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)
        irois = Variable(torch.zeros(iboxes.shape[0], iboxes.shape[1], iboxes.shape[2] + 1))

        if im_data.is_cuda:
            irois = irois.cuda()

        irois[:, :, 1:] = iboxes

        # do roi pooling based on predicted rois
        if cfg.POOLING_MODE == 'crop':
            # pdb.set_trace()
            # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
            grid_xy = _affine_grid_gen(irois.view(-1, 5), base_feat.size()[2:], self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:, :, :, 1], grid_xy.data[:, :, :, 0]], 3).contiguous()
            iroi_pooled_feat = self.RCNN_roi_crop(base_feat, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                iroi_pooled_feat = F.max_pool2d(iroi_pooled_feat, 2, 2)
        elif cfg.POOLING_MODE == 'align':
            iroi_pooled_feat = self.RCNN_roi_align(base_feat, irois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            iroi_pooled_feat = self.RCNN_roi_pool(base_feat, irois.view(-1, 5))

        # feed pooled features to top  model
        iroi_pooled_feat = self._ihead_to_tail(iroi_pooled_feat)

        spa_feat = self.spaCNN(spa_maps[0])
        spa_feat1 = self.spa_feat(spa_feat)

        obj_feat1 = self.obj_feat(obj_vecs[0])

        # compute object classification probability
        roi_feat1 = self.iRCNN_feat(iroi_pooled_feat)

        feat = torch.cat([spa_feat1, obj_feat1, roi_feat1], dim=1)
        cls_score = self.classifier(feat)
        cls_prob = F.sigmoid(cls_score)

        RCNN_loss_cls = 0
        RCNN_loss_bin = 0

        if self.training:
            # classification loss
            bin_classes = bin_classes[0, :, 0:1]
            bin_weights = bin_classes + 1
            RCNN_loss_cls = F.binary_cross_entropy(cls_prob, bin_classes, weight=bin_weights)

        cls_prob = cls_prob.view(batch_size, irois.size(1), -1)
        bin_prob = Variable(torch.zeros(batch_size, irois.size(1), 2)).cuda()

        return cls_prob, bin_prob, RCNN_loss_cls, RCNN_loss_bin

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        new_modules = [self.iRCNN_feat, self.spa_feat, self.obj_feat, self.classifier]
        for module in new_modules:
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if hasattr(layer, 'weight'):
                        normal_init(layer, 0, 0.01, cfg.TRAIN.TRUNCATED)
            else:
                normal_init(module, 0, 0.01, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
