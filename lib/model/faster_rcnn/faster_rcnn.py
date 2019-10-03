import random
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


class _fasterRCNN(nn.Module):
    """ faster RCNN """

    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic

        # define rpn
        # self.RCNN_rpn = _RPN(self.dout_base_model)
        # self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0 / 16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0 / 16.0)

        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.RCNN_roi_crop = _RoICrop()

    def forward(self, im_data, im_info, hboxes, oboxes, iboxes, hoi_classes, bin_classes, num_hois):
        batch_size = im_data.size(0)

        im_info = im_info.data
        hboxes = hboxes.data
        oboxes = oboxes.data
        iboxes = iboxes.data
        num_hois = num_hois.data

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)

        hrois = Variable(torch.zeros(hboxes.shape[0], hboxes.shape[1], hboxes.shape[2] + 1))
        orois = Variable(torch.zeros(oboxes.shape[0], oboxes.shape[1], oboxes.shape[2] + 1))
        irois = Variable(torch.zeros(iboxes.shape[0], iboxes.shape[1], iboxes.shape[2] + 1))

        if im_data.is_cuda:
            hrois = hrois.cuda()
            orois = orois.cuda()
            irois = irois.cuda()

        hrois[:, :, 1:] = hboxes
        orois[:, :, 1:] = oboxes
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
        iroi_pooled_feat = self._head_to_tail(iroi_pooled_feat)

        if cfg.POOLING_MODE == 'crop':
            # pdb.set_trace()
            # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
            grid_xy = _affine_grid_gen(hrois.view(-1, 5), base_feat.size()[2:], self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:, :, :, 1], grid_xy.data[:, :, :, 0]], 3).contiguous()
            hroi_pooled_feat = self.RCNN_roi_crop(base_feat, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                hroi_pooled_feat = F.max_pool2d(hroi_pooled_feat, 2, 2)
        elif cfg.POOLING_MODE == 'align':
            hroi_pooled_feat = self.RCNN_roi_align(base_feat, hrois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            hroi_pooled_feat = self.RCNN_roi_pool(base_feat, hrois.view(-1, 5))

        # feed pooled features to top  model
        hroi_pooled_feat = self._head_to_tail(hroi_pooled_feat)

        if cfg.POOLING_MODE == 'crop':
            # pdb.set_trace()
            # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
            grid_xy = _affine_grid_gen(orois.view(-1, 5), base_feat.size()[2:], self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:, :, :, 1], grid_xy.data[:, :, :, 0]], 3).contiguous()
            oroi_pooled_feat = self.RCNN_roi_crop(base_feat, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                oroi_pooled_feat = F.max_pool2d(oroi_pooled_feat, 2, 2)
        elif cfg.POOLING_MODE == 'align':
            oroi_pooled_feat = self.RCNN_roi_align(base_feat, orois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            oroi_pooled_feat = self.RCNN_roi_pool(base_feat, orois.view(-1, 5))

        # feed pooled features to top  model
        oroi_pooled_feat = self._head_to_tail(oroi_pooled_feat)


        # compute object classification probability
        icls_score = self.iRCNN_cls_score(iroi_pooled_feat)
        ibin_score = self.iRCNN_bin_score(iroi_pooled_feat)

        hcls_score = self.hRCNN_cls_score(hroi_pooled_feat)
        hbin_score = self.hRCNN_bin_score(hroi_pooled_feat)

        ocls_score = self.oRCNN_cls_score(oroi_pooled_feat)
        obin_score = self.oRCNN_bin_score(oroi_pooled_feat)

        cls_score = (icls_score + hcls_score + ocls_score) / 3.0
        bin_score = (ibin_score + hbin_score + obin_score) / 3.0

        bin_prob = F.softmax(bin_score, 1)
        cls_prob = F.sigmoid(cls_score)

        RCNN_loss_cls = -1
        RCNN_loss_bin = -1

        if self.training:
            # classification loss
            RCNN_loss_cls = F.binary_cross_entropy(cls_prob, hoi_classes.view(-1, hoi_classes.shape[2]), size_average=False)
            RCNN_loss_bin = F.cross_entropy(bin_score, bin_classes.view(bin_classes.shape[1]), size_average=False)

        cls_prob = cls_prob.view(batch_size, irois.size(1), -1)
        bin_prob = bin_prob.view(batch_size, irois.size(1), -1)

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

        normal_init(self.iRCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.iRCNN_bin_score, 0, 0.01, cfg.TRAIN.TRUNCATED)

        normal_init(self.hRCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.hRCNN_bin_score, 0, 0.01, cfg.TRAIN.TRUNCATED)

        normal_init(self.oRCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.oRCNN_bin_score, 0, 0.01, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
