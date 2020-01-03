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

        self.hidden = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(p=0.5))

    def forward(self, spa_map):
        conv1 = self.conv1(spa_map)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        pool_feat = pool2.view(spa_map.shape[0], -1)
        return self.hidden(pool_feat)


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
        self.spaCNN = SpaConv()

        self.spa_cls_score = nn.Sequential(
            nn.Linear(5408, 1024),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, self.n_classes))

        self.obj_cls_score = nn.Sequential(
            nn.Linear(300, 512),
            nn.LeakyReLU(),
            nn.Linear(512, self.n_classes))

        self.obj_attention = nn.Sequential(
            nn.Linear(300, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 6))

    def forward(self, im_data, im_info,
                hboxes, oboxes, iboxes,
                pboxes, sboxes,
                hoi_classes, bin_classes,
                hoi_masks, spa_maps,
                obj_vecs, num_hois):

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
        prois = Variable(torch.zeros(pboxes.shape[0], pboxes.shape[1] * pboxes.shape[2], pboxes.shape[3] + 1))
        srois = Variable(torch.zeros(sboxes.shape[0], sboxes.shape[1], sboxes.shape[2] + 1))

        if im_data.is_cuda:
            hrois = hrois.cuda()
            orois = orois.cuda()
            irois = irois.cuda()
            prois = prois.cuda()
            srois = srois.cuda()

        hrois[:, :, 1:] = hboxes
        orois[:, :, 1:] = oboxes
        irois[:, :, 1:] = iboxes
        prois[:, :, 1:] = pboxes.view(pboxes.shape[0], -1, pboxes.shape[3])
        srois[:, :, 1:] = sboxes

        iroi_pooled_feat = self.RCNN_roi_align(base_feat, irois.view(-1, 5))
        iroi_pooled_feat = self._ihead_to_tail(iroi_pooled_feat)

        hroi_pooled_feat = self.RCNN_roi_align(base_feat, hrois.view(-1, 5))
        hroi_pooled_feat = self._hhead_to_tail(hroi_pooled_feat)

        oroi_pooled_feat = self.RCNN_roi_align(base_feat, orois.view(-1, 5))
        oroi_pooled_feat = self._ohead_to_tail(oroi_pooled_feat)

        proi_pooled_feat = self.RCNN_roi_align(base_feat, prois.view(-1, 5))
        proi_pooled_feat = self._phead_to_tail(proi_pooled_feat)

        sroi_pooled_feat = self.RCNN_roi_align(base_feat, srois.view(-1, 5))
        sroi_pooled_feat = self._shead_to_tail(sroi_pooled_feat)

        spa_feat = self.spaCNN(spa_maps[0])
        scls_score = self.spa_cls_score(spa_feat)
        scls_prob = F.sigmoid(scls_score)

        # vcls_score = self.obj_cls_score(obj_vecs[0])
        # vcls_prob = F.sigmoid(vcls_score)

        # compute object classification probability
        icls_score = self.iRCNN_cls_score(iroi_pooled_feat)
        icls_prob = F.sigmoid(icls_score)

        hcls_score = self.hRCNN_cls_score(hroi_pooled_feat)
        hcls_prob = F.sigmoid(hcls_score)

        ocls_score = self.oRCNN_cls_score(oroi_pooled_feat)
        ocls_prob = F.sigmoid(ocls_score)

        # obj_att = self.obj_attention(obj_vecs[0])
        # part_att_feats = []
        # for i in range(obj_att.shape[1]):
        #     part_att = obj_att[:, i:i+1]
        #     part_feat = proi_pooled_feat[:, i*2048:(i+1)*2048]
        #     part_att_feat = part_att * part_feat
        #     part_att_feats.append(part_att_feat)
        # att_proi_pooled_feat = torch.cat(part_att_feats, dim=1)
        # att_proi_pooled_feat = Variable(att_proi_pooled_feat.cuda())
        # att_proi_pooled_feat = att_proi_pooled_feat + proi_pooled_feat
        # pcls_score = self.pRCNN_cls_score(att_proi_pooled_feat)

        pcls_score = self.pRCNN_cls_score(proi_pooled_feat)
        pcls_prob = F.sigmoid(pcls_score)

        ccls_score = self.sRCNN_cls_score(sroi_pooled_feat)
        ccls_prob = F.sigmoid(ccls_score)
        ccls_prob = ccls_prob.repeat((icls_prob.shape[0], 1))

        # cls_prob = (icls_prob + hcls_prob + ocls_prob + pcls_prob + ccls_prob) * scls_prob * vcls_prob
        cls_prob = (icls_prob + hcls_prob + ocls_prob + pcls_prob + ccls_prob) * scls_prob

        RCNN_loss_cls = 0
        RCNN_loss_bin = 0

        if self.training:
            # classification loss
            pos_map = bin_classes[0, :, 0].long()
            hoi_masks = hoi_masks.view(-1, hoi_masks.shape[2])
            scls_loss = F.binary_cross_entropy(scls_prob * hoi_masks, hoi_classes.view(-1, hoi_classes.shape[2]), size_average=False)
            icls_loss = F.binary_cross_entropy(icls_prob * hoi_masks, hoi_classes.view(-1, hoi_classes.shape[2]), size_average=False)
            hcls_loss = F.binary_cross_entropy(hcls_prob * hoi_masks, hoi_classes.view(-1, hoi_classes.shape[2]), size_average=False)
            ocls_loss = F.binary_cross_entropy(ocls_prob * hoi_masks, hoi_classes.view(-1, hoi_classes.shape[2]), size_average=False)
            # vcls_loss = F.binary_cross_entropy(vcls_prob * hoi_masks, hoi_classes.view(-1, hoi_classes.shape[2]), size_average=False)
            pcls_loss = F.binary_cross_entropy(pcls_prob * hoi_masks, hoi_classes.view(-1, hoi_classes.shape[2]), size_average=False)
            ccls_loss = F.binary_cross_entropy(ccls_prob * hoi_masks, hoi_classes.view(-1, hoi_classes.shape[2]), size_average=False)
            # RCNN_loss_cls = scls_loss + icls_loss + hcls_loss + ocls_loss + vcls_loss + pcls_loss + ccls_loss
            RCNN_loss_cls = scls_loss + icls_loss + hcls_loss + ocls_loss + pcls_loss + ccls_loss

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

        # new_modules = [self.iRCNN_cls_score,
        #                self.hRCNN_cls_score,
        #                self.oRCNN_cls_score,
        #                self.obj_cls_score,
        #                self.spa_cls_score]
        #
        # for module in new_modules:
        #     for layer in module:
        #         if hasattr(layer, 'weight'):
        #             normal_init(layer, 0, 0.01, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
