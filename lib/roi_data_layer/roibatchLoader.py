
"""The data layer used during training to train a Fast R-CNN network.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
from PIL import Image
import torch

from model.utils.config import cfg
from roi_data_layer.minibatch import get_minibatch, get_minibatch
from roi_data_layer.spatial_map import gen_spatial_map
from roi_data_layer.pose_map import gen_pose_obj_map, gen_pose_obj_map1

import numpy as np
import random
import time
import pdb


class roibatchLoader(data.Dataset):
  def __init__(self, roidb, ratio_list, ratio_index, batch_size, num_classes, training=True, normalize=None):
    self._roidb = roidb
    self._num_classes = num_classes
    # we make the height of image consistent to trim_height, trim_width
    self.trim_height = cfg.TRAIN.TRIM_HEIGHT
    self.trim_width = cfg.TRAIN.TRIM_WIDTH
    self.max_num_box = cfg.MAX_NUM_GT_BOXES
    self.training = training
    self.normalize = normalize
    self.ratio_list = ratio_list
    self.ratio_index = ratio_index
    self.batch_size = batch_size
    self.data_size = len(self.ratio_list)

    # given the ratio_list, we want to make the ratio same for each batch.
    self.ratio_list_batch = torch.Tensor(self.data_size).zero_()
    num_batch = int(np.ceil(len(ratio_index) / batch_size))
    for i in range(num_batch):
        left_idx = i*batch_size
        right_idx = min((i+1)*batch_size-1, self.data_size-1)

        if ratio_list[right_idx] < 1:
            # for ratio < 1, we preserve the leftmost in each batch.
            target_ratio = ratio_list[left_idx]
        elif ratio_list[left_idx] > 1:
            # for ratio > 1, we preserve the rightmost in each batch.
            target_ratio = ratio_list[right_idx]
        else:
            # for ratio cross 1, we make it to be 1.
            target_ratio = np.array(1)

        self.ratio_list_batch[left_idx:(right_idx+1)] = torch.tensor(target_ratio.astype(np.float64)) # trainset ratio list ,each batch is same number

  def __getitem__(self, index):
    if self.training:
        index_ratio = int(self.ratio_index[index])
    else:
        index_ratio = index

    # get the anchor index for current sample index
    # here we set the anchor index to the last one
    # sample in this group
    minibatch_db = [self._roidb[index_ratio]]
    blobs = get_minibatch(minibatch_db, self._num_classes)
    im_data = torch.from_numpy(blobs['image'])
    dp_data = torch.from_numpy(blobs['depth'])
    im_info = torch.from_numpy(blobs['im_info'])
    # we need to random shuffle the bounding box.
    data_height, data_width = im_data.size(1), im_data.size(2)

    hoi_inds = np.array(range(len(blobs['hboxes'])))
    np.random.shuffle(hoi_inds)
    num_hoi = len(hoi_inds)

    blobs['hboxes'] = blobs['hboxes'][hoi_inds]
    blobs['oboxes'] = blobs['oboxes'][hoi_inds]
    blobs['iboxes'] = blobs['iboxes'][hoi_inds]

    blobs['pbox_lists'] = blobs['pbox_lists'][hoi_inds]
    blobs['pbox_lists1'] = blobs['pbox_lists1'][hoi_inds]

    blobs['hoi_classes'] = blobs['hoi_classes'][hoi_inds]
    blobs['vrb_classes'] = blobs['vrb_classes'][hoi_inds]
    blobs['bin_classes'] = blobs['bin_classes'][hoi_inds]

    blobs['hoi_masks'] = blobs['hoi_masks'][hoi_inds]
    blobs['vrb_masks'] = blobs['vrb_masks'][hoi_inds]

    gt_boxes = np.concatenate((blobs['hboxes'], blobs['oboxes'], blobs['iboxes']))
    gt_boxes = torch.from_numpy(gt_boxes)

    gt_pboxes = np.reshape(blobs['pbox_lists'], (blobs['pbox_lists'].shape[0], 6, 4))
    gt_pboxes = torch.from_numpy(gt_pboxes)

    gt_classes = torch.from_numpy(blobs['hoi_classes'])
    gt_verbs = torch.from_numpy(blobs['vrb_classes'])
    gt_binaries = torch.from_numpy(blobs['bin_classes'])
    gt_hoi_masks = torch.from_numpy(blobs['hoi_masks'])
    gt_vrb_masks = torch.from_numpy(blobs['vrb_masks'])

    raw_spa_maps = np.zeros((num_hoi, 2, 64, 64))
    for i in range(num_hoi):
        raw_spa_maps[i] = gen_spatial_map(blobs['hboxes'][i], blobs['oboxes'][i])
    gt_spa_maps = torch.from_numpy(raw_spa_maps).float()

    raw_pose_maps = np.zeros((num_hoi, 8, 224, 224))
    gt_pboxes1 = np.reshape(blobs['pbox_lists1'], (blobs['pbox_lists1'].shape[0], 6, 5))
    for i in range(num_hoi):
        raw_pose_maps[i] = gen_pose_obj_map1(blobs['hboxes'][i], blobs['oboxes'][i], blobs['iboxes'][i], gt_pboxes1[i])
    gt_pose_maps = torch.from_numpy(raw_pose_maps).float()

    ########################################################
    # padding the input image to fixed size for each group #
    ########################################################

    # NOTE1: need to cope with the case where a group cover both conditions. (done)
    # NOTE2: need to consider the situation for the tail samples. (no worry)
    # NOTE3: need to implement a parallel data loader. (no worry)
    # get the index range

    # if the image need to crop, crop to the target size.
    ratio = self.ratio_list_batch[index]

    if self._roidb[index_ratio]['need_crop']:
        if ratio < 1:
            # this means that data_width << data_height, we need to crop the
            # data_height
            min_y_p = int(torch.min(gt_pboxes[:, :, 1]))
            max_y_p = int(torch.max(gt_pboxes[:, :, 3]))
            min_y_b = int(torch.min(gt_boxes[:,1]))
            max_y_b = int(torch.max(gt_boxes[:,3]))
            max_y = max(max_y_b, max_y_p)
            min_y = min(min_y_b, min_y_p)

            trim_size = int(np.floor(data_width / ratio))
            if trim_size > data_height:
                trim_size = data_height
            box_region = max_y - min_y + 1
            if min_y == 0:
                y_s = 0
            else:
                if (box_region-trim_size) < 0:
                    y_s_min = max(max_y-trim_size, 0)
                    y_s_max = min(min_y, data_height-trim_size)
                    if y_s_min == y_s_max:
                        y_s = y_s_min
                    else:
                        y_s = np.random.choice(range(y_s_min, y_s_max))
                else:
                    y_s_add = int((box_region-trim_size)/2)
                    if y_s_add == 0:
                        y_s = min_y
                    else:
                        y_s = np.random.choice(range(min_y, min_y+y_s_add))
            # crop the image
            im_data = im_data[:, y_s:(y_s + trim_size), :, :]
            dp_data = dp_data[:, y_s:(y_s + trim_size), :, :]

            # shift y coordiante of gt_boxes
            gt_boxes[:, 1] = gt_boxes[:, 1] - float(y_s)
            gt_boxes[:, 3] = gt_boxes[:, 3] - float(y_s)
            gt_pboxes[:, :, 1] = gt_pboxes[:, :, 1] - float(y_s)
            gt_pboxes[:, :, 3] = gt_pboxes[:, :, 3] - float(y_s)

            # update gt bounding box according the trip
            gt_boxes[:, 1].clamp_(0, trim_size - 1)
            gt_boxes[:, 3].clamp_(0, trim_size - 1)
            gt_pboxes[:, :, 1].clamp_(0, trim_size - 1)
            gt_pboxes[:, :, 3].clamp_(0, trim_size - 1)

        else:
            # this means that data_width >> data_height, we need to crop the
            # data_width
            min_x_p = int(torch.min(gt_pboxes[:, :, 0]))
            max_x_p = int(torch.max(gt_pboxes[:, :, 2]))
            min_x_b = int(torch.min(gt_boxes[:, 0]))
            max_x_b = int(torch.max(gt_boxes[:, 2]))
            min_x = min(min_x_b, min_x_p)
            max_x = max(max_x_b, max_x_p)

            trim_size = int(np.ceil(data_height * ratio))
            if trim_size > data_width:
                trim_size = data_width
            box_region = max_x - min_x + 1
            if min_x == 0:
                x_s = 0
            else:
                if (box_region-trim_size) < 0:
                    x_s_min = max(max_x-trim_size, 0)
                    x_s_max = min(min_x, data_width-trim_size)
                    if x_s_min == x_s_max:
                        x_s = x_s_min
                    else:
                        x_s = np.random.choice(range(x_s_min, x_s_max))
                else:
                    x_s_add = int((box_region-trim_size)/2)
                    if x_s_add == 0:
                        x_s = min_x
                    else:
                        x_s = np.random.choice(range(min_x, min_x+x_s_add))
            # crop the image
            im_data = im_data[:, :, x_s:(x_s + trim_size), :]
            dp_data = dp_data[:, :, x_s:(x_s + trim_size), :]

            # shift x coordiante of gt_boxes
            gt_boxes[:, 0] = gt_boxes[:, 0] - float(x_s)
            gt_boxes[:, 2] = gt_boxes[:, 2] - float(x_s)
            gt_pboxes[:, :, 0] = gt_pboxes[:, :, 0] - float(x_s)
            gt_pboxes[:, :, 2] = gt_pboxes[:, :, 2] - float(x_s)
            # update gt bounding box according the trip
            gt_boxes[:, 0].clamp_(0, trim_size - 1)
            gt_boxes[:, 2].clamp_(0, trim_size - 1)
            gt_pboxes[:, :, 0].clamp_(0, trim_size - 1)
            gt_pboxes[:, :, 2].clamp_(0, trim_size - 1)

    # based on the ratio, padding the image.
    if ratio < 1:
        # this means that data_width < data_height
        trim_size = int(np.floor(data_width / ratio))

        padding_im_data = torch.FloatTensor(int(np.ceil(data_width / ratio)), \
                                         data_width, im_data.shape[-1]).zero_()

        padding_im_data[:data_height, :, :] = im_data[0]

        padding_dp_data = torch.FloatTensor(int(np.ceil(data_width / ratio)), \
                                            data_width, dp_data.shape[-1]).zero_()

        padding_dp_data[:data_height, :, :] = dp_data[0]
        # update im_info
        im_info[0, 0] = padding_im_data.size(0)
        # print("height %d %d \n" %(index, anchor_idx))
    elif ratio > 1:
        # this means that data_width > data_height
        # if the image need to crop.
        padding_im_data = torch.FloatTensor(data_height, \
                                         int(np.ceil(data_height * ratio)), im_data.shape[-1]).zero_()
        padding_im_data[:, :data_width, :] = im_data[0]

        padding_dp_data = torch.FloatTensor(data_height, \
                                            int(np.ceil(data_height * ratio)), dp_data.shape[-1]).zero_()
        padding_dp_data[:, :data_width, :] = dp_data[0]

        im_info[0, 1] = padding_im_data.size(1)
    else:
        trim_size = min(data_height, data_width)
        padding_im_data = torch.FloatTensor(trim_size, trim_size, 3).zero_()
        padding_im_data = im_data[0][:trim_size, :trim_size, :]
        padding_dp_data = dp_data[0][:trim_size, :trim_size, :]
        # gt_boxes.clamp_(0, trim_size)
        gt_boxes[:, :4].clamp_(0, trim_size)
        gt_pboxes[:, :, :4].clamp_(0, trim_size)
        im_info[0, 0] = trim_size
        im_info[0, 1] = trim_size

    # check the bounding box:
    not_keep = (gt_boxes[:, 0] == gt_boxes[:, 2]) | (gt_boxes[:, 1] == gt_boxes[:, 3])
    for i in range(not_keep.shape[0]):
        if not_keep[i] == 1:
            ii = i % num_hoi
            not_keep[ii + num_hoi * 0] = 1
            not_keep[ii + num_hoi * 1] = 1
            not_keep[ii + num_hoi * 2] = 1

    keep3 = torch.nonzero(not_keep == 0).view(-1)
    keep  = torch.nonzero(not_keep[:num_hoi] == 0).view(-1)

    if keep3.numel() != 0:

        assert keep3.shape[0] == keep.shape[0] * 3

        gt_boxes = gt_boxes[keep3]
        gt_pboxes = gt_pboxes[keep]
        gt_classes = gt_classes[keep]
        gt_verbs = gt_verbs[keep]
        gt_binaries = gt_binaries[keep]
        gt_spa_maps = gt_spa_maps[keep]
        gt_hoi_masks = gt_hoi_masks[keep]
        gt_vrb_masks = gt_vrb_masks[keep]
        gt_pose_maps = gt_pose_maps[keep]

        gt_num_boxes = int(gt_boxes.size(0) / 3)
        assert gt_num_boxes * 3 == gt_boxes.size(0)

        num_boxes = int(min(gt_num_boxes, self.max_num_box))
        hboxes_padding = gt_boxes[gt_num_boxes * 0: gt_num_boxes * 0 + num_boxes]
        oboxes_padding = gt_boxes[gt_num_boxes * 1: gt_num_boxes * 1 + num_boxes]
        iboxes_padding = gt_boxes[gt_num_boxes * 2: gt_num_boxes * 2 + num_boxes]

        pboxes_padding = gt_pboxes[:num_boxes]
        hoi_classes_padding = gt_classes[:num_boxes]
        vrb_classes_padding = gt_verbs[:num_boxes]
        bin_classes_padding = gt_binaries[:num_boxes].long()
        spa_maps_padding = gt_spa_maps[:num_boxes]
        hoi_masks_padding = gt_hoi_masks[:num_boxes]
        vrb_masks_padding = gt_vrb_masks[:num_boxes]
        pose_maps_padding = gt_pose_maps[:num_boxes]
    else:
        hboxes_padding = torch.FloatTensor(1, gt_boxes.size(1)).zero_()
        oboxes_padding = torch.FloatTensor(1, gt_boxes.size(1)).zero_()
        iboxes_padding = torch.FloatTensor(1, gt_boxes.size(1)).zero_()
        pboxes_padding = torch.FloatTensor(1, gt_pboxes.size(1), gt_pboxes.size(2)).zero_()
        hoi_classes_padding = torch.FloatTensor(1, gt_classes.size(1)).zero_()
        vrb_classes_padding = torch.FloatTensor(1, gt_verbs.size(1)).zero_()
        bin_classes_padding = torch.LongTensor(1).zero_()
        spa_maps_padding = torch.LongTensor(1, 2, 64, 64).zero_()
        hoi_masks_padding = torch.LongTensor(1, gt_classes.size(1)).zero_()
        vrb_masks_padding = torch.LongTensor(1, gt_verbs.size(1)).zero_()
        pose_maps_padding = torch.LongTensor(1, 8, 64, 64).zero_()
        num_boxes = 0

        # permute trim_data to adapt to downstream processing
    padding_im_data = padding_im_data.permute(2, 0, 1).contiguous()
    padding_dp_data = padding_dp_data.permute(2, 0, 1).contiguous()
    im_info = im_info.view(3)

    return padding_im_data, padding_dp_data, im_info, \
           hboxes_padding, oboxes_padding, iboxes_padding, pboxes_padding, \
           hoi_classes_padding, vrb_classes_padding, bin_classes_padding, \
           hoi_masks_padding, vrb_masks_padding, \
           spa_maps_padding, pose_maps_padding, num_boxes

  def __len__(self):
    return len(self._roidb)
