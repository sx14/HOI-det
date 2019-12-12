
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
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes

import numpy as np
import random
import time
import pdb


def bbox_trans(human_box_roi, object_box_roi, size=64):
    human_box = human_box_roi.copy()
    object_box = object_box_roi.copy()

    union_box = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]),
                          max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]

    height = union_box[3] - union_box[1] + 1
    width = union_box[2] - union_box[0] + 1

    if height > width:
        ratio = 'height'
    else:
        ratio = 'width'

    # shift the top-left corner to (0,0)
    human_box[0] -= union_box[0]
    human_box[2] -= union_box[0]
    human_box[1] -= union_box[1]
    human_box[3] -= union_box[1]
    object_box[0] -= union_box[0]
    object_box[2] -= union_box[0]
    object_box[1] -= union_box[1]
    object_box[3] -= union_box[1]

    if ratio == 'height':  # height is larger than width

        human_box[0] = 0 + size * human_box[0] / height
        human_box[1] = 0 + size * human_box[1] / height
        human_box[2] = (size * width / height - 1) - size * (width - 1 - human_box[2]) / height
        human_box[3] = (size - 1) - size * (height - 1 - human_box[3]) / height

        object_box[0] = 0 + size * object_box[0] / height
        object_box[1] = 0 + size * object_box[1] / height
        object_box[2] = (size * width / height - 1) - size * (width - 1 - object_box[2]) / height
        object_box[3] = (size - 1) - size * (height - 1 - object_box[3]) / height

        # Need to shift horizontally
        union_box = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]),
                              max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]
        if human_box[3] > object_box[3]:
            human_box[3] = size - 1
        else:
            object_box[3] = size - 1

        shift = size / 2 - (union_box[2] + 1) / 2
        human_box += [shift, 0, shift, 0]
        object_box += [shift, 0, shift, 0]

    else:  # width is larger than height

        human_box[0] = 0 + size * human_box[0] / width
        human_box[1] = 0 + size * human_box[1] / width
        human_box[2] = (size - 1) - size * (width - 1 - human_box[2]) / width
        human_box[3] = (size * height / width - 1) - size * (height - 1 - human_box[3]) / width

        object_box[0] = 0 + size * object_box[0] / width
        object_box[1] = 0 + size * object_box[1] / width
        object_box[2] = (size - 1) - size * (width - 1 - object_box[2]) / width
        object_box[3] = (size * height / width - 1) - size * (height - 1 - object_box[3]) / width

        # Need to shift vertically
        union_box = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]),
                              max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]

        if human_box[2] > object_box[2]:
            human_box[2] = size - 1
        else:
            object_box[2] = size - 1

        shift = size / 2 - (union_box[3] + 1) / 2

        human_box = human_box + [0, shift, 0, shift]
        object_box = object_box + [0, shift, 0, shift]

    return np.round(human_box), np.round(object_box)


def gen_spatial_map(human_box, object_box):
    hbox, obox = bbox_trans(human_box, object_box)
    spa_map = np.zeros((2, 64, 64), dtype='float32')
    spa_map[0, int(hbox[1]):int(hbox[3]) + 1, int(hbox[0]):int(hbox[2]) + 1] = 1
    spa_map[1, int(obox[1]):int(obox[3]) + 1, int(obox[0]):int(obox[2]) + 1] = 1
    return spa_map


class roibatchLoader(data.Dataset):
  def __init__(self, roidb, ratio_list, ratio_index, batch_size, num_classes, obj2vec, training=True, normalize=None):
    self._roidb = roidb
    self._obj2vec = obj2vec
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
    data = torch.from_numpy(blobs['data'])
    im_info = torch.from_numpy(blobs['im_info'])
    # we need to random shuffle the bounding box.
    data_height, data_width = data.size(1), data.size(2)

    hoi_inds = np.array(range(len(blobs['hboxes'])))
    np.random.shuffle(hoi_inds)
    num_hoi = len(hoi_inds)

    blobs['hboxes'] = blobs['hboxes'][hoi_inds]
    blobs['oboxes'] = blobs['oboxes'][hoi_inds]
    blobs['iboxes'] = blobs['iboxes'][hoi_inds]
    blobs['vrb_classes'] = blobs['vrb_classes'][hoi_inds]
    blobs['obj_classes'] = blobs['obj_classes'][hoi_inds]
    blobs['bin_classes'] = blobs['bin_classes'][hoi_inds]
    blobs['vrb_masks'] = blobs['vrb_masks'][hoi_inds]

    gt_boxes = np.concatenate((blobs['hboxes'], blobs['oboxes'], blobs['iboxes']))
    gt_boxes = torch.from_numpy(gt_boxes)

    gt_verbs = np.tile(blobs['vrb_classes'], 3)
    gt_verbs = torch.from_numpy(gt_verbs)

    gt_binaries = np.tile(blobs['bin_classes'], (3, 1))
    gt_binaries = torch.from_numpy(gt_binaries)

    gt_vrb_masks = np.tile(blobs['vrb_masks'], (3, 1))
    gt_vrb_masks = torch.from_numpy(gt_vrb_masks)

    raw_spa_maps = np.zeros((num_hoi, 2, 64, 64))
    for i in range(num_hoi):
        raw_spa_maps[i] = gen_spatial_map(blobs['hboxes'][i], blobs['oboxes'][i])
    raw_spa_maps = np.tile(raw_spa_maps, (3, 1, 1, 1))
    gt_spa_maps = torch.from_numpy(raw_spa_maps).float()

    raw_obj_vecs = self._obj2vec[blobs['obj_classes']]
    raw_obj_vecs = np.tile(raw_obj_vecs, (3, 1))
    gt_obj_vecs = torch.from_numpy(raw_obj_vecs).float()

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
            min_y = int(torch.min(gt_boxes[:,1]))
            max_y = int(torch.max(gt_boxes[:,3]))
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
            data = data[:, y_s:(y_s + trim_size), :, :]

            # shift y coordiante of gt_boxes
            gt_boxes[:, 1] = gt_boxes[:, 1] - float(y_s)
            gt_boxes[:, 3] = gt_boxes[:, 3] - float(y_s)

            # update gt bounding box according the trip
            gt_boxes[:, 1].clamp_(0, trim_size - 1)
            gt_boxes[:, 3].clamp_(0, trim_size - 1)

        else:
            # this means that data_width >> data_height, we need to crop the
            # data_width
            min_x = int(torch.min(gt_boxes[:,0]))
            max_x = int(torch.max(gt_boxes[:,2]))
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
            data = data[:, :, x_s:(x_s + trim_size), :]

            # shift x coordiante of gt_boxes
            gt_boxes[:, 0] = gt_boxes[:, 0] - float(x_s)
            gt_boxes[:, 2] = gt_boxes[:, 2] - float(x_s)
            # update gt bounding box according the trip
            gt_boxes[:, 0].clamp_(0, trim_size - 1)
            gt_boxes[:, 2].clamp_(0, trim_size - 1)

    # based on the ratio, padding the image.
    if ratio < 1:
        # this means that data_width < data_height
        trim_size = int(np.floor(data_width / ratio))

        padding_data = torch.FloatTensor(int(np.ceil(data_width / ratio)), \
                                         data_width, 3).zero_()

        padding_data[:data_height, :, :] = data[0]
        # update im_info
        im_info[0, 0] = padding_data.size(0)
        # print("height %d %d \n" %(index, anchor_idx))
    elif ratio > 1:
        # this means that data_width > data_height
        # if the image need to crop.
        padding_data = torch.FloatTensor(data_height, \
                                         int(np.ceil(data_height * ratio)), 3).zero_()
        padding_data[:, :data_width, :] = data[0]
        im_info[0, 1] = padding_data.size(1)
    else:
        trim_size = min(data_height, data_width)
        padding_data = torch.FloatTensor(trim_size, trim_size, 3).zero_()
        padding_data = data[0][:trim_size, :trim_size, :]
        # gt_boxes.clamp_(0, trim_size)
        gt_boxes[:, :4].clamp_(0, trim_size)
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

    keep = torch.nonzero(not_keep == 0).view(-1)

    if keep.numel() != 0:
        gt_boxes = gt_boxes[keep]
        gt_verbs = gt_verbs[keep]
        gt_binaries = gt_binaries[keep]
        gt_spa_maps = gt_spa_maps[keep]
        gt_vrb_masks = gt_vrb_masks[keep]
        gt_obj_vecs = gt_obj_vecs[keep]

        gt_num_boxes = int(gt_boxes.size(0) / 3)

        assert gt_num_boxes * 3 == gt_boxes.size(0)

        num_boxes = int(min(gt_num_boxes, self.max_num_box))
        hboxes_padding = gt_boxes[gt_num_boxes * 0: gt_num_boxes * 0 + num_boxes]
        oboxes_padding = gt_boxes[gt_num_boxes * 1: gt_num_boxes * 1 + num_boxes]
        iboxes_padding = gt_boxes[gt_num_boxes * 2: gt_num_boxes * 2 + num_boxes]

        vrb_classes_padding = gt_verbs[:num_boxes].long()
        bin_classes_padding = gt_binaries[:num_boxes].long()
        spa_maps_padding = gt_spa_maps[:num_boxes]
        vrb_masks_padding = gt_vrb_masks[:num_boxes]
        obj_vecs_padding = gt_obj_vecs[:num_boxes]
    else:
        hboxes_padding = torch.FloatTensor(1, gt_boxes.size(1)).zero_()
        oboxes_padding = torch.FloatTensor(1, gt_boxes.size(1)).zero_()
        iboxes_padding = torch.FloatTensor(1, gt_boxes.size(1)).zero_()
        vrb_classes_padding = torch.LongTensor(1).zero_()
        bin_classes_padding = torch.LongTensor(1).zero_()
        spa_maps_padding = torch.FloatTensor(1, 2, 64, 64).zero_()
        vrb_masks_padding = torch.FloatTensor(1, gt_vrb_masks.size(1)).zero_()
        obj_vecs_padding = torch.FloatTensor(1, self._obj2vec.shape[1]).zero_()
        num_boxes = 0

        # permute trim_data to adapt to downstream processing
    padding_data = padding_data.permute(2, 0, 1).contiguous()
    im_info = im_info.view(3)

    return padding_data, im_info, \
           hboxes_padding, oboxes_padding, iboxes_padding, \
           vrb_classes_padding, bin_classes_padding, \
           vrb_masks_padding, spa_maps_padding, \
           obj_vecs_padding, num_boxes

  def __len__(self):
    return len(self._roidb)
