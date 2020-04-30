from __future__ import print_function
from __future__ import absolute_import
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import xml.dom.minidom as minidom

import os
import cv2
from collections import defaultdict
# import PIL
import numpy as np
import scipy.sparse
import subprocess
import math
import random
from random import randint
import glob
import uuid
import scipy.io as sio
import xml.etree.ElementTree as ET
import pickle
from .imdb import imdb
from .imdb import ROOT_DIR
from . import ds_utils
from .voc_eval import voc_eval
from datasets.pose_map import est_part_boxes, gen_part_boxes
# TODO: make fast_rcnn irrelevant
# >>>> obsolete, because it depends on sth outside of this project
from model.utils.config import cfg

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

# <<<< obsolete


def iou(box1, box2):
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2

    xi1 = max(x11, x21)
    yi1 = max(y11, y21)
    xi2 = min(x12, x22)
    yi2 = min(y12, y22)

    wi = xi2 - xi1 + 1
    hi = yi2 - yi1 + 1

    if wi > 0 and hi > 0:
        areai = wi * hi

        w1 = x12 - x11 + 1
        h1 = y12 - y11 + 1
        area1 = w1 * h1

        w2 = x22 - x21 + 1
        h2 = y22 - y21 + 1
        area2 = w2 * h2

        iou = areai * 1.0 / (area1 + area2 - areai)
    else:
        iou = 0

    return iou


class vidor_hoid(imdb):

    @staticmethod
    def load_obj2vec(data_path):
        obj2vec_path = os.path.join(data_path, 'object_vectors.mat')
        with open(obj2vec_path) as f:
            obj2vec = pickle.load(f)
        return obj2vec

    @staticmethod
    def load_hoi_classes(data_path):
        obj_cate_path = os.path.join(data_path, 'object_labels.txt')
        pre_cate_path = os.path.join(data_path, 'predicate_labels.txt')
        hoi_cate_path = os.path.join(data_path, 'interaction_labels.txt')
        with open(obj_cate_path) as f:
            obj_cates = [line.strip() for line in f.readlines()]
        with open(pre_cate_path) as f:
            pre_cates = ['__no_interaction__'] + [line.strip() for line in f.readlines()]
        with open(hoi_cate_path) as f:
            # N obj_cate base (with "__no_interaction__obj")
            hoi_cates = ['__no_interaction__+'+obj_cate for obj_cate in obj_cates] + [line.strip() for line in f.readlines()]
        return obj_cates, pre_cates, hoi_cates

    def __init__(self, image_set, version):
        imdb.__init__(self, 'vidor_hoid_' + version + '_' + image_set)
        self._version = version
        self._image_set = image_set
        self._data_path = self._get_default_path()
        self.obj_classes, self.vrb_classes, self.hoi_classes = self.load_hoi_classes(self._data_path)
        self._classes = self.hoi_classes
        self.obj_class2ind = dict(zip(self.obj_classes, xrange(len(self.obj_classes))))
        self.vrb_class2ind = dict(zip(self.vrb_classes, xrange(len(self.vrb_classes))))
        self.hoi_class2ind = dict(zip(self.hoi_classes, xrange(len(self.hoi_classes))))

        self._class_to_ind = dict(zip(self._classes, xrange(len(self._classes))))
        self._image_ext = '.JPEG'
        self._all_image_info = self._load_image_set_info()
        self._obj2vec = None
        self._image_index = None
        # Default to roidb handler
        # self._roidb_handler = self.selective_search_roidb
        self._roidb_handler = self.gt_roidb

    @property
    def obj2vec(self):
        if self._obj2vec is None:
            self._obj2vec = self.load_obj2vec(self._data_path)
        return self._obj2vec

    @property
    def roidb(self):
        # A roidb is a list of dictionaries, each with the following keys:
        #   boxes
        #   gt_overlaps
        #   gt_classes
        #   flipped
        if self._roidb is not None and self._image_index is not None:
            return self._roidb

        roidb_dict = self.roidb_handler()
        self._image_index = sorted(roidb_dict.keys())
        self._roidb = [roidb_dict[image_id] for image_id in self._image_index]
        return self._roidb

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return i

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'Data', 'VID', self._image_set, index + self._image_ext)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_info(self):
        print('Loading image set info ...')
        cache_file = os.path.join(self.cache_path, self.name + '_gt_img_info.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                all_image_info = pickle.load(fid)
            print('{} image set info loaded from {}'.format(self.name, cache_file))
            return all_image_info

        image_root = os.path.join(self._data_path, 'Data', 'VID', self._image_set)

        all_image_info = {}
        for image_file in os.listdir(image_root):
            image_id = image_file.split('.')[0]
            image_path = os.path.join(image_root, image_file)
            image = cv2.imread(image_path)
            im_h, im_w = image.shape[:2]
            all_image_info[image_id] = [im_w, im_h]

        with open(cache_file, 'wb') as fid:
            pickle.dump(all_image_info, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote image set info to {}'.format(cache_file))
        return all_image_info

    def _get_default_path(self):
        return os.path.join(cfg.DATA_DIR, 'vidor_hoid_'+self._version, 'image_data')

    def gt_roidb(self):
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                gt_roidb_dict = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return gt_roidb_dict

        gt_roidb_dict = self._load_all_annotations()

        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb_dict, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))
        return gt_roidb_dict

    @staticmethod
    def augment_box(bbox, shape, augment=5):

        def bb_IOU(boxA, boxB):

            ixmin = np.maximum(boxA[0], boxB[0])
            iymin = np.maximum(boxA[1], boxB[1])
            ixmax = np.minimum(boxA[2], boxB[2])
            iymax = np.minimum(boxA[3], boxB[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((boxB[2] - boxB[0] + 1.) * (boxB[3] - boxB[1] + 1.) +
                   (boxA[2] - boxA[0] + 1.) * (boxA[3] - boxA[1] + 1.) - inters)

            overlaps = inters / uni
            return overlaps

        thres_ = 0.7

        box = np.array([bbox[0], bbox[1], bbox[2], bbox[3]]).reshape(1, 4)
        box = box.astype(np.float64)

        count = 0
        time_count = 0
        while count < augment:

            time_count += 1
            height = bbox[3] - bbox[1]
            width = bbox[2] - bbox[0]

            height_cen = (bbox[3] + bbox[1]) / 2
            width_cen = (bbox[2] + bbox[0]) / 2

            ratio = 1 + randint(-10, 10) * 0.01

            height_shift = randint(-np.floor(height), np.floor(height)) * 0.1
            width_shift = randint(-np.floor(width), np.floor(width)) * 0.1

            H_0 = max(0, width_cen + width_shift - ratio * width / 2)
            H_2 = min(shape[1] - 1, width_cen + width_shift + ratio * width / 2)
            H_1 = max(0, height_cen + height_shift - ratio * height / 2)
            H_3 = min(shape[0] - 1, height_cen + height_shift + ratio * height / 2)

            if bb_IOU(bbox, np.array([H_0, H_1, H_2, H_3])) > thres_:
                box_ = np.array([H_0, H_1, H_2, H_3]).reshape(1, 4)
                box = np.concatenate((box, box_), axis=0)
                count += 1
            if time_count > 150:
                return box
        return box

    def augment_hoi_instances(self, raw_hois, im_hw):
        new_hois = []
        for raw_hoi in raw_hois:

            hbox = raw_hoi[2][:4]
            obox = raw_hoi[3][:4]
            aug_hboxes = self.augment_box(hbox, im_hw)
            aug_oboxes = self.augment_box(obox, im_hw)

            aug_hboxes = aug_hboxes[:min(len(aug_hboxes), len(aug_oboxes))]
            aug_oboxes = aug_oboxes[:min(len(aug_hboxes), len(aug_oboxes))]

            for i in range(aug_hboxes.shape[0]):
                aug_cls_ids = raw_hoi[1]
                aug_hbox = aug_hboxes[i].tolist()
                aug_hbox.append(raw_hoi[3][4])
                aug_obox = aug_oboxes[i].tolist()
                aug_obox.append(raw_hoi[3][4])
                new_hois.append([0,             # stub
                                 aug_cls_ids,
                                 aug_hbox,
                                 aug_obox,
                                 raw_hoi[4]])
        return new_hois

    def _gen_obj2pre_mask(self, pos_insts):
        obj2pre_mask = np.zeros((len(self.obj_classes), len(self.vrb_classes)))
        obj2pre_mask[:, 0] = 1  # _no_interaction_
        for inst in pos_insts:
            obj_cls_idx = inst[3][4]
            pre_cls_idx_list = inst[1]
            for pre_cls_idx in pre_cls_idx_list:
                obj2pre_mask[obj_cls_idx, pre_cls_idx] = 1
        return obj2pre_mask

    def _load_all_annotations(self):
        print('Loading raw annotations ...')
        anno_ng_tmp = pickle.load(open(os.path.join(self._data_path, '%s_NEG_with_pose.pkl' % self._image_set)))
        anno_gt_tmp = pickle.load(open(os.path.join(self._data_path, '%s_POS_with_pose.pkl' % self._image_set)))
        obj2pre_mask = self._gen_obj2pre_mask(anno_gt_tmp)

        print('Processing annotations ...')
        anno_gt_db = defaultdict(list)
        for hoi_ins_gt in anno_gt_tmp:
            image_id = hoi_ins_gt[0]
            anno_gt_db[image_id].append(hoi_ins_gt)

        anno_ng_db = defaultdict(list)
        for hoi_ins_gt in anno_ng_tmp:
            image_id = hoi_ins_gt[0]
            anno_ng_db[image_id].append(hoi_ins_gt)

        all_annos = {}
        for image_id, img_pos_hois in anno_gt_db.items():
            image_name = image_id

            # augment positive instances
            image_hw = [self._all_image_info[image_name][1],
                        self._all_image_info[image_name][0]]
            img_pos_hois = self.augment_hoi_instances(img_pos_hois, image_hw)

            # select negative instances
            if image_id in anno_ng_db and len(anno_ng_db[image_id]) > 0:
                img_neg_hois0 = anno_ng_db[image_id]
                if len(img_neg_hois0) > len(img_pos_hois):
                    inds = random.sample(range(len(img_neg_hois0)), len(img_pos_hois))
                else:
                    inds = []
                    for i in range(int(len(img_pos_hois) / len(img_neg_hois0))):
                        inds += range(len(img_neg_hois0))
                    for i in range(len(img_pos_hois) - len(inds)):
                        inds.append(i)
                img_neg_hois = [img_neg_hois0[ind] for ind in inds]
                assert len(img_neg_hois) == (len(img_pos_hois))
            else:
                img_neg_hois = []

            # boxes: x1, y1, x2, y2
            image_anno = {'hboxes': [],
                          'oboxes': [],
                          'iboxes': [],
                          'pbox_lists': [],
                          'hoi_classes': [],
                          'obj_classes': [],
                          'vrb_classes': [],
                          'bin_classes': [],
                          'hoi_masks': [],
                          'vrb_masks': [],
                          'width': self._all_image_info[image_name][0],
                          'height': self._all_image_info[image_name][1],
                          'flipped': False}
            all_annos[image_name] = image_anno

            for pn, hois in enumerate([img_pos_hois, img_neg_hois]):
                for raw_hoi in hois:

                    vrb_class_ids = raw_hoi[1]
                    hbox = raw_hoi[2][:4]
                    obox = raw_hoi[3][:4]
                    obj_class_id = raw_hoi[3][4]
                    ibox = [min(hbox[0], obox[0]), min(hbox[1], obox[1]),
                            max(hbox[2], obox[2]), max(hbox[3], obox[3])]
                    image_anno['hboxes'].append(hbox)
                    image_anno['oboxes'].append(obox)
                    image_anno['iboxes'].append(ibox)
                    image_anno['vrb_classes'].append(vrb_class_ids)
                    image_anno['obj_classes'].append(obj_class_id)
                    hoi_class_ids = [self.hoi_class2ind[self.vrb_classes[vrb_class_id]+'+'+self.obj_classes[obj_class_id]]
                                     for vrb_class_id in vrb_class_ids]
                    image_anno['hoi_classes'].append(hoi_class_ids)
                    image_anno['vrb_masks'].append(obj2pre_mask[obj_class_id].tolist())

                    raw_key_points = raw_hoi[4]
                    if raw_key_points is None or len(raw_key_points) != 51:
                        image_anno['pbox_lists'].append(est_part_boxes(hbox))
                    else:
                        key_points = np.array(raw_key_points).reshape((17, 3))
                        image_anno['pbox_lists'].append(gen_part_boxes(hbox, key_points, image_hw))

                    if pn == 0:
                        # positive - 0
                        image_anno['bin_classes'].append(0)
                    else:
                        # negative - 1
                        image_anno['bin_classes'].append(1)

            # list -> np.array
            if len(image_anno['hboxes']) == 0:
                image_anno['hboxes'] = np.zeros((0, 4))
                image_anno['oboxes'] = np.zeros((0, 4))
                image_anno['iboxes'] = np.zeros((0, 4))
                image_anno['pbox_lists'] = np.zeros((0, 6*4))
                image_anno['obj_classes'] = np.zeros(0)
                image_anno['bin_classes'] = np.zeros(0, 2)
                image_anno['hoi_classes'] = np.zeros((0, len(self.hoi_classes)))
                image_anno['hoi_masks'] = np.zeros((0, len(self.hoi_classes)))
                image_anno['vrb_classes'] = np.zeros((0, len(self.vrb_classes)))
                image_anno['vrb_masks'] = np.ones((0, len(self.vrb_classes)))
            else:
                image_anno['hboxes'] = np.array(image_anno['hboxes'])
                image_anno['oboxes'] = np.array(image_anno['oboxes'])
                image_anno['iboxes'] = np.array(image_anno['iboxes'])
                image_anno['obj_classes'] = np.array(image_anno['obj_classes'])
                image_anno['pbox_lists'] = np.array(image_anno['pbox_lists'])

                bin_classes = image_anno['bin_classes']
                image_anno['bin_classes'] = np.zeros((len(bin_classes), 2))
                for i, ins_class in enumerate(bin_classes):
                    image_anno['bin_classes'][i, ins_class] = 1

                hoi_classes = image_anno['hoi_classes']
                image_anno['hoi_classes'] = np.zeros((len(hoi_classes), len(self.hoi_classes)))
                for i, ins_hois in enumerate(hoi_classes):
                    for hoi_id in ins_hois:
                        image_anno['hoi_classes'][i, hoi_id] = 1

                hoi_intervals = image_anno['hoi_masks']
                image_anno['hoi_masks'] = np.zeros((len(image_anno['hboxes']), len(self.vrb_classes)))

                vrb_classes = image_anno['vrb_classes']
                image_anno['vrb_classes'] = np.zeros((len(vrb_classes), len(self.vrb_classes)))
                for i, ins_verbs in enumerate(vrb_classes):
                    for vrb_id in ins_verbs:
                        image_anno['vrb_classes'][i, vrb_id] = 1

                vrb_masks = image_anno['vrb_masks']
                image_anno['vrb_masks'] = np.array(vrb_masks)

        return all_annos

    def append_flipped_images(self):
        import copy
        num_images = len(self.roidb)
        widths = [self.roidb[i]['width'] for i in range(num_images)]
        for i in range(num_images):
            new_entry = copy.deepcopy(self.roidb[i])
            new_entry['flipped'] = True

            box_types = ['hboxes', 'oboxes', 'iboxes']
            for box_type in box_types:
                boxes = self.roidb[i][box_type].copy()
                oldx1 = boxes[:, 0].copy()
                oldx2 = boxes[:, 2].copy()
                boxes[:, 0] = widths[i] - oldx2 - 1
                boxes[:, 2] = widths[i] - oldx1 - 1
                assert (boxes[:, 2] >= boxes[:, 0]).all()
                new_entry[box_type] = boxes

            pbox_lists = self.roidb[i]['pbox_lists'].copy()
            pboxes = pbox_lists.reshape((-1, 4))
            oldx1 = pboxes[:, 0].copy()
            oldx2 = pboxes[:, 2].copy()
            pboxes[:, 0] = widths[i] - oldx2 - 1
            pboxes[:, 2] = widths[i] - oldx1 - 1
            assert (pboxes[:, 2] >= pboxes[:, 0]).all()
            pbox_lists = pboxes.reshape((-1, 6 * 4))
            new_entry['pbox_lists'] = pbox_lists

            self.roidb.append(new_entry)
        self._image_index = self._image_index * 2

    def _get_widths(self):
        all_widths = []
        for image_id in sorted(self._image_index):
            all_widths.append(self._all_image_info[image_id][0])     # image width
        return all_widths

