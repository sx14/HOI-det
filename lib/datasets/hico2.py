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


class hoi_class:
    def __init__(self, object_name, verb_name, hoi_id):
        self._object_name = object_name
        self._verb_name = verb_name
        self._hoi_id = hoi_id

    def object_name(self):
        return self._object_name

    def verb_name(self):
        return self._verb_name

    def hoi_name(self):
        return self._verb_name + ' ' + self._object_name


class hico2(imdb):

    @staticmethod
    def load_obj2vec(data_path):
        obj2vec_path = os.path.join(data_path, 'obj2vec.pkl')
        with open(obj2vec_path) as f:
            obj2vec = pickle.load(f)
        return obj2vec

    @staticmethod
    def load_hoi_classes(data_path):
        hoi_cls_list = []
        obj_cls_list = []
        obj_cls2ind = {}
        vrb_cls_list = []
        vrb_cls2ind = {}
        obj2int = {}
        hoi2vrb = {}
        vrb2hoi = {}

        with open(os.path.join(data_path, 'hoi_categories.pkl')) as f:
            mat_hoi_classes = pickle.load(f)

        obj_id = 0
        vrb_id = 0
        for hoi_cls_id, hoi_cls in enumerate(mat_hoi_classes):
            obj_cls_name = hoi_cls.split(' ')[1]
            if obj_cls_name not in obj_cls2ind:
                obj_cls_list.append(obj_cls_name)
                obj_cls2ind[obj_cls_name] = obj_id
                obj_id += 1
                obj2int[obj_cls_name] = [hoi_cls_id, hoi_cls_id]
            else:
                obj2int[obj_cls_name][1] = hoi_cls_id

            vrb_cls_name = hoi_cls.split(' ')[0]
            if vrb_cls_name not in vrb_cls2ind:
                vrb_cls_list.append(vrb_cls_name)
                vrb_cls2ind[vrb_cls_name] = vrb_id
                vrb2hoi[vrb_id] = [hoi_cls_id]
                vrb_id += 1
            else:
                vrb2hoi[vrb_cls2ind[vrb_cls_name]].append(hoi_cls_id)

            hoi2vrb[hoi_cls_id] = vrb_cls2ind[vrb_cls_name]
            hoi_cls_list.append(hoi_class(obj_cls_name, vrb_cls_name, hoi_cls_id))
        return hoi_cls_list, obj_cls_list, vrb_cls_list, obj2int, hoi2vrb, vrb2hoi

    def __init__(self, image_set, version):
        imdb.__init__(self, 'hico2_' + version + '_' + image_set)
        self._version = version
        self._image_set = image_set
        self._data_path = self._get_default_path()

        self.hoi_classes, self.obj_classes, self.vrb_classes, self.obj2int, self.hoi2vrb, _ = self.load_hoi_classes(self._data_path)
        # self._classes = [hoi_class.hoi_name() for hoi_class in self.hoi_classes]
        self._classes = self.vrb_classes
        self.hoi_class2ind = dict(zip([hoi_class.hoi_name() for hoi_class in self.hoi_classes], xrange(len(self.hoi_classes))))
        self.obj_class2ind = dict(zip(self.obj_classes, xrange(len(self.obj_classes))))
        self.verb_class2ind = dict(zip(self.vrb_classes, xrange(len(self.vrb_classes))))

        self._class_to_ind = dict(zip(self._classes, xrange(len(self._classes))))
        self._image_ext = '.jpg'
        self._all_image_info = self._load_image_set_info()
        self._obj2vec = None
        self._image_index = None
        # Default to roidb handler
        # self._roidb_handler = self.selective_search_roidb
        self._roidb_handler = self.gt_roidb

    @property
    def obj2vec(self):
        if self._obj2vec is None:
            obj2vec_path = os.path.join(self._data_path, 'obj2vec.pkl')
            with open(obj2vec_path) as f:
                obj2vec = pickle.load(f)
                self._obj2vec = obj2vec
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
        image_path = os.path.join(self._data_path, 'images', self._image_set + '2015',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_info(self):
        print('Loading image set info ...')
        all_image_info = {}

        mat_anno_db = sio.loadmat(os.path.join(self._data_path, 'anno_bbox_%s.mat' % self._version))
        mat_anno_db = mat_anno_db['bbox_' + self._image_set]

        for mat_anno in mat_anno_db[0, :]:
            image_id = mat_anno['filename'][0].split('.')[0]
            all_image_info[image_id] = [mat_anno['size']['width'][0, 0][0, 0], mat_anno['size']['height'][0, 0][0, 0]]

        return all_image_info

    def _get_default_path(self):
        return os.path.join(cfg.DATA_DIR, 'hico')

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
                   (boxA[2] - boxA[0] + 1.) *
                   (boxA[3] - boxA[1] + 1.) - inters)

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

            hbox = raw_hoi[2]
            obox = raw_hoi[3]
            aug_hboxes = self.augment_box(hbox, im_hw)
            aug_oboxes = self.augment_box(obox, im_hw)

            aug_hboxes = aug_hboxes[:min(len(aug_hboxes), len(aug_oboxes))]
            aug_oboxes = aug_oboxes[:min(len(aug_hboxes), len(aug_oboxes))]

            for i in range(aug_hboxes.shape[0]):
                aug_cls_ids = raw_hoi[1]
                aug_hbox = aug_hboxes[i]
                aug_obox = aug_oboxes[i]
                new_hois.append([0,             # stub
                                 aug_cls_ids,
                                 aug_hbox,
                                 aug_obox,
                                 0,             # stub
                                 0,             # stub
                                 0,             # stub
                                 raw_hoi[5]])
        return new_hois

    def _load_all_annotations(self):
        all_annos = {}

        print('Loading annotations ...')
        anno_ng_db = pickle.load(open(os.path.join(self._data_path, '%s_NG_HICO_with_pose.pkl' % self._image_set)))
        anno_gt_tmp = pickle.load(open(os.path.join(self._data_path, '%s_GT_HICO_with_pose.pkl' % self._image_set)))

        print('Processing annotations ...')
        anno_gt_db = {}
        for hoi_ins_gt in anno_gt_tmp:
            image_id = hoi_ins_gt[0]
            if image_id in anno_gt_db:
                anno_gt_db[image_id].append(hoi_ins_gt)
            else:
                anno_gt_db[image_id] = [hoi_ins_gt]

        image_id_template = 'HICO_train2015_%s'
        for image_id, img_pos_hois in anno_gt_db.items():
            image_name = image_id_template % str(image_id).zfill(8)

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

                    hoi_class_ids = raw_hoi[1]
                    if isinstance(hoi_class_ids, int):
                        hoi_class_ids = [hoi_class_ids]
                    hoi_classes = [self.hoi_classes[class_id] for class_id in hoi_class_ids]
                    obj_class_name = hoi_classes[0].object_name()
                    obj_class_id = self.obj_class2ind[obj_class_name]

                    hbox = raw_hoi[2]
                    obox = raw_hoi[3]
                    ibox = [min(hbox[0], obox[0]), min(hbox[1], obox[1]),
                            max(hbox[2], obox[2]), max(hbox[3], obox[3])]
                    image_anno['hboxes'].append(hbox)
                    image_anno['oboxes'].append(obox)
                    image_anno['iboxes'].append(ibox)
                    image_anno['hoi_classes'].append(hoi_class_ids)
                    image_anno['vrb_classes'].append([self.hoi2vrb[hoi_id] for hoi_id in hoi_class_ids])
                    image_anno['obj_classes'].append(obj_class_id)
                    image_anno['hoi_masks'].append(self.obj2int[obj_class_name])
                    image_anno['vrb_masks'].append([self.hoi2vrb[hoi]
                                                    for hoi in range(self.obj2int[obj_class_name][0],
                                                                     self.obj2int[obj_class_name][1]+1)])

                    raw_key_points = raw_hoi[7]
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
                image_anno['vrb_classes'] = np.zeros((0, len(self.vrb_classes)))
                image_anno['hoi_masks'] = np.ones((0, len(self.hoi_classes)))
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
                for i, ins_classes in enumerate(hoi_classes):
                    for cls in ins_classes:
                        image_anno['hoi_classes'][i, cls] = 1

                hoi_intervals = image_anno['hoi_masks']
                image_anno['hoi_masks'] = np.zeros((len(hoi_intervals), len(self.hoi_classes)))
                for i, ins_interval in enumerate(hoi_intervals):
                    image_anno['hoi_masks'][i, ins_interval[0]:ins_interval[1]+1] = 1

                vrb_classes = image_anno['vrb_classes']
                image_anno['vrb_classes'] = np.zeros((len(vrb_classes), len(self.vrb_classes)))
                for i, ins_verbs in enumerate(vrb_classes):
                    for vrb_id in ins_verbs:
                        image_anno['vrb_classes'][i, vrb_id] = 1

                vrb_masks = image_anno['vrb_masks']
                image_anno['vrb_masks'] = np.zeros((len(vrb_masks), len(self.vrb_classes)))
                for i, ins_verbs in enumerate(vrb_masks):
                    for vrb_id in ins_verbs:
                        image_anno['vrb_masks'][i, vrb_id] = 1

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

            self.roidb.append(new_entry)
        self._image_index = self._image_index * 2

    def _get_widths(self):
        mat_anno_db = sio.loadmat(os.path.join(self._data_path, 'anno_bbox_%s.mat' % self._version))
        mat_anno_db = mat_anno_db['bbox_' + self._image_set]
        all_widths = []
        for image_id, image_anno in enumerate(mat_anno_db[0, :]):
            image_name = image_anno['filename'][0].split('.')[0]
            assert self._image_index[image_id] == image_name
            all_widths.append(image_anno['size']['width'][0, 0][0, 0])     # image width
        return all_widths


    def _get_voc_results_file_template(self):
        # hico/results/<hoi_det_test_walk_dog.txt
        filename = 'hoi_det_' + self._image_set + '_{:s}.txt'
        filedir = os.path.join(self._data_path, 'results')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            print('Writing {} HICO results file'.format(cls))
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir='output'):
        annopath = os.path.join(
            self._devkit_path,
            'VOC' + self._version,
            'Annotations',
            '{:s}.xml')
        imagesetfile = os.path.join(
            self._devkit_path,
            'VOC' + self._version,
            'ImageSets',
            'Main',
            self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self._version) < 2010 else False
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def _do_matlab_eval(self, output_dir='output'):
        print('-----------------------------------------------------')
        print('Computing results with the official MATLAB eval code.')
        print('-----------------------------------------------------')
        path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
            .format(self._devkit_path, self._get_comp_id(),
                    self._image_set, output_dir)
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir):
        self._write_voc_results_file(all_boxes)
        self._do_python_eval(output_dir)
        if self.config['matlab_eval']:
            self._do_matlab_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                os.remove(filename)

