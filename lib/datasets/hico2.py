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
import glob
import uuid
import scipy.io as sio
import xml.etree.ElementTree as ET
import pickle
from .imdb import imdb
from .imdb import ROOT_DIR
from . import ds_utils
from .voc_eval import voc_eval

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

    def _load_hoi_classes(self):
        hoi_cls_list = []
        object_cls_list = []
        verb_cls_list = []
        with open(os.path.join(self._data_path, 'hoi_categories.pkl')) as f:
            mat_hoi_classes = pickle.load(f)
        for hoi_cls_id, hoi_cls in enumerate(mat_hoi_classes):
            object_cls_name = hoi_cls.split(' ')[1]
            if object_cls_name not in object_cls_list:
                object_cls_list.append(object_cls_name)

            verb_cls_name = hoi_cls.split(' ')[0]
            if verb_cls_name not in verb_cls_list:
                verb_cls_list.append(verb_cls_name)

            hoi_cls_list.append(hoi_class(object_cls_name, verb_cls_name, hoi_cls_id))
        return hoi_cls_list, object_cls_list, verb_cls_list

    def __init__(self, image_set, version):
        imdb.__init__(self, 'hico2_' + version + '_' + image_set)
        self._version = version
        self._image_set = image_set
        self._data_path = self._get_default_path()

        self.hoi_classes, self.object_classes, self.verb_classes = self._load_hoi_classes()
        self._classes = [hoi_class.hoi_name() for hoi_class in self.hoi_classes]
        self.hoi_class2ind = dict(zip(self._classes, xrange(self.num_classes)))
        self.obj_class2ind = dict(zip(self.object_classes, xrange(len(self.object_classes))))
        self.verb_class2ind = dict(zip(self.verb_classes, xrange(len(self.verb_classes))))

        self._class_to_ind = dict(zip(self._classes, xrange(len(self._classes))))
        self._image_ext = '.jpg'
        self._all_image_info = self._load_image_set_info()
        self._image_index = None
        # Default to roidb handler
        # self._roidb_handler = self.selective_search_roidb
        self._roidb_handler = self.gt_roidb

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

    def _load_all_annotations(self):
        all_annos = {}

        print('Loading annotations ...')
        anno_ng_db = pickle.load(open(os.path.join(self._data_path, '%s_NG_HICO.pkl' % self._image_set)))
        anno_gt_tmp = pickle.load(open(os.path.join(self._data_path, '%s_GT_HICO.pkl' % self._image_set)))

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
                assert len(img_neg_hois) == len(img_pos_hois)
            else:
                img_neg_hois = []

            # boxes: x1, y1, x2, y2
            image_anno = {'hboxes': [],
                          'oboxes': [],
                          'iboxes': [],
                          'hoi_classes': [],
                          'obj_classes': [],
                          'bin_classes': [],
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
                    obj_class_id = self.obj_class2ind[hoi_classes[0].object_name()]

                    hbox = raw_hoi[2]
                    obox = raw_hoi[3]
                    ibox = [min(hbox[0], obox[0]), min(hbox[1], obox[1]),
                            max(hbox[2], obox[2]), max(hbox[3], obox[3])]
                    image_anno['hboxes'].append(hbox)
                    image_anno['oboxes'].append(obox)
                    image_anno['iboxes'].append(ibox)
                    image_anno['hoi_classes'].append(hoi_class_ids)
                    image_anno['obj_classes'].append(obj_class_id)
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
                image_anno['obj_classes'] = np.zeros(0)
                image_anno['bin_classes'] = np.zeros(0, 2)
                image_anno['hoi_classes'] = np.zeros((0, self.num_classes))
            else:
                image_anno['hboxes'] = np.array(image_anno['hboxes'])
                image_anno['oboxes'] = np.array(image_anno['oboxes'])
                image_anno['iboxes'] = np.array(image_anno['iboxes'])
                image_anno['obj_classes'] = np.array(image_anno['obj_classes'])

                bin_classes = image_anno['bin_classes']
                image_anno['bin_classes'] = np.zeros((len(bin_classes), 2))
                for i, ins_class in enumerate(bin_classes):
                    image_anno['bin_classes'][i, ins_class] = 1

                hoi_classes = image_anno['hoi_classes']
                image_anno['hoi_classes'] = np.zeros((len(hoi_classes), self.num_classes))
                for i, ins_classes in enumerate(hoi_classes):
                    for cls in ins_classes:
                        image_anno['hoi_classes'][i, cls] = 1
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

