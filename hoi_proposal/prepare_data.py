# @CreateTime : 2021/5/18
# @Author : sunx
import os
import json
import numpy as np
from math import log2
import pickle
from my_utils import *
from gen_relation_anno import load_category_id_to_name




def human_object_relative(hbox, obox):
    hx1, hy1, hx2, hy2 = hbox
    hxc = (hx1 + hx2) / 2.0
    hyc = (hy1 + hy2) / 2.0
    hw = hx2 - hx1 + 1.0
    hh = hy2 - hy1 + 1.0
    ox1, oy1, ox2, oy2 = obox
    oxc = (ox1 + ox2) / 2.0
    oyc = (oy1 + oy2) / 2.0
    ow = ox2 - ox1 + 1.0
    oh = oy2 - oy1 + 1.0
    feat = [(hxc - oxc) / hw,
            (hyc - oyc) / hh,
            (hx1 - oxc) / hw,
            (hy1 - oyc) / hh,
            log2(hw / ow),
            log2(hh / oh)]
    return feat


def keypoint_object_relative(kpt, hbox, obox):
    hx1, hy1, hx2, hy2 = hbox
    hxc = (hx1 + hx2) / 2.0
    hyc = (hy1 + hy2) / 2.0
    hw = hx2 - hx1 + 1.0
    hh = hy2 - hy1 + 1.0
    kpt_x, kpt_y, kpt_scr = kpt
    ox1, oy1, ox2, oy2 = obox
    oxc = (ox1 + ox2) / 2.0
    oyc = (oy1 + oy2) / 2.0
    ow = ox2 - ox1 + 1.0
    oh = oy2 - oy1 + 1.0
    feat = [(kpt_x - oxc) / hw,
            (kpt_y - oyc) / hh,
            (kpt_x - ox1) / hw,
            (kpt_y - oy1) / hh,
            kpt_scr]
    return feat


def prepare_feature(data_root, prepare_root, split='train'):
    pos_anno_path = os.path.join(prepare_root, '%s_GT_PIC_with_pose.pkl' % split)
    neg_anno_path = os.path.join(prepare_root, '%s_NG_PIC_with_pose.pkl' % split)
    vrb_classes = load_category_id_to_name(data_root, 'relation')
    obj_classes = load_category_id_to_name(data_root, 'object')
    with open(pos_anno_path, 'rb') as f:
        pos_anno = pickle.load(f)
    with open(neg_anno_path, 'rb') as f:
        neg_anno = pickle.load(f)

    features = []
    vrb_labels = []
    bin_labels = []
    for image_id in pos_anno:
        im_pos_anno = pos_anno[image_id]
        im_neg_anno = neg_anno[image_id]

        for pos_neg, annos in enumerate([im_pos_anno, im_neg_anno]):
            for anno in annos:
                vrb_cates = anno[1]
                obj_cate = anno[4]
                sbox = anno[2]
                obox = anno[3]
                kpts = anno[7][0]

                ho_feat = human_object_relative(sbox, obox)
                ko_feats = []
                for k in range(len(kpts['x'])):
                    x = kpts['x'][k]
                    y = kpts['y'][k]
                    s = kpts['score'][k]
                    kpt = [x, y, s]
                    ko_feat = keypoint_object_relative(kpt, sbox, obox)
                    ko_feats.append(ko_feat)

                obj_cate_vec = [0] * len(obj_classes)
                obj_cate_vec[obj_cate] = 1

                feat = ho_feat + obj_cate_vec
                for f in ko_feats:
                    feat += f

                vrb_cate_vec = [0] * len(vrb_classes)
                for vrb_cate in vrb_cates:
                    vrb_cate_vec[vrb_cate] = 1

                features.append(feat)
                vrb_labels.append(vrb_cate_vec)
                bin_labels.append(1 if pos_neg == 0 else 0)

    features = np.array(features)
    vrb_labels = np.array(vrb_labels)
    bin_labels = np.array(bin_labels)
    bin_labels = bin_labels[:, np.newaxis]
    res = {'feature': features,
           'vrb_labels': vrb_labels,
           'bin_labels': bin_labels}
    print(features.shape)
    print(vrb_labels.shape)
    print(bin_labels.shape)
    save_path = os.path.join(prepare_root, 'feature_label_%s.pkl' % split)
    with open(save_path, 'wb') as f:
        pickle.dump(res, f)
    print('saved at: %s' % save_path)


data_root = 'data/hoia'
prepare_root = 'data/hoia/proposal'
prepare_feature(data_root, prepare_root, 'train')
prepare_feature(data_root, prepare_root, 'test')