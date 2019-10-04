import os
import random
import pickle
from math import log, e

import gensim
import scipy.io as sio
import numpy as np


class HOIClass:
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


def object_class_mapping(hoi_classes, hoi_obj_classes):
    hoi_range = [(161, 170), (11, 24), (66, 76), (147, 160), (1, 10), (55, 65), (187, 194), (568, 576), (32, 46),
                 (563, 567), (326, 330), (503, 506), (415, 418), (244, 247), (25, 31), (77, 86), (112, 129), (130, 146),
                 (175, 186), (97, 107), (314, 325), (236, 239), (596, 600), (343, 348), (209, 214), (577, 584),
                 (353, 356), (539, 546), (507, 516), (337, 342), (464, 474), (475, 483), (489, 502), (369, 376),
                 (225, 232), (233, 235), (454, 463), (517, 528), (534, 538), (47, 54), (589, 595), (296, 305),
                 (331, 336), (377, 383), (484, 488), (253, 257), (215, 224), (199, 208), (439, 445), (398, 407),
                 (258, 264), (274, 283), (357, 363), (419, 429), (306, 313), (265, 273), (87, 92), (93, 96), (171, 174),
                 (240, 243), (108, 111), (551, 558), (195, 198), (384, 389), (394, 397), (435, 438), (364, 368),
                 (284, 290), (390, 393), (408, 414), (547, 550), (450, 453), (430, 434), (248, 252), (291, 295),
                 (585, 588), (446, 449), (529, 533), (349, 352), (559, 562)]
    hoi_obj2ind = dict(zip(hoi_obj_classes, xrange(len(hoi_obj_classes))))
    det_obj_classes = [hoi_classes[int[0] - 1].object_name() for int in hoi_range]
    det_obj2hoi_obj = {}
    for i in range(len(det_obj_classes)):
        obj_name = det_obj_classes[i]
        det_obj_ind = i+1
        hoi_obj_ind = hoi_obj2ind[obj_name]
        det_obj2hoi_obj[det_obj_ind] = hoi_obj_ind
    return det_obj2hoi_obj



def load_object_word2vec(object_classes, w2v_path, save_dir):
    print('Loading obj2vec ...')

    obj2vec_path = os.path.join(save_dir, 'hico_obj2vec.pkl')
    if os.path.exists(obj2vec_path):
        with open(obj2vec_path) as f:
            obj2vec = pickle.load(f)
        return obj2vec

    # load pretrained word2vec
    model = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=True)
    obj2vec = np.zeros((len(object_classes), 300))
    for i, obj_class in enumerate(object_classes):
        obj_class_clean = obj_class

        if obj_class == 'dining_table':
            obj_class_clean = 'table'
        elif obj_class == 'baseball_bat':
            obj_class_clean = 'bat'
        elif obj_class == 'baseball_glove':
            obj_class_clean = 'glove'
        elif obj_class == 'hair_drier':
            obj_class_clean = 'drier'
        elif obj_class == 'potted_plant':
            obj_class_clean = 'plant'
        elif obj_class == 'cell_phone':
            obj_class_clean = 'phone'
        elif obj_class == 'fire_hydrant':
            obj_class_clean = 'hydrant'
        elif obj_class == 'hot_dog':
            obj_class_clean = 'bread'
        elif obj_class == 'parking_meter':
            obj_class_clean = 'meter'
        elif obj_class == 'sports_ball':
            obj_class_clean = 'ball'
        elif obj_class == 'stop_sign':
            obj_class_clean = 'sign'
        elif obj_class == 'teddy_bear':
            obj_class_clean = 'toy'
        elif obj_class == 'tennis_racket':
            obj_class_clean = 'racket'
        elif obj_class == 'traffic_light':
            obj_class_clean = 'light'
        elif obj_class == 'wine_glass':
            obj_class_clean = 'glass'

        vec = model[obj_class_clean]
        if vec is None or len(vec) == 0 or np.sum(vec) == 0:
            print('[WARNING] %s' % obj_class)
        obj2vec[i] = vec

    with open(obj2vec_path, 'wb') as f:
        pickle.dump(obj2vec, f)
    return obj2vec


def load_hoi_classes(hoi_class_path):
    hoi_cls_list = []
    obj_cls_list = []
    vrb_cls_list = []
    with open(hoi_class_path) as f:
        mat_hoi_classes = pickle.load(f)
    for hoi_cls_id, hoi_cls in enumerate(mat_hoi_classes):
        obj_cls_name = hoi_cls.split(' ')[1]
        if obj_cls_name not in obj_cls_list:
            obj_cls_list.append(obj_cls_name)

        vrb_cls_name = hoi_cls.split(' ')[0]
        if vrb_cls_name not in vrb_cls_list:
            vrb_cls_list.append(vrb_cls_name)

        hoi_cls_list.append(HOIClass(obj_cls_name, vrb_cls_name, hoi_cls_id))

    hoi2int = [[] for _ in range(len(hoi_cls_list))]
    curr_hoi_stt = 0
    curr_obj = hoi_cls_list[0].object_name()
    for i in range(1, len(hoi_cls_list)):
        hoi = hoi_cls_list[i]
        if hoi.object_name() != curr_obj:
            # last interval ended
            curr_hoi_end = i - 1
            for j in range(curr_hoi_stt, curr_hoi_end + 1):
                hoi2int[j] = [curr_hoi_stt, curr_hoi_end]
            curr_hoi_stt = i
            curr_obj = hoi.object_name()
    curr_hoi_end = len(hoi_cls_list) - 1
    for j in range(curr_hoi_stt, curr_hoi_end + 1):
        hoi2int[j] = [curr_hoi_stt, curr_hoi_end]

    # obj2int = [[] for _ in range(len(obj_cls_list))]
    # curr_obj = hoi_cls_list[0].object_name()
    # curr_int_stt = 0
    # curr_obj_ind = 0
    # for i in range(1, len(hoi_cls_list)):
    #     obj = hoi_cls_list[i].object_name()
    #     if obj != curr_obj:
    #         curr_int_end = i - 1
    #         assert curr_obj == obj_cls_list[curr_obj_ind]
    #         obj2int[curr_obj_ind] = [curr_int_stt, curr_int_end]
    #         curr_int_stt = i
    #         curr_obj = obj
    #         curr_obj_ind += 1
    # obj2int[curr_obj_ind] = [curr_int_stt, len(hoi_cls_list) - 1]

    return hoi_cls_list, obj_cls_list, vrb_cls_list, hoi2int


def load_image_info(anno_path, save_dir, image_set='train'):
    print('Loading image set info ...')

    save_path = os.path.join(save_dir, 'hico_image_info_%s.pkl' % image_set)
    if os.path.exists(save_path):
        with open(save_path) as f:
            all_image_info = pickle.load(f)
        return all_image_info

    all_image_info = {}
    mat_anno_db = sio.loadmat(anno_path)
    mat_anno_db = mat_anno_db['bbox_%s' % image_set]

    for mat_anno in mat_anno_db[0, :]:
        image_id = mat_anno['filename'][0].split('.')[0]
        image_id = int(image_id[-8:])
        all_image_info[image_id] = [mat_anno['size']['width'][0, 0][0, 0], mat_anno['size']['height'][0, 0][0, 0]]

    with open(save_path, 'wb') as f:
        pickle.dump(all_image_info, f)
    return all_image_info


def extract_spatial_feature(box1, box2, image_size):
    img_w, img_h = image_size
    img_w = float(img_w)
    img_h = float(img_h)

    sbj_h = box1['ymax'] - box1['ymin'] + 1
    sbj_w = box1['xmax'] - box1['xmin'] + 1
    obj_h = box2['ymax'] - box2['ymin'] + 1
    obj_w = box2['xmax'] - box2['xmin'] + 1
    spatial_feat = [
        box1['xmin'] * 1.0 / img_w,
        box1['ymin'] * 1.0 / img_h,
        box1['xmax'] * 1.0 / img_w,
        box1['ymax'] * 1.0 / img_h,
        (sbj_h * sbj_w * 1.0) / (img_h * img_w),
        box2['xmin'] * 1.0 / img_w,
        box2['ymin'] * 1.0 / img_h,
        box2['xmax'] * 1.0 / img_w,
        box2['ymax'] * 1.0 / img_h,
        (obj_h * obj_w * 1.0) / (img_h * img_w),
        (box1['xmin'] - box2['xmin'] + 1) / (obj_w * 1.0),
        (box1['ymin'] - box2['ymin'] + 1) / (obj_h * 1.0),
        log(sbj_w * 1.0 / obj_w, e),
        log(sbj_h * 1.0 / obj_h, e)]
    return spatial_feat


def prepare_hico(hico_root, save_dir):
    hoi_db_path = os.path.join(save_dir, 'hico_trainval_anno.pkl')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if os.path.exists(hoi_db_path):
        print('Loading annotations ...')
        with open(hoi_db_path) as f:
            hoi_db = pickle.load(f)
        return hoi_db

    image_info_path = os.path.join(hico_root, 'anno_bbox_full.mat')
    image_info = load_image_info(image_info_path, save_dir)

    hoi_class_path = os.path.join(hico_root, 'hoi_categories.pkl')
    hoi_cates, obj_cates, vrb_cates, _ = load_hoi_classes(hoi_class_path)
    obj2ind = dict(zip(obj_cates, xrange(len(obj_cates))))

    hoi_class_num = len(hoi_cates)
    obj2vec = load_object_word2vec(obj_cates, 'GoogleNews-vectors-negative300.bin', save_dir)

    print('Loading annotations ...')
    anno_gt_path = os.path.join(hico_root, 'train_GT_HICO_with_pose.pkl')
    anno_ng_path = os.path.join(hico_root, 'train_NG_HICO_with_pose.pkl')
    anno_gt = pickle.load(open(anno_gt_path))
    anno_ng = pickle.load(open(anno_ng_path))

    hboxes = []
    oboxes = []
    spa_feats = []
    hoi_classes = []
    bin_classes = []
    obj_classes = []
    skeletons = []

    print('Processing annotations ...')
    anno_gt_db = {}
    for hoi_ins_gt in anno_gt:
        image_id = hoi_ins_gt[0]
        if image_id in anno_gt_db:
            anno_gt_db[image_id].append(hoi_ins_gt)
        else:
            anno_gt_db[image_id] = [hoi_ins_gt]

    for image_id, img_pos_hois in anno_gt_db.items():

        image_size = image_info[image_id]
        if image_size[0] == 0 or image_size[1] == 0:
            print(image_id)

        if image_id in anno_ng and len(anno_ng[image_id]) > 0:
            img_neg_hois0 = anno_ng[image_id]
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

        for pn, hois in enumerate([img_pos_hois, img_neg_hois]):
            for raw_hoi in hois:
                hbox = raw_hoi[2]
                obox = raw_hoi[3]
                bin_class = pn  # pos: 0; neg: 1
                hoi_class_ids = raw_hoi[1]
                if isinstance(hoi_class_ids, int):
                    hoi_class_ids = [hoi_class_ids]

                obj_class = obj2ind[hoi_cates[hoi_class_ids[0]].object_name()]
                hoi_class = [0] * hoi_class_num
                if pn == 0:
                    skeleton = raw_hoi[5]
                else:
                    skeleton = raw_hoi[7]

                for id in hoi_class_ids:
                    hoi_class[id] = 1

                hbox_tmp = {
                    'xmin': float(hbox[0]),
                    'ymin': float(hbox[1]),
                    'xmax': float(hbox[2]),
                    'ymax': float(hbox[3]),
                }
                obox_tmp = {
                    'xmin': float(obox[0]),
                    'ymin': float(obox[1]),
                    'xmax': float(obox[2]),
                    'ymax': float(obox[3]),
                }
                spa_feat = extract_spatial_feature(hbox_tmp, obox_tmp, image_size)
                spa_feats.append(spa_feat)
                hboxes.append(hbox)
                oboxes.append(obox)
                obj_classes.append(obj_class)
                hoi_classes.append(hoi_class)
                bin_classes.append(bin_class)
                skeletons.append(skeleton)

    num_item = len(hboxes)
    num_train = int(num_item * 0.7)

    train_db = {
        'obj2vec': obj2vec,
        'hboxes': np.array(hboxes[:num_train]),
        'oboxes': np.array(oboxes[:num_train]),
        'spa_feats': np.array(spa_feats[:num_train]),
        'obj_classes': np.array(obj_classes[:num_train]),
        'hoi_classes': np.array(hoi_classes[:num_train]),
        'bin_classes': np.array(bin_classes[:num_train]),
        'skeletons': skeletons[:num_train]
    }

    val_db = {
        'obj2vec': obj2vec,
        'hboxes': np.array(hboxes[num_train:]),
        'oboxes': np.array(oboxes[num_train:]),
        'spa_feats': np.array(spa_feats[num_train:]),
        'obj_classes': np.array(obj_classes[num_train:]),
        'hoi_classes': np.array(hoi_classes[num_train:]),
        'bin_classes': np.array(bin_classes[num_train:]),
        'skeletons': skeletons[num_train:]
    }

    hoi_db = {
        'train': train_db,
        'val': val_db,
    }

    with open(hoi_db_path, 'wb') as f:
        pickle.dump(hoi_db, f)
    return hoi_db