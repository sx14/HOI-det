# @CreateTime : 2021/3/24
# @Author : sunx

import os
import pickle
import numpy as np
import shutil
from collections import defaultdict
from utils import *


def load_category_id_to_name(data_root, element):
    cate_list_path = os.path.join(data_root, 'categories_list', '%s_categories.json' % element)
    cate_list = load_json(cate_list_path)
    id_to_name = {item['id']: item['name'] for item in cate_list}
    for id, name in id_to_name.items():
        if '/' in name:
            id_to_name[id] = name.replace('/', '_')
        if ' ' in name:
            id_to_name[id] = name.replace(' ', '_')
    return id_to_name


def gen_positive_instances(data_root, prepare_root, split='train'):
    # load relation annotation
    rlt_path = os.path.join(data_root, 'relations_%s.json' % split)
    rlt_anno = load_json(rlt_path)

    # load instance annotation
    ins_info_path = os.path.join(prepare_root, 'instances_%s.json' % split)
    ins_info = load_json(ins_info_path)

    # merge relation instances
    merged_rlt_anno = {}
    for im_anno in rlt_anno:
        sbj_obj_to_pres = {}
        im_id = im_anno['name'].split('.')[0]
        merged_rlt_anno[im_id] = sbj_obj_to_pres
        im_rlts = im_anno['relations']
        for rlt in im_rlts:
            sbj_id = rlt['subject']
            obj_id = rlt['object']
            pre_id = rlt['relation']

            if '%d_%d' % (sbj_id, obj_id) not in sbj_obj_to_pres:
                sbj_obj_to_pres['%d_%d' % (sbj_id, obj_id)] = {pre_id}
            else:
                sbj_obj_to_pres['%d_%d' % (sbj_id, obj_id)].add(pre_id)

    # generate hico-style annotation
    rlt_anno_hico = {}
    for im_id, im_rlts in merged_rlt_anno.items():
        im_insts = ins_info[im_id]
        im_rlts_hico = []
        rlt_anno_hico[im_id] = im_rlts_hico
        for sbj_obj, pres in im_rlts.items():
            sbj_id, obj_id = sbj_obj.split('_')
            sbj_id, obj_id = int(sbj_id), int(obj_id)

            sbj = im_insts[str(sbj_id)]
            obj = im_insts[str(obj_id)]

            if sbj_id == obj_id:
                continue

            sbj_cate_id = sbj['category']
            obj_cate_id = obj['category']
            if sbj_cate_id != 1:
                continue

            sbj_box = [sbj['box']['xmin'],
                       sbj['box']['ymin'],
                       sbj['box']['xmax'],
                       sbj['box']['ymax']]

            obj_box = [obj['box']['xmin'],
                       obj['box']['ymin'],
                       obj['box']['xmax'],
                       obj['box']['ymax'],]

            rlt_inst = [
                0,
                pres,
                sbj_box,
                obj_box,
                obj_cate_id,
                sbj_id,
                obj_id,
                0
            ]
            im_rlts_hico.append(rlt_inst)

    rlt_anno_hico_path = os.path.join(prepare_root, '%s_GT_PIC.pkl' % split)
    with open(rlt_anno_hico_path, 'wb') as f:
        pickle.dump(rlt_anno_hico, f)
    print('positive instances saved at: %s' % rlt_anno_hico_path)



def gen_negative_instances(data_root, prepare_root, split='train'):
    # load positive relation instances
    rlt_anno_hico_path = os.path.join(prepare_root, '%s_GT_PIC.pkl' % split)
    with open(rlt_anno_hico_path, 'rb') as f:
        rlt_anno = pickle.load(f)

    # load object instance annotation
    ins_info_path = os.path.join(prepare_root, 'instances_%s.json' % split)
    ins_info = load_json(ins_info_path)

    neg_rlt_anno_hico = {}
    for im_id, im_rlts in rlt_anno.items():
        pos_pairs = set()
        for rlt in im_rlts:
            sbj_id, obj_id = rlt[5], rlt[6]
            pos_pairs.add('%s_%s' % (sbj_id, obj_id))

        im_insts = ins_info[im_id]
        im_neg_rlts = []

        neg_rlt_anno_hico[im_id] = im_neg_rlts
        for sbj_id, sbj_inst in im_insts.items():
            # subject must be human
            if sbj_inst['category'] != 1:
                continue

            sbj_box = [sbj_inst['box']['xmin'],
                       sbj_inst['box']['ymin'],
                       sbj_inst['box']['xmax'],
                       sbj_inst['box']['ymax']]

            for obj_id, obj_inst in im_insts.items():
                if sbj_id == obj_id:
                    continue

                obj_box = [obj_inst['box']['xmin'],
                           obj_inst['box']['ymin'],
                           obj_inst['box']['xmax'],
                           obj_inst['box']['ymax'], ]

                sbj_obj = '%s_%s' % (sbj_id, obj_id)
                if sbj_obj in pos_pairs:
                    # exist relation
                    continue
                else:
                    im_neg_rlts.append([
                        0,
                        [0],
                        sbj_box,
                        obj_box,
                        obj_inst['category'],
                        sbj_id,
                        obj_id,
                        0
                    ])

    neg_rlt_anno_hico_path = os.path.join(prepare_root, '%s_NG_PIC.pkl' % split)
    with open(neg_rlt_anno_hico_path, 'wb') as f:
        pickle.dump(neg_rlt_anno_hico, f)
    print('negative instances saved at: %s' % neg_rlt_anno_hico_path)


def object_predicate_pair_distribution(data_root, prepare_root, split='train'):
    # category id to name
    pre_id_to_name = load_category_id_to_name(data_root, 'relation')
    obj_id_to_name = load_category_id_to_name(data_root, 'label')

    # load relation annotation
    rlt_path = os.path.join(data_root, 'relations_%s.json' % split)
    rlt_anno = load_json(rlt_path)

    # instance id to category
    ins_info_path = os.path.join(prepare_root, 'instances_%s.json' % split)
    ins_info = load_json(ins_info_path)

    # collect
    relation_num = 0
    self_relation_num = 0
    invalid_relation = []
    obj_to_pre_dist = np.zeros((len(obj_id_to_name), len(pre_id_to_name)))
    obj_to_pres_tmp = defaultdict(set)
    for im_rlt_anno in rlt_anno:
        im_rlts = im_rlt_anno['relations']
        im_id = im_rlt_anno['name'].split('.')[0]
        im_objs = ins_info[im_id]
        for i, rlt in enumerate(im_rlts):
            relation_num += 1
            sbj_ins_id = str(rlt['subject'])
            obj_ins_id = str(rlt['object'])
            sbj_cate = im_objs[sbj_ins_id]['category']
            obj_cate = im_objs[obj_ins_id]['category']
            rlt_cate = rlt['relation']

            if sbj_ins_id == obj_ins_id:
                self_relation_num += 1

            if sbj_cate != 1:
                invalid_relation.append('%s_%d' % (im_id, i))
                continue

            # update
            obj_to_pre_dist[obj_cate][rlt_cate] += 1
            obj_to_pres_tmp[obj_cate].add(rlt_cate)

    # count -> distribution
    for obj_cate in obj_id_to_name:
        obj_to_pre_dist[obj_cate] = obj_to_pre_dist[obj_cate] / max(obj_to_pre_dist[obj_cate].sum(), 1)

    # {obj: set(pre)} -> {obj: list(pre)}
    obj_to_pres = {}
    for obj_cate in obj_id_to_name:
        if obj_cate not in obj_to_pres_tmp:
            obj_to_pres[obj_cate] = []
        else:
            obj_to_pres[obj_cate] = sorted(list(obj_to_pres_tmp[obj_cate]))

    print('invalid relation num: %d/%d' % (len(invalid_relation), relation_num))
    print('self    relation num: %d/%d' % (self_relation_num, relation_num))

    # 保存obj-to-pre分布
    obj_to_pre_dist_path = os.path.join(prepare_root, 'obj_to_vrb_dist.pkl')
    with open(obj_to_pre_dist_path, 'wb') as f:
        import pickle
        pickle.dump(obj_to_pre_dist, f)

    # 保存obj-to-pres
    obj_to_pres_path = os.path.join(prepare_root, 'object_to_relations.json')
    with open(obj_to_pres_path, 'w') as f:
        import json
        json.dump(obj_to_pres, f)


if __name__ == '__main__':
    data_root = 'data/pic'
    prepare_root = 'data/pic/mlcnet_data'
    object_predicate_pair_distribution(data_root, prepare_root)
    gen_positive_instances(data_root, prepare_root)
    gen_negative_instances(data_root, prepare_root)
