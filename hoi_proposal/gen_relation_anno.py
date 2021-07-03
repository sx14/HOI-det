# @CreateTime : 2021/3/24
# @Author : sunx

import os
import pickle
import numpy as np
import shutil
from collections import defaultdict
from my_utils import *

def load_category_id_to_name(data_root, element):
    cate_list_path = os.path.join(data_root, '%s.txt' % element)

    with open(cate_list_path, 'r') as f:
        cate_list = f.readlines()
        cate_list = [obj_cate.strip() for obj_cate in cate_list]
        cate_list = ['none'] + cate_list

    id_to_name = {i: cate for i, cate in enumerate(cate_list)}
    return id_to_name


def gen_positive_instances(data_root, prepare_root, split='train'):
    # load relation annotation
    rlt_path = os.path.join(data_root, '%s_hoia_with_pose.json' % split)
    rlt_anno = load_json(rlt_path)

    # merge relation instances
    merged_rlt_anno = {}
    merged_obj_anno = {}
    bad_sbj_cnt = 0
    all_sbj_cnt = 0
    # collect object instances
    for im_anno in rlt_anno:
        sbj_obj_to_pres = defaultdict(set)
        obj_id_to_obj = {}
        im_id = im_anno['file_name'].split('.')[0]
        merged_rlt_anno[im_id] = sbj_obj_to_pres
        merged_obj_anno[im_id] = obj_id_to_obj
        im_rlts = im_anno['hoi_annotation']
        im_objs = im_anno['annotations']

        for obj in im_objs:
            if int(obj['category_id']) == 1:
                all_sbj_cnt += 1
                if 'pose_keypoints' not in obj:
                    bad_sbj_cnt += 1

        merged_obj_anno[im_id] = im_objs
        for rlt in im_rlts:
            sbj_id = rlt['subject_id']
            obj_id = rlt['object_id']
            pre = rlt['category_id']
            sbj_obj_to_pres['%d_%d' % (sbj_id, obj_id)].add(pre)

    # generate hico-style annotation
    bad_hoi_cnt = 0
    pos_hoi_cnt = 0
    rlt_anno_hico = {}
    for im_id, im_rlts in merged_rlt_anno.items():
        im_insts = merged_obj_anno[im_id]
        im_rlts_hico = []
        rlt_anno_hico[im_id] = im_rlts_hico
        for sbj_obj, pres in im_rlts.items():
            sbj_id, obj_id = sbj_obj.split('_')
            sbj_id, obj_id = int(sbj_id), int(obj_id)

            if sbj_id == obj_id or sbj_id >= len(im_insts) or obj_id >= len(im_insts):
                bad_hoi_cnt += 1
                continue

            sbj = im_insts[sbj_id]
            obj = im_insts[obj_id]

            sbj_cate_id = int(sbj['category_id'])
            obj_cate_id = int(obj['category_id'])

            sbj_box = sbj['bbox']
            obj_box = obj['bbox']

            if sbj_cate_id != 1 or 'pose_keypoints' not in sbj:
                bad_hoi_cnt += 1
                continue

            rlt_inst = [
                0,
                pres,
                sbj_box,
                obj_box,
                obj_cate_id,
                sbj_id,
                obj_id,
                sbj['pose_keypoints']
            ]
            im_rlts_hico.append(rlt_inst)
            pos_hoi_cnt += 1

    print('bad sbj count: %d/%d' % (bad_sbj_cnt, all_sbj_cnt))
    print('bad pos-hoi count: %d/%d' % (bad_hoi_cnt, pos_hoi_cnt))
    rlt_anno_hico_path = os.path.join(prepare_root, '%s_GT_PIC_with_pose.pkl' % split)
    with open(rlt_anno_hico_path, 'wb') as f:
        pickle.dump(rlt_anno_hico, f)
    print('positive instances saved at: %s' % rlt_anno_hico_path)


def gen_negative_instances(data_root, prepare_root, split='train'):
    # load positive relation instances
    rlt_anno_hico_path = os.path.join(prepare_root, '%s_GT_PIC_with_pose.pkl' % split)
    with open(rlt_anno_hico_path, 'rb') as f:
        rlt_anno = pickle.load(f)

    # collect object instances
    org_rlt_path = os.path.join(data_root, '%s_hoia_with_pose.json' % split)
    org_rlt_anno = load_json(org_rlt_path)
    ins_info = {}
    for im_anno in org_rlt_anno:
        im_id = im_anno['file_name'].split('.')[0]
        im_objs = im_anno['annotations']
        ins_info[im_id] = im_objs

    neg_rlt_anno_hico = {}
    neg_hoi_cnt = 0
    for im_id, im_rlts in rlt_anno.items():
        pos_pairs = set()
        for rlt in im_rlts:
            sbj_id, obj_id = rlt[5], rlt[6]
            pos_pairs.add('%s_%s' % (sbj_id, obj_id))

        im_insts = ins_info[im_id]
        im_neg_rlts = []
        neg_rlt_anno_hico[im_id] = im_neg_rlts
        for sbj_id, sbj_inst in enumerate(im_insts):
            # subject must be human
            sbj_cate_id = int(sbj_inst['category_id'])
            sbj_box = sbj_inst['bbox']

            if sbj_cate_id != 1 or 'pose_keypoints' not in sbj_inst:
                continue

            for obj_id, obj_inst in enumerate(im_insts):
                if sbj_id == obj_id:
                    continue

                obj_box = obj_inst['bbox']
                obj_cate_id = int(obj_inst['category_id'])

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
                        obj_cate_id,
                        sbj_id,
                        obj_id,
                        sbj_inst['pose_keypoints']
                    ])
                    neg_hoi_cnt += 1

    print('neg hoi count: %d' % neg_hoi_cnt)
    neg_rlt_anno_hico_path = os.path.join(prepare_root, '%s_NG_PIC_with_pose.pkl' % split)
    with open(neg_rlt_anno_hico_path, 'wb') as f:
        pickle.dump(neg_rlt_anno_hico, f)
    print('negative instances saved at: %s' % neg_rlt_anno_hico_path)


def get_prior(data_root, prepare_root):
    obj_cates = load_category_id_to_name(data_root, 'object')
    pre_cates = load_category_id_to_name(data_root, 'relation')
    rlt_anno = load_json(os.path.join(data_root, 'train_2019.json'))

    prior_table = np.zeros((len(obj_cates), len(pre_cates)))
    invalid_hoi_no_human_cnt = 0
    invalid_hoi_bad_rlt_cnt = 0
    for im_anno in rlt_anno:
        objs = im_anno['annotations']
        hois = im_anno['hoi_annotation']
        for hoi in hois:
            sbj_id = hoi['subject_id']
            obj_id = hoi['object_id']
            if sbj_id >= len(objs) or obj_id >= len(objs):
                invalid_hoi_bad_rlt_cnt += 1
                continue
            sbj_cate = int(objs[sbj_id]['category_id'])
            obj_cate = int(objs[obj_id]['category_id'])
            pre_cate = int(hoi['category_id'])
            if sbj_cate != 1:
                invalid_hoi_no_human_cnt += 1
                continue
            prior_table[obj_cate][pre_cate] += 1

    for obj_cate in range(1, prior_table.shape[0]):
        prior_table[obj_cate] = prior_table[obj_cate] / prior_table[obj_cate].sum()

    print('bad anno (no human): %d' % invalid_hoi_no_human_cnt)
    print('bad anno (bad anno): %d' % invalid_hoi_bad_rlt_cnt)
    prior_path = os.path.join(prepare_root, 'prior.pkl')
    with open(prior_path, 'wb') as f:
        pickle.dump(prior_table, f)
        print('prior saved at: %s' % prior_path)
    return prior_table


if __name__ == '__main__':
    data_root = 'data/hoia'
    prepare_root = 'data/hoia/proposal'
    get_prior(data_root, prepare_root)
    gen_positive_instances(data_root, prepare_root)
    gen_negative_instances(data_root, prepare_root)
    gen_positive_instances(data_root, prepare_root, split='test')
    gen_negative_instances(data_root, prepare_root, split='test')
