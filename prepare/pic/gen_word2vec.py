# @CreateTime : 2021/3/25
# @Author : sunx

import os
import pickle
import numpy as np
from utils import *
from gensim import models

category_map = {
    'PC': 'computer',
    'kitchen_island': 'kitchen',
    'money_coin': 'coin',
    'street_light': 'streetlight',
    'body_building_apparatus': 'apparatus',
    'military_equipment': 'military',
    'amusement_facilities': 'amusement',
    'swimming_things': 'swimming',
    'fishing_rod': 'rod',
    'painting/poster': 'painting',
    'remote_control': 'controller',
}

def gen_word2vec(data_root, prepare_root):
    cate_list_path = os.path.join(data_root, 'categories_list', 'label_categories.json')
    cate_list = load_categories(cate_list_path)

    print('loading GoogleNews-vectors-negative300 ... ')
    all_w2v_path = os.path.join(prepare_root, 'GoogleNews-vectors-negative300.bin')
    all_w2v = models.KeyedVectors.load_word2vec_format(all_w2v_path, binary=True)

    cate_vectors = np.zeros((len(cate_list), 300))
    for i, cate in enumerate(cate_list):
        if cate in category_map:
            cate = category_map[cate]
        cate_vec = all_w2v[cate]
        cate_vectors[i] = cate_vec

    save_path = os.path.join(prepare_root, 'obj2vec.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(cate_vectors, f)
    print('object vectors save at: %s' % save_path)


data_root = 'data/pic'
prepare_root = 'data/pic/mlcnet_data'
gen_word2vec(data_root, prepare_root)