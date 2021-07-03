# @CreateTime : 2021/3/25
# @Author : sunx

import os
import pickle
import numpy as np
from utils import *
from gensim import models

def gen_word2vec(data_root, prepare_root):
    cate_list_path = os.path.join(data_root, 'object.txt')
    with open(cate_list_path, 'r') as f:
        cate_list = f.readlines()
        cate_list = [obj_cate.strip() for obj_cate in cate_list]
        cate_list = ['none'] + cate_list

    print('loading GoogleNews-vectors-negative300 ... ')
    all_w2v_path = os.path.join(prepare_root, 'GoogleNews-vectors-negative300.bin')
    all_w2v = models.KeyedVectors.load_word2vec_format(all_w2v_path, binary=True)

    cate_vectors = np.zeros((len(cate_list), 300))
    for i, cate in enumerate(cate_list):
        cate_vec = all_w2v[cate]
        cate_vectors[i] = cate_vec

    save_path = os.path.join(prepare_root, 'obj2vec.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(cate_vectors, f)
    print('object vectors save at: %s' % save_path)


data_root = 'data/hoia'
prepare_root = 'data/hoia/mlcnet_data'
gen_word2vec(data_root, prepare_root)