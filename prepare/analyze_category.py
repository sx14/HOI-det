import _init_paths
import os
from tqdm import tqdm
import numpy as np
import pickle
from datasets.hico2 import hico2
from utils.show_box import show_boxes
from global_config import DS_ROOT


def show_categories(data_root):
    hoi_classes, obj_classes, vrb_classes, obj2int, hoi2vrb, _ = hico2.load_hoi_classes(data_root)
    for obj in obj2int:
        hoi_start_end = obj2int[obj]
        hois = hoi_classes[hoi_start_end[0]:hoi_start_end[1]+1]
        print('==== %s ====' % obj)
        for hoi in hois:
            print(hoi.hoi_name())


def cooccurent_verbs(data_root):
    hoi_classes, obj_classes, vrb_classes, obj2int, hoi2vrb, _ = hico2.load_hoi_classes(data_root)
    vrb2ind = dict(zip(vrb_classes, range(len(vrb_classes))))
    anno_path = os.path.join(data_root, 'train_GT_HICO.pkl')
    cooccurence_matrix = np.zeros((len(vrb_classes), len(vrb_classes)))
    print('Loading annotations ...')
    with open(anno_path) as f:
        anno_db = pickle.load(f)
    for hoi in tqdm(anno_db):
        hoi_cates = hoi[1]
        hoi_cates = [hoi_classes[hoi_cate] for hoi_cate in hoi_cates]
        vrb_inds = [vrb2ind[hoi_cate.verb_name()] for hoi_cate in hoi_cates]
        for vrb1 in vrb_inds:
            for vrb2 in vrb_inds:
                cooccurence_matrix[vrb1, vrb2] += 1

    for vrb_id, vrb in enumerate(vrb_classes):
        print('==== %s ====' % vrb)
        cooccurences = cooccurence_matrix[vrb_id]
        cooccurences = cooccurences / cooccurences[vrb_id]
        sorted_co_vrb_ids = np.argsort(cooccurences)[::-1]
        for co_vrb_id in sorted_co_vrb_ids:
            if cooccurences[co_vrb_id] > 0:
                print(('%s' % vrb_classes[co_vrb_id]).rjust(10) + ':%.8f' % (cooccurences[co_vrb_id]))


if __name__ == '__main__':
    cooccurent_verbs(DS_ROOT)