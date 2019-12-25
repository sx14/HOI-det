import os
import pickle
import numpy as np
from tqdm import tqdm

import _init_paths
from global_config import PROJECT_ROOT, DS_ROOT
from datasets.hico2 import hico2



def load_annotation(dataset_root):
    anno_path = os.path.join(dataset_root, 'train_GT_HICO_with_pose.pkl')
    print('Loading annotations ...')
    with open(anno_path) as f:
        anno = pickle.load(f)
    return anno


def get_hoi_frequency(anno_db):
    hoi_counter = np.zeros(600)
    ins_num = 0

    print('Counting ...')
    for instance in tqdm(anno_db):
        hois = instance[1]
        hoi_counter[hois] += 1
        ins_num += len(hois)

    return hoi_counter / ins_num


def show_hoi_frequency(freqs, hoi_cates):
    freq_order = np.argsort(freqs)[::-1]
    for order in freq_order:
        print('%s:'.rjust(20) % hoi_cates[order].hoi_name()+'%.4f' % freqs[order])


if __name__ == '__main__':
    anno_db = load_annotation(DS_ROOT)
    hoi_freqs = get_hoi_frequency(anno_db)

    hico_freq_path = os.path.join(DS_ROOT, 'hoi_frequencies.pkl')
    with open(hico_freq_path, 'wb') as f:
        pickle.dump(hoi_freqs, f)

    hoi_classes, obj_classes, vrb_classes, obj2int, hoi2vrb, vrb2hoi = hico2.load_hoi_classes(DS_ROOT)
    show_hoi_frequency(hoi_freqs, hoi_classes)



