import os
import pickle
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

import _init_paths
from global_config import PROJECT_ROOT, DS_ROOT
from datasets.hico2 import hico2

hoi_classes, obj_classes, vrb_classes, obj2int, hoi2vrb, vrb2hoi = hico2.load_hoi_classes(DS_ROOT)
hoi_labels = np.array([hoi.hoi_name() for hoi in hoi_classes])

exp_name = 'rcnn_w2v_init_8ep'
eval_def_path = os.path.join(PROJECT_ROOT, 'output', 'results', exp_name, 'eval_result_def.mat')
eval_ko_path = os.path.join(PROJECT_ROOT, 'output', 'results', exp_name, 'eval_result_ko.mat')
hoi_freq_path = os.path.join(DS_ROOT, 'hoi_frequencies.pkl')

with open(hoi_freq_path) as f:
    hoi_freqs = pickle.load(f)

eval_def = sio.loadmat(eval_def_path)
def_recs = eval_def['REC'].reshape(600)
def_aps = eval_def['AP'].reshape(600)

def_order = np.argsort(hoi_freqs)[::-1]
hoi_freqs = hoi_freqs[def_order]
def_hois = hoi_labels[def_order]
def_aps = def_aps[def_order]
def_recs = def_recs[def_order]


for i in range(0, 12):
    fig = plt.figure(figsize=(15, 10))
    fig.tight_layout()

    hois = def_hois[i*50: (i+1)*50]
    recs = def_recs[i*50: (i+1)*50]
    aps = def_aps[i*50: (i+1)*50]
    freqs = hoi_freqs[i*50: (i+1)*50]

    ax1 = fig.subplots()
    xlocation = np.linspace(1, len(hois) * 0.6, len(hois))
    rects03 = ax1.bar(xlocation + 0.0, freqs, width=0.2, color='r', linewidth=1, alpha=0.8)
    rects01 = ax1.bar(xlocation + 0.2, aps, width=0.2, color='b', linewidth=1, alpha=0.8)
    rects02 = ax1.bar(xlocation + 0.4, recs, width=0.2, color='g', linewidth=1, alpha=0.8)
    plt.xticks(xlocation + 0.15, hois, fontsize=12, rotation=90)
    plt.show()





