import os
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


eval_def = sio.loadmat(eval_def_path)
def_recs = eval_def['REC'].reshape(600)
def_aps = eval_def['AP'].reshape(600)

def_order = np.argsort(def_recs)[::-1]
def_hois = hoi_labels[def_order]
def_aps = def_aps[def_order]
def_recs = def_recs[def_order]

color_list = ['b', 'g', 'r']

for i in range(0, 12):
    fig = plt.figure(figsize=(15, 10))
    fig.tight_layout()

    hois = def_hois[i*50: (i+1)*50]
    recs = def_recs[i*50: (i+1)*50]
    aps = def_aps[i*50: (i+1)*50]

    ax1 = fig.subplots()
    xlocation = np.linspace(1, len(hois) * 0.6, len(hois))
    rects01 = ax1.bar(xlocation, aps, width=0.2, color='b', linewidth=1, alpha=0.8)
    rects02 = ax1.bar(xlocation + 0.2, recs, width=0.2, color='g', linewidth=1, alpha=0.8)
    plt.xticks(xlocation + 0.15, hois, fontsize=12, rotation=90)
    plt.show()






eval_ko = sio.loadmat(eval_ko_path)
eval_ko_rec = eval_ko['REC']
eval_ko_ap = eval_ko['AP']




