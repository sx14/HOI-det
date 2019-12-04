import os
from global_config import DS_ROOT
import cv2
import numpy as np

cate_counter_path = 'hico_scene_ratio.npy'
if not os.path.exists(cate_counter_path):
    cate_counter = np.zeros(151).astype(np.float64)
    hico_scene_root = os.path.join(DS_ROOT, 'scene', 'labels', 'train2015')
    for image_id in os.listdir(hico_scene_root):
        image_path = os.path.join(hico_scene_root, image_id)
        label = cv2.imread(image_path)

        for r in range(label.shape[0]):
            for c in range(label.shape[1]):
                cate_id = label[r, c]
                cate_counter[cate_id] += 1
    np.save(cate_counter_path, hico_scene_root)
else:
    cate_counter = np.load(cate_counter_path)


with open('objectInfo150.txt') as f:
    lines = f.readlines()[1:]
    obj_infos = [line.split('\t') for line in lines]

cate_names = ['background'] + [info[-1].strip() for info in obj_infos]
pixel_sum = cate_counter.sum()
cate_ratio = cate_counter / pixel_sum
cate_sort = np.argsort(cate_ratio)[::-1]
for cate_id in cate_sort:
    print('%s: %f' % (cate_names[cate_id], cate_ratio[cate_id]))