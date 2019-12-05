import os
from global_config import DS_ROOT
import cv2
import numpy as np

cate_pix_cnt_path = 'hico_scene_pixel_ratio.npy'
cate_img_cnt_path = 'hico_scene_image_ratio.npy'
if not os.path.exists(cate_pix_cnt_path) or not os.path.exists(cate_img_cnt_path):
    cate_pix_cnt = np.zeros(151)
    cate_img_cnt = np.zeros(151)
    hico_scene_root = os.path.join(DS_ROOT, 'scene', 'labels', 'train2015')
    for image_id in os.listdir(hico_scene_root):
        image_path = os.path.join(hico_scene_root, image_id)
        label = cv2.imread(image_path)

        curr_img_cnt = np.zeros(151)
        curr_pix_cnt = np.zeros(151)
        for r in range(label.shape[0]):
            for c in range(label.shape[1]):
                cate_id = label[r, c]
                curr_pix_cnt[cate_id] += 1
                curr_img_cnt[cate_id] = 1

        for cate in range(151):
            cate_mask = label == cate
            cate_pix = cate_mask.sum()
            curr_pix_cnt[cate] = cate_pix
            if cate_pix > 0:
                curr_img_cnt[cate] = 1

        cate_img_cnt = cate_img_cnt + curr_img_cnt
        cate_pix_cnt = cate_pix_cnt + curr_pix_cnt

    np.save(cate_img_cnt_path, cate_img_cnt)
    np.save(cate_pix_cnt_path, cate_pix_cnt)
else:
    cate_pix_cnt = np.load(cate_pix_cnt_path)
    cate_img_cnt = np.load(cate_img_cnt_path)


with open('objectInfo150.txt') as f:
    lines = f.readlines()[1:]
    obj_infos = [line.split('\t') for line in lines]
cate_names = ['background'] + [info[-1].strip() for info in obj_infos]


print('===== pixel ratio =====')
pixel_sum = cate_pix_cnt.sum()
cate_pix_ratio = cate_pix_cnt / pixel_sum
cate_pix_sort = np.argsort(cate_pix_ratio)[::-1]
for cate_id in cate_pix_sort:
    print('%s: %f' % (cate_names[cate_id], cate_pix_ratio[cate_id]))

print('===== image ratio =====')
image_sum = len(os.listdir(hico_scene_root))
cate_img_ratio = cate_img_cnt / image_sum
cate_img_sort = np.argsort(cate_img_ratio)[::-1]
for cate_id in cate_pix_sort:
    print('%s: %f' % (cate_names[cate_id], cate_img_ratio[cate_id]))