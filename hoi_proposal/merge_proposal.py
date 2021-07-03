# @CreateTime : 2021/6/9
# @Author : sunx

import json
import pickle
from collections import defaultdict


proposal_path = '../output/hoia_full/test2019_results.pkl'
# proposal_path = '../output/hoia_full/test2021_results.pkl'
hoi_path = '../output/hoia_full/all_hoi_detections_2019_with_yolo_no_aug_avg_fuse_pose.json'
# hoi_path = '../output/hoia_full/all_hoi_detections_2021_yolo_avg_fuse_pose.json'
hoi_sav_path = '../output/hoia_full/all_hoi_detections_2019_with_yolo_no_aug_avg_fuse_pose_pro.json'
# hoi_sav_path = '../output/hoia_full/all_hoi_detections_2021_yolo_avg_fuse_pose_pro.json'


with open(proposal_path, 'rb') as f:
    proposal_res = pickle.load(f)
    proposal_info = proposal_res['info']
    proposal_prob = proposal_res['results']

with open(hoi_path, 'r') as f:
    hoi_res = json.load(f)

im_id_to_pro = defaultdict(dict)
for i in range(len(proposal_info)):
    info = proposal_info[i]
    prob = proposal_prob[i]
    file_name = info['file_name']
    hoi_id = info['hoi_id']
    im_id_to_pro[file_name][hoi_id] = prob

for im_res in hoi_res:
    im_file_name = im_res['file_name']
    if im_file_name not in im_id_to_pro:
        continue
    im_hois = im_res['hoi_prediction']
    for hoi_id, hoi in enumerate(im_hois):
        if hoi_id in im_id_to_pro[im_file_name]:
            hoi['score'] *= im_id_to_pro[im_file_name][hoi_id]

with open(hoi_sav_path, 'w') as f:
    json.dump(hoi_res, f)
print('merged result saved at: %s' % hoi_sav_path)