# @CreateTime : 2021/5/21
# @Author : sunx

from my_utils import *
from tqdm import tqdm

def res_to_dict(res):
    fid_to_res = {item['file_name']: item for item in res}
    return fid_to_res

def merge(res1, res2, res1_cates, res2_cates):
    res1 = res_to_dict(res1)
    res2 = res_to_dict(res2)
    merged_res = []
    fids = set(list(res1.keys())+list(res2.keys()))
    for fid in tqdm(fids):
        merged_item = {
            'file_name': fid,
            'predictions': [],
            'hoi_prediction': []
        }
        merged_res.append(merged_item)

        if fid in res1:
            items1 = res1[fid]
            merged_item['predictions'] = items1['predictions']
            for hoi in items1['hoi_prediction']:
                cid = hoi['category_id']
                if cid in res1_cates:
                    merged_item['hoi_prediction'].append(hoi)

        if fid in res2:
            items2 = res2[fid]
            org_obj_cnt = len(merged_item['predictions'])
            merged_item['predictions'] += items2['predictions']
            for hoi in items2['hoi_prediction']:
                cid = hoi['category_id']
                hoi['subject_id'] += org_obj_cnt
                hoi['object_id'] += org_obj_cnt
                if cid in res2_cates:
                    merged_item['hoi_prediction'].append(hoi)
    return merged_res


res1_path = '/home/magus/data/C0008-challenge/results/1-naive-prior-val.json'
res2_path = '/home/magus/data/C0008-challenge/results/2-multi-branch-cnn-val-with-yolo.json'
res1_cates = {1, 7}
res2_cates = {2, 3, 4, 5, 6, 8, 9, 10}
res1 = load_json(res1_path)
res2 = load_json(res2_path)
merged_res = merge(res1, res2, res1_cates, res2_cates)
save_path = '/home/magus/data/C0008-challenge/results/3-merge1&2-val-with-casc-yolo.json'
save_json(merged_res, save_path)