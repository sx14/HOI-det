# @CreateTime : 2021/3/9
# @Author : sunx

import os
import math
import json
import cv2
from multiprocessing import Process, Manager
from scipy import stats
import numpy as np
from tqdm import tqdm
from utils import *


def gen_instance_categories(data_root, save_root, split='train', proc_num=8):
    def process(ins_root, sem_root, im_ids, res_dict, proc_id):
        # 处理一批图像

        print('proc %d start!' % proc_id)
        ins_cates = {}
        for im_id in im_ids:
            # 加载实例分割和语义分割mask
            ins_mask_path = os.path.join(ins_root, im_id + '.png')
            sem_mask_path = os.path.join(sem_root, im_id + '.png')
            ins_mask = cv2.imread(ins_mask_path, cv2.IMREAD_GRAYSCALE)
            sem_mask = cv2.imread(sem_mask_path, cv2.IMREAD_GRAYSCALE)

            im_ins_cates = {}
            ins_cates[im_id.split('.')[0]] = im_ins_cates

            # 获取当前图像所有instance id
            ins_ids = np.unique(ins_mask)
            ins_ids = ins_ids[ins_ids != 0]

            # 对每一个instance，获取类别和box
            for ins_id in ins_ids:
                one_ins_mask = ins_mask == ins_id
                one_ins_box = cal_box(one_ins_mask)
                one_ins_box = {k: int(v) for k, v in one_ins_box.items()}
                one_ins_cate = stats.mode(sem_mask[one_ins_mask])[0][0]
                im_ins_cates[str(ins_id)] = {'category': int(one_ins_cate),
                                             'box': one_ins_box}
        res_dict[proc_id] = ins_cates
        print('proc %d finish!' % proc_id)

    # 加载split的image id
    image_ids_path = os.path.join(data_root, 'list5', '%s_id' % split)
    image_ids = load_image_list(image_ids_path)

    ins_root = os.path.join(data_root, 'instance', split)
    sem_root = os.path.join(data_root, 'semantic', split)
    ins_im_ids = os.listdir(ins_root)
    sem_im_ids = os.listdir(sem_root)
    ins_num = len(ins_im_ids)
    sem_num = len(sem_im_ids)
    assert ins_num == sem_num == len(image_ids)

    # # 单进程处理
    # proc_res_dict = {}
    # process(ins_root, sem_root, image_list, proc_res_dict, 0)

    # 多进程处理
    proc_im_num = math.ceil(ins_num / proc_num)
    proc_manager = Manager()
    proc_res_dict = proc_manager.dict()
    proc_jobs = []
    for proc_id, im_id in enumerate(range(0, ins_num, proc_im_num)):
        proc_im_ids = image_ids[im_id: im_id + proc_im_num]
        proc_job = Process(target=process, args=(ins_root, sem_root,
                                                 proc_im_ids, proc_res_dict, proc_id))
        proc_jobs.append(proc_job)
        proc_job.start()

    for jobs in proc_jobs:
        jobs.join()

    # 合并多进程处理结果
    ins_cates = {}
    for res in proc_res_dict.values():
        ins_cates.update(res)
    assert len(ins_cates.keys()) == ins_num

    # 保存
    save_path = os.path.join(save_root, 'instances_%s.json' % split)
    save_json(ins_cates, save_path)


if __name__ == '__main__':
    data_root = os.path.join('data', 'pic')
    save_root = os.path.join('data', 'pic', 'mlcnet_data')
    gen_instance_categories(data_root, save_root, 'train')
    gen_instance_categories(data_root, save_root, 'val')
