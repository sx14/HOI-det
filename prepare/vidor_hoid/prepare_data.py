import os
from copy import deepcopy
import time
import glob
import json
import shutil
from collections import defaultdict

from tqdm import tqdm


def prepare_images(video_root, image_root, sample_interval=60):
    if not os.path.exists(image_root):
        os.makedirs(image_root)

    print('Copying frames (interval=%d) ...' % sample_interval)
    time.sleep(2)
    sample_cnt = 0
    for pid in tqdm(sorted(os.listdir(video_root))):
        pkg_root = os.path.join(video_root, pid)
        for vid in sorted(os.listdir(pkg_root)):
            frm_root = os.path.join(pkg_root, vid)
            frame_files = sorted(os.listdir(frm_root))
            frame_samples = frame_files[::sample_interval]
            for frame_file in frame_samples:
                frame_path_src = os.path.join(frm_root, frame_file)
                frame_path_dst = os.path.join(image_root, '%s_%s' % (vid, frame_file))
                shutil.copyfile(frame_path_src, frame_path_dst)
            sample_cnt += len(frame_samples)
    time.sleep(1)
    print('Sampled %d frames.' % sample_cnt)


def prepare_anno_jsons(vid_anno_root, img_anno_root, sample_interval=60):
    if not os.path.exists(img_anno_root):
        os.makedirs(img_anno_root)

    print('Generating frame annotation files (interval=%d) ...' % sample_interval)
    time.sleep(2)
    sample_cnt = 0
    for pid in tqdm(sorted(os.listdir(vid_anno_root))):
        pkg_root = os.path.join(vid_anno_root, pid)
        for vid_file in sorted(os.listdir(pkg_root)):
            vid_anno_file_path = os.path.join(pkg_root, vid_file)
            with open(vid_anno_file_path) as f:
                vid_anno = json.load(f)
                vid_len = vid_anno['frame_count']

            tid2cate = {}
            for traj_info in vid_anno['subject/objects']:
                tid2cate[traj_info['tid']] = traj_info['category']

            frm_tid2det = [{} for _ in range(vid_len)]
            frm_dets = vid_anno['trajectories']
            for fid in range(vid_len):
                dets = frm_dets[fid]
                for det in dets:
                    frm_tid2det[fid][det['tid']] = {'bbox': [det['bbox']['xmin'],
                                                             det['bbox']['ymin'],
                                                             det['bbox']['xmax'],
                                                             det['bbox']['ymax']],
                                                    'category': tid2cate[det['tid']]}

            frm_relations = [[] for _ in range(vid_len)]
            for relation in vid_anno['relation_instances']:
                stt_fid = relation['begin_fid']
                end_fid = relation['end_fid']
                sbj_tid = relation['subject_tid']
                obj_tid = relation['object_tid']
                for fid in range(stt_fid, end_fid):
                    frm_relations[fid].append({
                        'sbj_tid': sbj_tid,
                        'obj_tid': obj_tid,
                        'predicate': relation['predicate']})

            for fid in range(0, vid_len, sample_interval):
                dets = frm_tid2det[fid]
                rlts = frm_relations[fid]
                frm_anno = {
                    'detections': dets,
                    'interactions': rlts}
                vid = vid_file.split('.')[0]
                frm_anno_path = os.path.join(img_anno_root, '%s_%06d.json' % (vid, fid))
                with open(frm_anno_path, 'w') as f:
                    json.dump(frm_anno, f)
                sample_cnt += 1
    time.sleep(1)
    print('Sampled %d frame annotations.' % sample_cnt)


def is_human(cate):
    return cate in {'adult', 'child', 'baby'}


def supplement_skeletons(anno_root, pose_root, anno_with_pose_root):

    def cal_iou(box1, box2):
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2

        xmini = max(xmin1, xmin2)
        ymini = max(ymin1, ymin2)
        xmaxi = min(xmax1, xmax2)
        ymaxi = min(ymax1, ymax2)

        area1 = (xmax1 - xmin1 + 1) * (ymax1 - ymin1 + 1)
        area2 = (xmax2 - xmin2 + 1) * (ymax2 - ymin2 + 1)
        areai = max((xmaxi - xmini + 1), 0) * max((ymaxi - ymini + 1), 0)

        return areai * 1.0 / (area1 + area2 - areai)

    def get_skeleton_box(kps):
        xmin = 9999
        ymin = 9999
        xmax = -999
        ymax = -999

        assert len(kps) == 51
        for j in range(0, len(kps), 3):
            x = kps[j + 0]
            y = kps[j + 1]

            xmin = min(xmin, x)
            ymin = min(ymin, y)
            xmax = max(xmax, x)
            ymax = max(ymax, y)

        return [xmin, ymin, xmax, ymax]

    if not os.path.exists(anno_with_pose_root):
        os.makedirs(anno_with_pose_root)

    print('Merging skeletons to frame annotations ...')
    time.sleep(2)
    hum_cnt = 0
    skt_cnt = 0
    for pkg_id in tqdm(os.listdir(pose_root)):
        pkg_root = os.path.join(pose_root, pkg_id)
        for pose_file in os.listdir(pkg_root):
            pose_path = os.path.join(pkg_root, pose_file)
            with open(pose_path) as f:
                all_skts = json.load(f)
            fid2skts = defaultdict(list)
            for skt in all_skts:
                fid = skt['image_id'].split('.')[0]
                fid2skts[fid].append(skt['keypoints'])

            vid = pose_file.split('.')[0]
            anno_files = glob.glob(os.path.join(anno_root, '%s_*.json' % vid))
            for anno_path in sorted(anno_files):
                with open(anno_path) as f:
                    anno = json.load(f)

                anno_file = anno_path.split('/')[-1]
                fid = anno_file.split('.')[0].split('_')[-1]
                frm_skts = fid2skts[fid]
                frm_dets = anno['detections']
                for det in frm_dets.values():
                    if not is_human(det['category']):
                        continue
                    hum_cnt += 1
                    max_iou = 0
                    max_skt_idx = -1
                    for skt_idx, skt in enumerate(frm_skts):
                        skt_box = get_skeleton_box(skt)
                        iou = cal_iou(det['bbox'], skt_box)
                        if iou > max_iou:
                            skt_cnt += 1
                            max_iou = iou
                            max_skt_idx = skt_idx

                    if max_iou > 0.4:
                        det['skeleton'] = frm_skts[max_skt_idx]
                    else:
                        det['skeleton'] = None

                anno_with_pose_path = os.path.join(anno_with_pose_root, anno_file)
                with open(anno_with_pose_path, 'w') as f:
                    json.dump(anno, f)
    time.sleep(1)
    print('Success ratio: %.2f' % (skt_cnt * 1.0 / hum_cnt))


def generate_anno_package(anno_root, pre2idx, obj2idx, save_root):

    def gen_negative_samples(tid2det, pos_insts):
        pos_sid_oid = {'%d_%d' % (rlt['sbj_tid'], rlt['obj_tid']) for rlt in pos_insts}

        neg_insts = []
        for sbj_tid, sbj_det in tid2det.items():
            if not is_human(sbj_det['category']):
                continue
            for obj_tid, obj_det in tid2det.items():
                if sbj_tid == obj_tid or '%s_%s' % (sbj_tid, obj_tid) in pos_sid_oid:
                    continue
                neg_insts.append({
                    'sbj_tid': sbj_tid,
                    'obj_tid': obj_tid,
                    'predicate': '__no_interaction__'})
        return neg_insts

    print('Generating annotation packages ...')
    time.sleep(2)
    pkgs = {'pos': [], 'neg': []}
    for anno_file in tqdm(sorted(os.listdir(anno_root))):
        img_id = anno_file.split('.')[0]
        anno_path = os.path.join(anno_root, anno_file)
        with open(anno_path) as f:
            frm_anno = json.load(f)

        tid2det = frm_anno['detections']
        pos_insts = frm_anno['interactions']
        neg_insts = gen_negative_samples(tid2det, pos_insts)
        frm_insts = {'pos': pos_insts, 'neg': neg_insts}

        for pn, insts in frm_insts.items():

            inst_list = []
            for inst in insts:

                pre_cate = inst['predicate']
                pre_idx = pre2idx[pre_cate]

                sbj_tid = str(inst['sbj_tid'])
                sbj_cate = tid2det[sbj_tid]['category']
                sbj_cate_idx = obj2idx[sbj_cate]
                sbj_box = deepcopy(tid2det[sbj_tid]['bbox'])
                sbj_box.append(sbj_cate_idx)
                sbj_skt = tid2det[sbj_tid]['skeleton']

                obj_tid = str(inst['obj_tid'])
                obj_cate = tid2det[obj_tid]['category']
                obj_cate_idx = obj2idx[obj_cate]
                obj_box = deepcopy(tid2det[obj_tid]['bbox'])
                obj_box.append(obj_cate_idx)
                inst_list.append([img_id, [pre_idx], sbj_box, obj_box, sbj_skt])

            pkgs[pn] += inst_list

    import pickle
    pos_pkg_path = os.path.join(save_root, 'train_POS_with_pose.pkl')
    with open(pos_pkg_path, 'w') as f:
        pickle.dump(pkgs['pos'], f)

    neg_pkg_path = os.path.join(save_root, 'train_NEG_with_pose.pkl')
    with open(neg_pkg_path, 'w') as f:
        pickle.dump(pkgs['neg'], f)


def load_category_list(cate_path):
    with open(cate_path) as f:
        return [l.strip() for l in f.readlines()]


if __name__ == '__main__':
    print('==== VidOR-HOID-mini data preparation ====')
    data_root = '../../data/vidor_hoid_mini'
    save_root = data_root + '/image_data'

    # sample frames
    video_root = data_root + '/Data/VID/train'
    image_root = save_root + '/Data/VID/train'
    prepare_images(video_root, image_root)

    # frame annotations
    vid_anno_root = data_root + '/anno/train'
    img_anno_root = save_root + '/anno/train'
    prepare_anno_jsons(vid_anno_root, img_anno_root)

    # supplement skeletons
    pose_root = data_root + '/Pose/VID/train'
    img_anno_with_pose_root = save_root + '/anno_with_pose/train'
    supplement_skeletons(img_anno_root, pose_root, img_anno_with_pose_root)

    # gen anno package
    obj_cate_path = data_root + '/object_labels.txt'
    pre_cate_path = data_root + '/predicate_labels.txt'
    obj_cates = load_category_list(obj_cate_path)
    pre_cates = load_category_list(pre_cate_path)
    pre_cates = ['__no_interaction__'] + pre_cates
    obj2idx = {cate: i for i, cate in enumerate(obj_cates)}
    pre2idx = {cate: i for i, cate in enumerate(pre_cates)}
    generate_anno_package(img_anno_with_pose_root, pre2idx, obj2idx, save_root)

    # copy labels, word vector
    shutil.copyfile(obj_cate_path, save_root+'/object_labels.txt')
    shutil.copyfile(pre_cate_path, save_root + '/predicate_labels.txt')
    shutil.copyfile(data_root + '/object_vectors.mat', save_root + '/object_vectors.mat')
    print('==== Done ====')