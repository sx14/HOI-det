import os
import time
import json
import copy
import yaml

import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from prepare import ext_spatial_feat
from dataset import VidOR


def merge(vid_res_dir, res_path):
    # generate final result file with low memory
    wf = open(res_path, 'w')
    wf.write('{')
    wf.write('"version": "VERSION 1.0",')
    wf.write('"external_data": {},')
    wf.write('"results": {')

    vid_res_names = os.listdir(vid_res_dir)
    vid_res_num = len(vid_res_names)
    for i in tqdm(range(vid_res_num)):
        vid_res_name = vid_res_names[i]
        vid_res_path = os.path.join(vid_res_dir, vid_res_name)
        with open(vid_res_path) as rf:
            # check file
            json_content = json.load(rf)[vid_res_name.split('.')[0]]
        with open(vid_res_path) as rf:
            vid_res = rf.readline()
            if i == (vid_res_num-1):
                vid_res_str = vid_res[1:-1]
            else:
                vid_res_str = vid_res[1:-1]+','
            wf.write(vid_res_str)

    wf.write('}')
    wf.write('}')
    wf.close()


def load_json(json_path):
    with open(json_path) as f:
        videos = json.load(f)
    return videos


def save_json(results, save_path):
    with open(save_path, 'w') as f:
        json.dump(results, f)


def load_model(config, target):
    from model import FCLayers
    model = FCLayers(config['%s_feat_dim' % target], config['num_classes'])

    weight_path = os.path.join(config['weights_dir'], 'model_%s_%d.pkl' % (target, config['test_epoch_num']))
    param_values = torch.load(weight_path).values()
    model_dict = model.state_dict()
    for name, param in zip(list(model_dict.keys()), list(param_values)):
        model_dict[name] = param
    model.load_state_dict(model_dict)
    return model


def generate_relation_segments(sbj, obj, seg_len=30):
    video_h = sbj['height']
    video_w = sbj['width']

    sbj_traj = sbj['trajectory']
    sbj_fids = sorted([int(fid) for fid in sbj_traj.keys()])
    sbj_stt_fid = min(sbj_fids)
    sbj_end_fid = max(sbj_fids)
    sbj_cls = sbj['category']
    sbj_scr = sbj['score']

    obj_traj = obj['trajectory']
    obj_fids = sorted([int(fid) for fid in obj_traj.keys()])
    obj_stt_fid = min(obj_fids)
    obj_end_fid = max(obj_fids)
    obj_cls = obj['category']
    obj_scr = obj['score']

    rela_segments = []
    if sbj_end_fid < obj_stt_fid or sbj_stt_fid > obj_end_fid:
        # no temporal intersection
        return rela_segments

    # intersection
    i_stt_fid = max(sbj_stt_fid, obj_stt_fid)
    i_end_fid = min(sbj_end_fid, obj_end_fid)

    for seg_stt_fid in range(i_stt_fid, i_end_fid + 1, seg_len):
        seg_end_fid = min(seg_stt_fid + seg_len - 1, i_end_fid)
        seg_dur = seg_end_fid - seg_stt_fid + 1

        if seg_dur >= 2:

            seg_sbj_traj = {}
            seg_obj_traj = {}

            for fid in range(seg_stt_fid, seg_end_fid + 1):
                seg_sbj_traj['%06d' % fid] = sbj_traj['%06d' % fid]
                seg_obj_traj['%06d' % fid] = obj_traj['%06d' % fid]

            seg = {'sbj_traj': seg_sbj_traj,
                   'obj_traj': seg_obj_traj,
                   'sbj_cls': sbj_cls,
                   'obj_cls': obj_cls,
                   'sbj_scr': sbj_scr,
                   'obj_scr': obj_scr,
                   'vid_h': video_h,
                   'vid_w': video_w,
                   'connected': False}
            rela_segments.append(seg)
    return rela_segments


def extract_segment_feature(rela_segment, object_vecs, ds):
    video_h = rela_segment['vid_h']
    video_w = rela_segment['vid_w']

    sbj_cls = rela_segment['sbj_cls']
    sbj_traj = rela_segment['sbj_traj']
    sbj_id = ds.obj_name2ind[sbj_cls]
    sbj_vec = object_vecs[sbj_id]

    obj_cls = rela_segment['obj_cls']
    obj_traj = rela_segment['obj_traj']
    obj_id = ds.obj_name2ind[obj_cls]
    obj_vec = object_vecs[obj_id]

    lang_feat = np.concatenate((sbj_vec, obj_vec))
    lang_feat = lang_feat[np.newaxis, :]

    fids = [int(fid) for fid in rela_segment['sbj_traj'].keys()]
    stt_fid = min(fids)
    end_fid = max(fids)
    spa_feats = []
    for fid in [stt_fid, end_fid]:
        sbj_box = {'xmin': sbj_traj['%06d' % fid][0],
                   'ymin': sbj_traj['%06d' % fid][1],
                   'xmax': sbj_traj['%06d' % fid][2],
                   'ymax': sbj_traj['%06d' % fid][3]}

        obj_box = {'xmin': obj_traj['%06d' % fid][0],
                   'ymin': obj_traj['%06d' % fid][1],
                   'xmax': obj_traj['%06d' % fid][2],
                   'ymax': obj_traj['%06d' % fid][3]}
        spa_feat = ext_spatial_feat(sbj_box, obj_box, video_h, video_w)
        spa_feats.append(spa_feat)

    seg_spa_feat = np.concatenate((spa_feats[0], spa_feats[1], spa_feats[1]-spa_feats[0]))
    seg_spa_feat = seg_spa_feat[np.newaxis, :]

    return seg_spa_feat, lang_feat


def predict_predicate(ds, rela_segments, spa_model, lan_model, use_gpu=True):
    if len(rela_segments) == 0:
        return rela_segments

    object_vecs = ds.obj_vecs
    spa_seg_feats = np.zeros((len(rela_segments), spa_model.feat_dim))
    lan_seg_feats = np.zeros((len(rela_segments), lan_model.feat_dim))
    for i, rela_seg in enumerate(rela_segments):
        spa_feat, lan_feat = extract_segment_feature(rela_seg, object_vecs, ds)
        spa_seg_feats[i] = spa_feat
        lan_seg_feats[i] = lan_feat
    spa_seg_feats = Variable(torch.from_numpy(spa_seg_feats)).float()
    lan_seg_feats = Variable(torch.from_numpy(lan_seg_feats)).float()

    if use_gpu:
        spa_seg_feats = spa_seg_feats.cuda()
        lan_seg_feats = lan_seg_feats.cuda()

    spa_outputs = spa_model(spa_seg_feats)
    spa_confs = F.softmax(spa_outputs, dim=1)
    lan_outputs = lan_model(lan_seg_feats)
    lan_confs = F.softmax(lan_outputs, dim=1)
    confs = spa_confs * 0.3 + lan_confs * 0.7

    all_rela_segments = [[] for _ in range(len(rela_segments))]
    # get top 10 predictions
    for t in range(10):
        pred_pre_scrs, pred_pre_ids = confs.data.max(1)  # get the index of the max log-probability
        for i, rela_seg in enumerate(rela_segments):
            rela_seg_copy = copy.deepcopy(rela_seg)
            pred_pre_id = pred_pre_ids[i].item()
            pred_pre_scr = pred_pre_scrs[i].item()
            pred_pre = ds.pre_ind2name[pred_pre_id]
            rela_seg_copy['pre_cls'] = pred_pre
            rela_seg_copy['pre_scr'] = pred_pre_scr
            all_rela_segments[i].append(rela_seg_copy)
            confs[i, pred_pre_id] = -1
    return all_rela_segments


def greedy_association(rela_cand_segments):
    if len(rela_cand_segments) == 0:
        return []

    rela_instances = []
    for i in range(len(rela_cand_segments)):
        curr_segments = rela_cand_segments[i]

        for j in range(len(curr_segments)):
            # current segment
            curr_segment = curr_segments[j]
            curr_scores = [curr_segment['pre_scr']]
            if curr_segment['connected']:
                continue
            else:
                curr_segment['connected'] = True

            for p in range(i+1, len(rela_cand_segments)):
                # connect next segment
                next_segments = rela_cand_segments[p]
                success = False
                for q in range(len(next_segments)):

                    next_segment = next_segments[q]
                    if next_segment['connected']:
                        continue

                    if curr_segment['pre_cls'] == next_segment['pre_cls']:
                        # merge trajectories
                        curr_sbj = curr_segment['sbj_traj']
                        curr_seg_sbj = next_segment['sbj_traj']
                        curr_sbj.update(curr_seg_sbj)
                        curr_obj = curr_segment['obj_traj']
                        curr_seg_obj = next_segment['obj_traj']
                        curr_obj.update(curr_seg_obj)

                        # record segment predicate scores
                        curr_scores.append(next_segment['pre_scr'])
                        next_segment['connected'] = True
                        success = True
                        break

                if not success:
                    break

            curr_segment['pre_scr'] = sum(curr_scores) / len(curr_scores)
            curr_segment['score'] = curr_segment['sbj_scr'] * curr_segment['obj_scr'] * curr_segment['pre_scr']
            rela_instances.append(curr_segment)
    return rela_instances


def filter(rela_cands, max_per_video):
    sorted_cands = sorted(rela_cands, key=lambda rela: rela['score'], reverse=True)
    return sorted_cands[:max_per_video]


def format(relas):
    format_relas = []
    for rela in relas:
        format_rela = dict()
        format_rela['triplet'] = [rela['sbj_cls'], rela['pre_cls'], rela['obj_cls']]
        format_rela['score'] = rela['score']

        sbj_traj = rela['sbj_traj']
        obj_traj = rela['obj_traj']
        sbj_fid_boxes = sorted(sbj_traj.items(), key=lambda fid_box: int(fid_box[0]))
        obj_fid_boxes = sorted(obj_traj.items(), key=lambda fid_box: int(fid_box[0]))
        stt_fid = int(sbj_fid_boxes[0][0])       # inclusive
        end_fid = int(sbj_fid_boxes[-1][0]) + 1  # exclusive
        format_rela['duration'] = [stt_fid, end_fid]

        format_sbj_traj = [fid_box[1] for fid_box in sbj_fid_boxes]
        format_obj_traj = [fid_box[1] for fid_box in obj_fid_boxes]
        format_rela['sub_traj'] = format_sbj_traj
        format_rela['obj_traj'] = format_obj_traj
        format_relas.append(format_rela)
    return format_relas


def main(config):
    print('========= testing =========')

    # load model
    print('Loading models ...')
    spa_model = load_model(config, 'spa')
    lan_model = load_model(config, 'lan')
    if config['use_gpu']:
        spa_model.cuda()
        lan_model.cuda()
    spa_model.eval()
    lan_model.eval()

    # load trajectory detection results
    print('Loading trajectory detection results ...')
    detection_result_path = config['object_trajectory_path']
    all_dets = load_json(detection_result_path)
    if 'results' in all_dets:
        all_dets = all_dets['results']

    # load dataset
    ds = VidOR(config['dataset_root'])

    result_root = config['result_root']
    if not os.path.exists(result_root):
        os.makedirs(result_root)

    # start testing
    MAX_PER_VIDEO = 2000
    ATOMIC_SEGMENT_LENGTH = 30

    for vid_cnt, pid_vid in enumerate(sorted(all_dets.keys())):
        # for each video
        pid, vid = pid_vid.split('/')
        save_path = os.path.join(result_root, vid + '.json')
        if os.path.exists(save_path):
            with open(save_path) as f:
                json.load(f)
            continue

        # relation generation for video
        vid_rlts = []
        vid_dets = all_dets[pid_vid]
        for sid, sbj in enumerate(vid_dets):
            for oid, obj in enumerate(vid_dets):
                if sid == oid: continue
                rela_cand_segs = generate_relation_segments(sbj, obj, seg_len=ATOMIC_SEGMENT_LENGTH)
                rela_cand_segs = predict_predicate(ds, rela_cand_segs, spa_model, lan_model, use_gpu=config['use_gpu'])
                rela_instances = greedy_association(rela_cand_segs)
                vid_rlts += rela_instances

        vid_rlts = filter(vid_rlts, MAX_PER_VIDEO)
        vid_rlts = format(vid_rlts)
        print('[%d/%d]: N-instance %d' % (vid_cnt + 1, len(all_dets), len(vid_rlts)))
        save_json({vid: vid_rlts}, save_path)

    if config['single_result_file']:
        print('Generating single output file ...')
        time.sleep(2)
        merge(result_root, config['result_path'])

    print('========= done =========')


if __name__ == '__main__':
    config_path = 'hoia_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.load(f)
    main(config)