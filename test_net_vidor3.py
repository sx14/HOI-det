import os
import copy
import json
from collections import defaultdict
from math import log, e

import pickle
import yaml
import numpy as np
import torch
from torch.autograd import Variable

import _init_paths
import os
import pickle
import numpy as np
import argparse
import pprint
import time
import cv2
import torch
from torch.autograd import Variable
import torch.optim as optim

import torchvision.transforms as transforms
from scipy.misc import imread
from roi_data_layer.roibatchLoader import gen_spatial_map
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
from generate_VCOCO_detection import generate_VCOCO_detection_and_eval
from datasets.pose_map import gen_part_boxes, est_part_boxes
from datasets.vidor_hoid import vidor_hoid
import pdb


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='vidor_hoid_mini', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfgs/vgg16.yml', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res101', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models',
                        default="weights")
    parser.add_argument('--output_dir', dest='output_dir',
                        help='directory to load images for demo',
                        default="output")
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        default=True,
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--parallel_type', dest='parallel_type',
                        help='which part of model to parallel, 0: all, 1: model before roi pooling',
                        default=0, type=int)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=6, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=18131, type=int)

    args = parser.parse_args()
    return args


def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)


def load_model(vrb_classes):
    args = parse_args()

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.USE_GPU_NMS = args.cuda
    np.random.seed(cfg.RNG_SEED)

    input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
    load_name = os.path.join(input_dir, 'ho-rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

    # initilize the network here.
    if args.net == 'vgg16':
        fasterRCNN = vgg16(vrb_classes, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(vrb_classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fasterRCNN = resnet(vrb_classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        fasterRCNN = resnet(vrb_classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
    else:
        print("Network is not defined")

    fasterRCNN.create_architecture()

    print("Loading checkpoint %s ..." % (load_name))
    if args.cuda > 0:
        checkpoint = torch.load(load_name)
    else:
        checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']

    if args.cuda > 0:
        cfg.CUDA = True

    if args.cuda > 0:
        fasterRCNN.cuda()
    fasterRCNN.eval()

    return fasterRCNN


class VidOR_HOID:

    def __init__(self, ds_name, data_root):
        self.data_root = data_root
        self.obj_cates, self.pre_cates = self.load_cates()
        self.sbj_cates = {'adult', 'child', 'baby'}
        self.obj_cate2idx = {cate: idx for idx, cate in enumerate(self.obj_cates)}
        self.pre_cate2idx = {cate: idx for idx, cate in enumerate(self.pre_cates)}
        self.obj_vecs = self.load_object_vectors()
        self.frame_root = os.path.join(data_root, 'Data', 'VID', 'val')
        self.cache_root = os.path.join(data_root, 'tmp')
        self.dataset_name = ds_name
        self.sbj2pre_mask = None
        self.obj2pre_mask = None
        self.SEG_LEN = 10
        self._load_annotations(os.path.join(data_root, 'anno_with_pose', 'training'))

    def _gen_positive_instances(self, org_insts, pkg_id, vid_id):
        insts = []
        seg_len = self.SEG_LEN
        for org_inst in org_insts:
            sbj_tid = org_inst['subject_tid']
            obj_tid = org_inst['object_tid']
            stt_frm_idx = org_inst['begin_fid']
            end_frm_idx = org_inst['end_fid']
            pre_cate = org_inst['predicate']

            for frm_idx in range(int(stt_frm_idx / seg_len) * seg_len,
                                 int(end_frm_idx / seg_len) * seg_len, seg_len):
                seg_stt_frm_idx = frm_idx
                seg_end_frm_idx = frm_idx + seg_len
                insts.append({
                    'pkg_id': pkg_id,
                    'vid_id': vid_id,
                    'stt_fid': seg_stt_frm_idx,
                    'end_fid': seg_end_frm_idx,
                    'sbj_tid': sbj_tid,
                    'obj_tid': obj_tid,
                    'sce_tid': -1,
                    'pre_cate': self.pre_cate2idx[pre_cate]})
        return insts

    @staticmethod
    def _load_trajectories(org_trajs, vid_info):
        vid_len = vid_info['frame_count']
        tid2traj = {}
        tid2dur = {}    # [stt_fid, end_fid)

        for frm_idx, frm_dets in enumerate(org_trajs):
            for det in frm_dets:
                tid = det['tid']
                box = [det['bbox']['xmin'],
                       det['bbox']['ymin'],
                       det['bbox']['xmax'],
                       det['bbox']['ymax']]
                if tid not in tid2traj:
                    tid2traj[tid] = [[-1] * 4] * vid_len
                if tid not in tid2dur:
                    tid2dur[tid] = [frm_idx, frm_idx+1]

                tid2traj[tid][frm_idx] = box
                tid2dur[tid][1] = frm_idx + 1

        for tid in tid2traj:
            tid2traj[tid] = np.array(tid2traj[tid]).astype(np.int)
        return tid2traj, tid2dur

    def _load_annotations(self, anno_root):

        cache_path = os.path.join(self.cache_root, '%s_%s_anno_cache.bin' % (self.dataset_name, 'train'))
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                data_cache = pickle.load(f)

            self.all_trajs = data_cache['all_trajs']
            self.all_insts = data_cache['all_insts']
            self.all_vid_info = data_cache['all_vid_info']
            self.all_traj_cates = data_cache['all_traj_cates']
            self.all_inst_count = data_cache['all_inst_count']
            self.sbj2pre_mask = data_cache['sbj2pre_mask']
            self.obj2pre_mask = data_cache['obj2pre_mask']
            return

        print('Processing annotations ...')
        time.sleep(2)
        self.all_vid_info = {}
        self.all_trajs = {}
        self.all_traj_cates = {}
        self.all_insts = []
        self.all_inst_count = []

        from tqdm import tqdm
        for pkg_id in tqdm(sorted(os.listdir(anno_root))):
            pkg_root = os.path.join(anno_root, pkg_id)
            for vid_anno_file in sorted(os.listdir(pkg_root)):
                vid_id = vid_anno_file.split('.')[0]
                vid_anno_path = os.path.join(pkg_root, vid_anno_file)
                with open(vid_anno_path) as f:
                    vid_anno = json.load(f)
                vid_info = {'frame_count': vid_anno['frame_count'],
                            'width': vid_anno['width'],
                            'height': vid_anno['height'],
                            'vid_id': vid_id,
                            'pkg_id': pkg_id}
                tid2traj, tid2dur = self._load_trajectories(vid_anno['trajectories'], vid_info)
                tid2cate_idx = {traj_info['tid']: self.obj_cate2idx[traj_info['category']]
                                for traj_info in vid_anno['subject/objects']}
                vid_insts = self._gen_positive_instances(vid_anno['relation_instances'], pkg_id, vid_id)
                self.all_vid_info[vid_id] = vid_info
                self.all_trajs[vid_id] = tid2traj
                self.all_traj_cates[vid_id] = tid2cate_idx
                self.all_insts += vid_insts
                self.all_inst_count.append(len(vid_insts))

        self._gen_pre_mask()

        if not os.path.exists(self.cache_root):
            os.makedirs(self.cache_root)

        with open(cache_path, 'w') as f:
            pickle.dump({'all_trajs': self.all_trajs,
                         'all_insts': self.all_insts,
                         'all_vid_info': self.all_vid_info,
                         'all_traj_cates': self.all_traj_cates,
                         'all_inst_count': self.all_inst_count,
                         'sbj2pre_mask': self.sbj2pre_mask,
                         'obj2pre_mask': self.obj2pre_mask}, f)
        print('%s created' % cache_path)

    def category_num(self, target):
        if target == 'predicate':
            return len(self.pre_cates)
        elif target == 'object':
            return len(self.obj_cates)
        else:
            return -1

    def is_subject(self, cate):
        if isinstance(cate, int):
            return self.obj_cates[cate] in self.sbj_cates
        else:
            return cate in self.sbj_cates

    def _gen_pre_mask(self):
        sbj2pre_mask = np.zeros((self.category_num('object'), self.category_num('predicate')))
        obj2pre_mask = np.zeros((self.category_num('object'), self.category_num('predicate')))
        sbj2pre_mask[:, 0] = 1
        obj2pre_mask[:, 0] = 1

        for inst in self.all_insts:
            vid = inst['vid_id']
            sbj_tid = inst['sbj_tid']
            obj_tid = inst['obj_tid']
            sbj_cate = self.all_traj_cates[vid][sbj_tid]
            obj_cate = self.all_traj_cates[vid][obj_tid]
            pre_cate = inst['pre_cate']
            obj2pre_mask[obj_cate, pre_cate] = 1
            sbj2pre_mask[sbj_cate, pre_cate] = 1

        self.obj2pre_mask = obj2pre_mask
        self.sbj2pre_mask = sbj2pre_mask

    def load_cates(self):
        # 0 base
        obj_label_path = os.path.join(self.data_root, 'object_labels.txt')
        with open(obj_label_path) as f:
            obj_cates = [line.strip() for line in f.readlines()]

        # 1 base
        pre_label_path = os.path.join(self.data_root, 'predicate_labels.txt')
        with open(pre_label_path) as f:
            pre_cates = ['__no_interaction__'] + [line.strip() for line in f.readlines()]

        return obj_cates, pre_cates

    def load_object_vectors(self):
        object_vector_path = os.path.join(self.data_root, 'object_vectors.mat')
        with open(object_vector_path) as f:
            return pickle.load(f)


class Tester:

    def __init__(self, dataset, model, all_dets, seg_len,
                 max_per_video, output_root, use_gpu=True):
        self.dataset = dataset
        self.model = model
        self.all_dets = all_dets
        self.seg_len = seg_len
        self.max_per_video = max_per_video
        self.output_root = output_root
        self.use_gpu = use_gpu

        if not os.path.exists(output_root):
            os.makedirs(output_root)

        if use_gpu:
            model.cuda()

    def generate_relation_segments(self, sbjs, objs, seg_id, frame):

        rela_segments = []

        for sbj in sbjs:
            for obj in objs:
                if sbj['id'] == obj['id']:
                    continue

                sbj_xmin = sbj['xmin']
                sbj_ymin = sbj['ymin']
                sbj_xmax = sbj['xmax']
                sbj_ymax = sbj['ymax']
                sbj_scr = sbj['score']
                sbj_cls = sbj['category']

                obj_xmin = obj['xmin']
                obj_ymin = obj['ymin']
                obj_xmax = obj['xmax']
                obj_ymax = obj['ymax']
                obj_scr = obj['score']
                obj_cls = obj['category']

                seg_sbj_traj = {}
                seg_obj_traj = {}
                seg_sbj_traj['%06d' % (seg_id * self.seg_len)] = [sbj_xmin, sbj_ymin, sbj_xmax, sbj_ymax]
                seg_obj_traj['%06d' % (seg_id * self.seg_len)] = [obj_xmin, obj_ymin, obj_xmax, obj_ymax]

                seg = {'seg_id': seg_id,
                       'sbj_traj': seg_sbj_traj,
                       'obj_traj': seg_obj_traj,
                       'sbj_pose_traj': None,
                       'sbj_cls': sbj_cls,
                       'obj_cls': obj_cls,
                       'sbj_scr': sbj_scr,
                       'obj_scr': obj_scr,
                       'vid_h': frame.shape[0],
                       'vid_w': frame.shape[1],
                       'instance_id': -1,
                       'instance_len': 1,
                       'connected': False}
                rela_segments.append(seg)
        return rela_segments

    def ext_language_feat(self, rela_segs):
        lan_feat = np.zeros((len(rela_segs), self.dataset.obj_vecs.shape[1] * 2))
        for i, rela_seg in enumerate(rela_segs):
            sbj_cate_idx = self.dataset.obj_cate2idx[rela_seg['sbj_cls']]
            obj_cate_idx = self.dataset.obj_cate2idx[rela_seg['obj_cls']]
            sbj_cate_vec = self.dataset.obj_vecs[sbj_cate_idx]
            obj_cate_vec = self.dataset.obj_vecs[obj_cate_idx]
            lan_feat[i] = np.concatenate((sbj_cate_vec, obj_cate_vec))
        return lan_feat

    def predict_predicate(self, rela_segs, frame_path):
        if len(rela_segs) == 0:
            return rela_segs

        im_data = torch.FloatTensor(1)
        im_info = torch.FloatTensor(1)
        num_hois = torch.LongTensor(1)
        hboxes = torch.FloatTensor(1)
        oboxes = torch.FloatTensor(1)
        iboxes = torch.FloatTensor(1)
        pboxes = torch.FloatTensor(1)
        sboxes = torch.FloatTensor(1)
        hoi_classes = torch.FloatTensor(1)
        bin_classes = torch.FloatTensor(1)
        hoi_masks = torch.FloatTensor(1)
        spa_maps = torch.FloatTensor(1)
        obj_vecs = torch.FloatTensor(1)

        # ship to cuda
        if self.use_gpu > 0:
            im_data = im_data.cuda()
            im_info = im_info.cuda()
            num_hois = num_hois.cuda()
            hboxes = hboxes.cuda()
            oboxes = oboxes.cuda()
            iboxes = iboxes.cuda()
            pboxes = pboxes.cuda()
            sboxes = sboxes.cuda()
            hoi_classes = hoi_classes.cuda()
            bin_classes = bin_classes.cuda()
            hoi_masks = hoi_masks.cuda()
            spa_maps = spa_maps.cuda()
            obj_vecs = obj_vecs.cuda()

        with torch.no_grad():
            im_data = Variable(im_data)
            im_info = Variable(im_info)
            num_hois = Variable(num_hois)
            hboxes = Variable(hboxes)
            oboxes = Variable(oboxes)
            iboxes = Variable(iboxes)
            pboxes = Variable(pboxes)
            sboxes = Variable(sboxes)
            hoi_classes = Variable(hoi_classes)
            bin_classes = Variable(bin_classes)
            hoi_masks = Variable(hoi_masks)
            spa_maps = Variable(spa_maps)
            obj_vecs = Variable(obj_vecs)

        im_in = np.array(imread(frame_path))
        if len(im_in.shape) == 2:
            im_in = im_in[:, :, np.newaxis]
            im_in = np.concatenate((im_in, im_in, im_in), axis=2)
        im_in = im_in[:, :, ::-1]  # rgb -> bgr
        blobs, im_scales = _get_image_blob(im_in)
        im_blob = blobs
        im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

        im_data_pt = torch.from_numpy(im_blob)
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)
        im_info_pt = torch.from_numpy(im_info_np)

        im_data.data.resize_(im_data_pt.size()).copy_(im_data_pt)
        im_info.data.resize_(im_info_pt.size()).copy_(im_info_pt)

        data_height = im_in.shape[0]
        data_width = im_in.shape[1]
        gt_sboxes = [
            [0, 0, data_width, data_height],
            [0, 0, data_width / 2, data_height / 2],
            [data_width / 2, 0, data_width, data_height / 2],
            [0, data_height / 2, data_width / 2, data_height],
            [data_width / 2, data_height / 2, data_width, data_height]]
        sboxes_raw = np.array(gt_sboxes)
        sboxes_raw = sboxes_raw[np.newaxis, :, :]

        hboxes_raw = np.zeros((500, 4))
        oboxes_raw = np.zeros((500, 4))
        iboxes_raw = np.zeros((500, 4))
        pboxes_raw = np.zeros((500, 6, 4))
        spa_maps_raw = np.zeros((500, 2, 64, 64))
        obj_vecs_raw = np.zeros((500, 300))
        pre_masks_raw = np.ones((500, 30))
        num_cand = 0

        for rela_seg in rela_segs:
            hbox = np.array(rela_seg['sbj_traj'][min(rela_seg['sbj_traj'])]).reshape((1, 4)).astype(np.float)
            obox = np.array(rela_seg['obj_traj'][min(rela_seg['obj_traj'])]).reshape((1, 4)).astype(np.float)
            ibox = np.array([min(hbox[0][0], obox[0][0]), min(hbox[0][1], obox[0][1]),
                             max(hbox[0][2], obox[0][2]), max(hbox[0][3], obox[0][3])]).reshape((1, 4)).astype(np.float)
            # raw_kps = rela_seg['sbj_pose_traj'][min(rela_seg['sbj_pose_traj'])]
            raw_kps = None
            if raw_kps != None and len(raw_kps) == 51:
                key_points = np.array(raw_kps).reshape((17, 3))
                pbox = gen_part_boxes(hbox[0].tolist(), key_points, im_in.shape[:2])
            else:
                pbox = est_part_boxes(hbox[0].tolist())

            pbox = np.array(pbox)
            pbox = pbox.reshape((6, 4))[np.newaxis, :, :]
            spa_map_raw = gen_spatial_map(hbox[0], obox[0])
            spa_maps_raw[num_cand] = spa_map_raw

            obj_class_id = self.dataset.obj_cate2idx[rela_seg['obj_cls']]
            obj_vec_raw = self.dataset.obj_vecs[obj_class_id]
            obj_vecs_raw[num_cand] = obj_vec_raw
            pre_masks_raw[num_cand] = pre_masks_raw[num_cand] * self.dataset.obj2pre_mask[obj_class_id]
            sbj_class_id = self.dataset.obj_cate2idx[rela_seg['sbj_cls']]
            pre_masks_raw[num_cand] = pre_masks_raw[num_cand] * self.dataset.sbj2pre_mask[sbj_class_id]

            hboxes_raw[num_cand] = hbox
            oboxes_raw[num_cand] = obox
            iboxes_raw[num_cand] = ibox
            pboxes_raw[num_cand] = pbox
            num_cand += 1

        hboxes_raw1 = hboxes_raw[np.newaxis, :num_cand]
        oboxes_raw1 = oboxes_raw[np.newaxis, :num_cand]
        iboxes_raw1 = iboxes_raw[np.newaxis, :num_cand]
        pboxes_raw1 = pboxes_raw[np.newaxis, :num_cand]
        pre_masks_raw1 = pre_masks_raw[np.newaxis, :num_cand]

        spa_maps_raw1 = spa_maps_raw[np.newaxis, :num_cand]
        obj_vecs_raw1 = obj_vecs_raw[np.newaxis, :num_cand]

        hboxes_t = torch.from_numpy(hboxes_raw1 * im_scales[0])
        oboxes_t = torch.from_numpy(oboxes_raw1 * im_scales[0])
        iboxes_t = torch.from_numpy(iboxes_raw1 * im_scales[0])
        pboxes_t = torch.from_numpy(pboxes_raw1 * im_scales[0])
        sboxes_t = torch.from_numpy(sboxes_raw * im_scales[0])
        spa_maps_t = torch.from_numpy(spa_maps_raw1)
        obj_vecs_t = torch.from_numpy(obj_vecs_raw1)

        hboxes.data.resize_(hboxes_t.size()).copy_(hboxes_t)
        oboxes.data.resize_(oboxes_t.size()).copy_(oboxes_t)
        iboxes.data.resize_(iboxes_t.size()).copy_(iboxes_t)
        pboxes.data.resize_(pboxes_t.size()).copy_(pboxes_t)
        sboxes.data.resize_(sboxes_t.size()).copy_(sboxes_t)
        spa_maps.data.resize_(spa_maps_t.size()).copy_(spa_maps_t)
        obj_vecs.data.resize_(obj_vecs_t.size()).copy_(obj_vecs_t)

        with torch.no_grad():
            probs, _, _, _ = model(im_data, im_info,
                                   hboxes,
                                   oboxes,
                                   iboxes,
                                   pboxes,
                                   sboxes,
                                   hoi_classes,
                                   bin_classes,
                                   hoi_masks,
                                   spa_maps,
                                   obj_vecs,
                                   num_hois)

        if self.use_gpu:
            probs = probs.cpu()
        probs = probs.data.numpy()[0]
        all_rela_segs = []

        for i in range(probs.shape[0]):
            rela_seg = rela_segs[i]
            obj_cls = rela_seg['obj_cls']
            obj_hoi_inds = [hoi_ind for hoi_ind, hoi in enumerate(dataset.hoi_classes) if obj_cls in hoi]
            rela_probs = probs[i]
            for hoi_idx in obj_hoi_inds:
                rela_seg_copy = copy.deepcopy(rela_seg)
                pred_hoi_scr = rela_probs[hoi_idx]
                pred_hoi = self.dataset.hoi_classes[hoi_idx]
                rela_seg_copy['pre_cls'] = pred_hoi
                rela_seg_copy['hoi_scr'] = pred_hoi_scr
                all_rela_segs.append(rela_seg_copy)

        return all_rela_segs

    @staticmethod
    def greedy_association(rela_cand_segments):

        def cal_iou(box1, box2):
            xmin1, ymin1, xmax1, ymax1 = box1
            xmin2, ymin2, xmax2, ymax2 = box2
            xmini = max(xmin1, xmin2)
            ymini = max(ymin1, ymin2)
            xmaxi = min(xmax1, xmax2)
            ymaxi = min(ymax1, ymax2)

            area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
            area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
            areai = max((xmaxi - xmini), 0) * max((ymaxi - ymini), 0)
            return areai * 1.0 / (area1 + area2 - areai)

        if len(rela_cand_segments) == 0:
            return []

        last_seg_id = -2
        next_instance_id = 0
        seg_ids = sorted(rela_cand_segments.keys())
        for seg_idx in range(len(seg_ids)-1):
            curr_seg_id = seg_ids[seg_idx]
            next_seg_id = seg_ids[seg_idx+1]

            curr_segments = rela_cand_segments[curr_seg_id]
            next_segments = rela_cand_segments[next_seg_id]

            if curr_seg_id - last_seg_id > 1:
                # init curr
                for i in range(len(curr_segments)):
                    curr_segments[i]['instance_id'] = next_instance_id
                    curr_segments[i]['connected'] = True
                    next_instance_id += 1

            last_seg_id = curr_seg_id

            if next_seg_id - curr_seg_id > 1:
                continue

            for curr_seg in curr_segments:
                best_next_id = -1
                best_next_iou = -1

                for next_id, next_seg in enumerate(next_segments):

                    if next_seg['connected']:
                        continue

                    if curr_seg['sbj_cls'] == next_seg['sbj_cls'] and \
                            curr_seg['obj_cls'] == next_seg['obj_cls'] and \
                            curr_seg['pre_cls'] == next_seg['pre_cls']:
                        curr_sbj_box = curr_seg['sbj_traj'][min(curr_seg['sbj_traj'].keys())]
                        curr_obj_box = curr_seg['obj_traj'][min(curr_seg['obj_traj'].keys())]
                        next_sbj_box = next_seg['sbj_traj'][min(next_seg['sbj_traj'].keys())]
                        next_obj_box = next_seg['obj_traj'][min(next_seg['obj_traj'].keys())]
                        iou = min(cal_iou(curr_sbj_box, next_sbj_box), cal_iou(curr_obj_box, next_obj_box))
                        if iou > best_next_iou:
                            best_next_iou = iou
                            best_next_id = next_id

                if best_next_iou > 0.6:
                    next_seg = next_segments[best_next_id]
                    next_seg['instance_id'] = curr_seg['instance_id']
                    next_seg['connected'] = True

            for next_seg in next_segments:
                if not next_seg['connected']:
                    next_seg['instance_id'] = next_instance_id
                    next_seg['connected'] = True
                    next_instance_id += 1

        rela_instances = {}
        for seg_id in seg_ids:
            rela_segments = rela_cand_segments[seg_id]
            for rela_seg in rela_segments:
                instance_id = rela_seg['instance_id']
                if instance_id not in rela_instances:
                    rela_instances[instance_id] = rela_seg
                else:
                    rela_instance = rela_instances[instance_id]
                    rela_instance['sbj_traj'].update(rela_seg['sbj_traj'])
                    rela_instance['obj_traj'].update(rela_seg['obj_traj'])
                    rela_instance['pre_scr'] += rela_seg['pre_scr']
                    rela_instance['instance_len'] += 1

        for instance_id in rela_instances:
            rela_instance = rela_instances[instance_id]
            rela_instance['pre_scr'] = rela_instance['pre_scr'] / rela_instance['instance_len']

        return [rela_inst for rela_inst in rela_instances.values() if rela_inst['instance_len'] >= 5]

    @staticmethod
    def filter(rela_cands, max_per_video):
        rela_cands = [rela_cand for rela_cand in rela_cands if rela_cand['pre_cls'] != '__no_interaction__']
        for rela_cand in rela_cands:
            rela_cand['score'] = rela_cand['hoi_scr']
        sorted_cands = sorted(rela_cands, key=lambda rela: rela['score'], reverse=True)
        return sorted_cands[:max_per_video]

    @staticmethod
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
            stt_fid = int(sbj_fid_boxes[0][0])          # inclusive
            end_fid = int(sbj_fid_boxes[-1][0]) + 1     # exclusive
            format_rela['duration'] = [stt_fid, end_fid]

            format_sbj_traj = [fid_box[1] for fid_box in sbj_fid_boxes]
            format_obj_traj = [fid_box[1] for fid_box in obj_fid_boxes]
            format_rela['sub_traj'] = format_sbj_traj
            format_rela['obj_traj'] = format_obj_traj
            format_relas.append(format_rela)
        return format_relas

    def get_fid2dets(self, vid_trajs):
        # x1, y1, x2, y2, cls, conf
        fid2dets = defaultdict(list)
        for traj in vid_trajs:
            cate = traj['category']
            cate_id = self.dataset.obj_cate2idx[cate]
            conf = traj['score']
            for fid in traj['trajectory']:
                det = traj['trajectory'][fid] + [cate_id, conf]
                fid2dets[fid].append(det)
        return fid2dets

    def run_video(self, vid_dets, vid_frm_root):

        def get_sbjs_and_objs(ds, dets):
            for i in range(len(dets)):
                dets[i]['id'] = i
            sbjs = [det for det in dets if ds.is_subject(det['category'])]
            objs = dets
            return sbjs, objs

        # collect relation segments
        frame0_path = os.path.join(vid_frm_root, '000000.JPEG')
        frame0 = cv2.imread(frame0_path)
        all_cand_segs = defaultdict(list)
        for fid_str in sorted(vid_dets.keys()):
            frm_dets = vid_dets[fid_str]
            sbjs, objs = get_sbjs_and_objs(self.dataset, frm_dets)
            seg_id = int(int(fid_str) / self.seg_len)
            rela_cand_segs = self.generate_relation_segments(sbjs, objs, seg_id, frame0)
            all_cand_segs[seg_id] = rela_cand_segs

        # predicate classification
        for seg_id in sorted(all_cand_segs.keys()):
            frame_path = os.path.join(vid_frm_root, '%06d.JPEG' % (seg_id * self.seg_len))
            cand_segs = all_cand_segs[seg_id]                      # list(seg)
            cand_segs = self.predict_predicate(cand_segs, frame_path)   # list(list(seg))
            all_cand_segs[seg_id] = cand_segs

        # association
        vid_relas = []
        rela_instances = self.greedy_association(all_cand_segs)
        vid_relas += rela_instances

        return vid_relas

    @staticmethod
    def load_toi_feat(video_feat_root):
        tid2feat = {}
        feat_files = os.listdir(video_feat_root)
        for feat_file in sorted(feat_files):
            tid = feat_file.split('.')[0]
            feat_path = os.path.join(video_feat_root, feat_file)
            with open(feat_path) as f:
                feat = pickle.load(f)
            tid2feat[int(tid)] = feat
        return tid2feat

    def run(self):
        vid_num = len(self.all_dets)
        for i, pid_vid in enumerate(sorted(self.all_dets.keys())):
            pid, vid = pid_vid.split('/')
            print('[%d/%d] %s' % (i+1, vid_num, vid))

            vid_save_path = os.path.join(self.output_root, vid + '.json')
            if os.path.exists(vid_save_path):
                with open(vid_save_path) as f:
                    json.load(f)
                continue
            vid_frm_root = os.path.join(self.dataset.frame_root, pid, vid)
            vid_relas = self.run_video(self.all_dets[pid_vid], vid_frm_root)
            vid_relas = self.filter(vid_relas, self.max_per_video)
            vid_relas = self.format(vid_relas)

            with open(vid_save_path, 'w') as f:
                json.dump({vid: vid_relas}, f)


if __name__ == '__main__':
    dataset_name = 'vidor_hoid_mini'
    dataset_root = os.path.join('data', dataset_name)

    # load DET/GT trajectories
    print('Loading object detections ...')
    test_det_path = 'data/%s/%s' % (dataset_name, 'object_detections_val.json')

    with open(test_det_path) as f:
        test_dets = json.load(f)
    dataset = VidOR_HOID(dataset_name, dataset_root)

    # load model
    print('Loading model ...')
    model = load_model(dataset.pre_cates)

    # init tester
    print('---- Testing start ----')
    output_root = os.path.join('output', dataset_name)
    tester = Tester(dataset, model, test_dets, 1, 2000, output_root)
    tester.run()
