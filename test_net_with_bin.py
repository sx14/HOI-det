# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import pickle
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as dset
from scipy.misc import imread
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader, gen_spatial_map
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
from generate_HICO_detection import generate_HICO_detection, org_obj2hoi

from  datasets.hico2 import hico2
import pdb

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='hico_full', type=str)
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
                      default=4, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=75265, type=int)


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


if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  cfg.USE_GPU_NMS = args.cuda

  print('Using config:')
  pprint.pprint(cfg)
  np.random.seed(cfg.RNG_SEED)

  output_path = os.path.join(args.output_dir, 'all_hoi_detections.pkl')
  if os.path.exists(output_path):
      print('Test results found!')
      print('Loading detection results ...')
      HICO = pickle.load(open(output_path, "rb"))

      generate_HICO_detection(HICO, 'output/results', 0.9, 0.1)
      os.chdir('benchmark')
      os.system('matlab -nodesktop -nosplash -r "Generate_detection ' + '../output/results/' + '/;quit;"')
      exit(0)

  # print('Loading object detections ...')
  # det_path = 'data/hico/Test_Faster_RCNN_R-50-PFN_2x_HICO_DET.pkl'
  # with open(det_path) as f:
  #     det_db = pickle.load(f)
  print('Loading interaction proposals ...')
  proposal_path = 'output/all_hoi_proposals.pkl'
  with open(proposal_path) as f:
      proposal_db = pickle.load(f)

  input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
  if not os.path.exists(input_dir):
    raise Exception('There is no input directory for loading network from ' + input_dir)
  load_name = os.path.join(input_dir,
    'ho_spa_rcnn3_lf_no_nis_3b_vrb_obj_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

  hoi_classes, obj_classes, vrb_classes, obj2int, hoi2vrb, vrb2hoi = hico2.load_hoi_classes(cfg.DATA_DIR + '/hico')
  obj2ind = dict(zip(obj_classes, range(len(obj_classes))))
  obj2vec = hico2.load_obj2vec(cfg.DATA_DIR + '/hico')
  obj2org_ind = {}
  for org_obj_id, hoi_id in enumerate(org_obj2hoi):
      if hoi_id >= 0:
        hoi_class = hoi_classes[hoi_id]
        obj2org_ind[hoi_class.object_name()] = org_obj_id

  pascal_classes = ['1'] * len(vrb_classes) 

  # initilize the network here.
  if args.net == 'vgg16':
    fasterRCNN = vgg16(pascal_classes, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res101':
    fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
    fasterRCNN = resnet(pascal_classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res152':
    fasterRCNN = resnet(pascal_classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
  else:
    print("Network is not defined")
    pdb.set_trace()

  fasterRCNN.create_architecture()

  print("Loading checkpoint %s ..." % (load_name))
  if args.cuda > 0:
    checkpoint = torch.load(load_name)
  else:
    checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
  fasterRCNN.load_state_dict(checkpoint['model'])
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']

  print('Load model successfully!')


  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_hois = torch.LongTensor(1)
  hboxes = torch.FloatTensor(1)
  oboxes = torch.FloatTensor(1)
  iboxes = torch.FloatTensor(1)
  vrb_classes = torch.FloatTensor(1)
  bin_classes = torch.FloatTensor(1)
  hoi_masks = torch.FloatTensor(1)
  spa_maps = torch.FloatTensor(1)
  obj_vecs = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda > 0:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_hois = num_hois.cuda()
    hboxes = hboxes.cuda()
    oboxes = oboxes.cuda()
    iboxes = iboxes.cuda()
    # hoi_classes = hoi_classes.cuda()
    vrb_classes = vrb_classes.cuda()
    bin_classes = bin_classes.cuda()
    hoi_masks = hoi_masks.cuda()
    spa_maps = spa_maps.cuda()
    obj_vecs = obj_vecs.cuda()

  # make variable
  with torch.no_grad():
      im_data = Variable(im_data)
      im_info = Variable(im_info)
      num_hois = Variable(num_hois)
      hboxes = Variable(hboxes)
      oboxes = Variable(oboxes)
      iboxes = Variable(iboxes)
      vrb_classes = Variable(vrb_classes)
      bin_classes = Variable(bin_classes)
      hoi_masks = Variable(hoi_masks)
      spa_maps = Variable(spa_maps)
      obj_vecs = Variable(obj_vecs)

  if args.cuda > 0:
    cfg.CUDA = True

  if args.cuda > 0:
    fasterRCNN.cuda()

  fasterRCNN.eval()

  start = time.time()
  human_thres = 0.4
  object_thres = 0.4

  num_images = len(proposal_db)
  print('Loaded Photo: {} images.'.format(num_images))

  all_results = {}
  image_path_template = 'data/hico/images/test2015/%s.jpg'
  for im_cnt, im_name in enumerate(proposal_db):
      print('test [%d/%d]' % (im_cnt + 1, num_images))
      im_id = int(im_name[-8:])
      im_file = image_path_template % im_name
      im_in = np.array(imread(im_file))
      if len(im_in.shape) == 2:
          im_in = im_in[:, :, np.newaxis]
          im_in = np.concatenate((im_in, im_in, im_in), axis=2)
      im_in = im_in[:, :, ::-1]     # rgb -> bgr
      im = im_in
      blobs, im_scales = _get_image_blob(im)

      im_results = []
      im_proposals = proposal_db[im_name]
      num_cand = len(im_proposals['human_boxes'])
      if num_cand == 0:
          all_results[im_id] = im_results
          continue

      hboxes_raw = np.array(im_proposals['human_boxes'])[:, :4]
      hscores = np.array(im_proposals['human_boxes'])[:, 4].tolist()
      oboxes_raw = np.array(im_proposals['object_boxes'])[:, :4]
      oscores = np.array(im_proposals['object_boxes'])[:, 4].tolist()
      olabels = im_proposals['object_labels']
      interactiveness = np.array(im_proposals['interactiveness'])
      iboxes_raw = np.zeros(hboxes_raw.shape)
      spa_maps_raw = np.zeros((num_cand, 2, 64, 64))
      obj_vecs_raw = np.zeros((num_cand, 300))

      obj_classes = [-1] * num_cand

      for i in range(num_cand):
          hbox = hboxes_raw[i]
          obox = oboxes_raw[i]
          ibox = np.array([min(hbox[0], obox[0]),
                           min(hbox[1], obox[1]),
                           max(hbox[2], obox[2]),
                           max(hbox[3], obox[3])])
          iboxes_raw[i] = ibox
          spa_map = gen_spatial_map(hbox, obox)
          spa_maps_raw[i] = spa_map
          obj_vecs_raw[i] = obj2vec[obj2ind[olabels[i]]]
          obj_classes[i] = obj2org_ind[olabels[i]]

      hboxes_raw = hboxes_raw[np.newaxis, :, :]
      oboxes_raw = oboxes_raw[np.newaxis, :, :]
      iboxes_raw = iboxes_raw[np.newaxis, :, :]
      spa_maps_raw = spa_maps_raw[np.newaxis, :, :, :, :]
      obj_vecs_raw = obj_vecs_raw[np.newaxis, :, :]
      hboxes_t = torch.from_numpy(hboxes_raw * im_scales[0])
      oboxes_t = torch.from_numpy(oboxes_raw * im_scales[0])
      iboxes_t = torch.from_numpy(iboxes_raw * im_scales[0])
      spa_maps_t = torch.from_numpy(spa_maps_raw)
      obj_vecs_t = torch.from_numpy(obj_vecs_raw)

      hboxes.data.resize_(hboxes_t.size()).copy_(hboxes_t)
      oboxes.data.resize_(oboxes_t.size()).copy_(oboxes_t)
      iboxes.data.resize_(iboxes_t.size()).copy_(iboxes_t)
      spa_maps.data.resize_(spa_maps_t.size()).copy_(spa_maps_t)
      obj_vecs.data.resize_(obj_vecs_t.size()).copy_(obj_vecs_t)

      assert len(im_scales) == 1, "Only single-image batch implemented"
      im_blob = blobs
      im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

      im_data_pt = torch.from_numpy(im_blob)
      im_data_pt = im_data_pt.permute(0, 3, 1, 2)
      im_info_pt = torch.from_numpy(im_info_np)

      im_data.data.resize_(im_data_pt.size()).copy_(im_data_pt)
      im_info.data.resize_(im_info_pt.size()).copy_(im_info_pt)

      det_tic = time.time()

      with torch.no_grad():
          vrb_prob, bin_prob, RCNN_loss_cls, RCNN_loss_bin = \
              fasterRCNN(im_data, im_info,
                         hboxes, oboxes, iboxes,
                         vrb_classes, bin_classes,
                         hoi_masks, spa_maps,
                         obj_vecs, num_hois)

      hoi_prob = np.zeros((1, num_cand, len(hoi_classes)))

      for j in range(num_cand):
          for vrb_id in range(vrb_prob.shape[2]):
              hoi_prob[0, j, vrb2hoi[vrb_id]] = vrb_prob[0, j, vrb_id]

      for j in range(num_cand):
          temp = []
          temp.append(hboxes_raw[0, j])  # Human box
          temp.append(oboxes_raw[0, j])  # Object box
          temp.append(obj_classes[j])    # Object class
          temp.append(hoi_prob[0, j].tolist())  # Score (600)
          temp.append(hscores[j])  # Human score
          temp.append(oscores[j])  # Object score
          temp.append([interactiveness[j], 1-interactiveness[j]])  # binary score
          im_results.append(temp)

      all_results[im_id] = im_results

  if not os.path.exists(args.output_dir):
      os.mkdir(args.output_dir)

  generate_HICO_detection(all_results, 'output/results', 0.9, 0.1)

  os.chdir('benchmark')
  os.system('matlab -nodesktop -nosplash -r "Generate_detection ' + '../output/results/' + '/;quit;"')

  print('Saving results ...')
  with open(output_path, 'wb') as f:
      pickle.dump(all_results, f)
