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
from generate_HICO_detection import generate_HICO_detection
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
      generate_HICO_detection(output_path, 'output/results', 1.0, 0.0)
      os.chdir('benchmark')
      os.system('matlab -nodesktop -nosplash -r "Generate_detection ' + '../output/results/' + '/;quit;"')
      exit(0)

  print('Loading object detections ...')
  det_path = 'data/hico/Test_Faster_RCNN_R-50-PFN_2x_HICO_DET.pkl'
  with open(det_path) as f:
      det_db = pickle.load(f)

  input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
  if not os.path.exists(input_dir):
    raise Exception('There is no input directory for loading network from ' + input_dir)
  load_name = os.path.join(input_dir,
    'ho_spa_rcnn3_lf_no_nis_3b_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

  pascal_classes = ['1'] * 600

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
  hoi_classes = torch.FloatTensor(1)
  bin_classes = torch.FloatTensor(1)
  hoi_masks = torch.FloatTensor(1)
  spa_maps = torch.FloatTensor(1)


  # ship to cuda
  if args.cuda > 0:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_hois = num_hois.cuda()
    hboxes = hboxes.cuda()
    oboxes = oboxes.cuda()
    iboxes = iboxes.cuda()
    hoi_classes = hoi_classes.cuda()
    bin_classes = bin_classes.cuda()
    hoi_masks = hoi_masks.cuda()
    spa_maps = spa_maps.cuda()

  # make variable
  with torch.no_grad():
      im_data = Variable(im_data)
      im_info = Variable(im_info)
      num_hois = Variable(num_hois)
      hboxes = Variable(hboxes)
      oboxes = Variable(oboxes)
      iboxes = Variable(iboxes)
      hoi_classes = Variable(hoi_classes)
      bin_classes = Variable(bin_classes)
      hoi_masks = Variable(hoi_masks)
      spa_maps = Variable(spa_maps)

  if args.cuda > 0:
    cfg.CUDA = True

  if args.cuda > 0:
    fasterRCNN.cuda()

  fasterRCNN.eval()

  start = time.time()
  human_thres = 0.8
  object_thres = 0.3

  num_images = len(det_db)
  print('Loaded Photo: {} images.'.format(num_images))

  all_results = {}
  image_path_template = 'data/hico/images/test2015/HICO_test2015_%s.jpg'
  for i, im_id in enumerate(det_db):
      print('test [%d/%d]' % (i + 1, num_images))
      im_file = image_path_template % str(im_id).zfill(8)
      im_in = np.array(imread(im_file))
      if len(im_in.shape) == 2:
          im_in = im_in[:, :, np.newaxis]
          im_in = np.concatenate((im_in, im_in, im_in), axis=2)
      im_in = im_in[:, :, ::-1]     # rgb -> bgr
      im = im_in
      blobs, im_scales = _get_image_blob(im)

      hboxes_raw = np.zeros((0, 4))
      oboxes_raw = np.zeros((0, 4))
      iboxes_raw = np.zeros((0, 4))
      spa_maps_raw = np.zeros((0, 2, 64, 64))
      obj_classes = []
      hscores = []
      oscores = []

      num_cand = 0
      im_results = []
      for human_det in det_db[im_id]:
          if (np.max(human_det[5]) > human_thres) and (human_det[1] == 'Human'):
              # This is a valid human
              hbox = np.array([human_det[2][0],
                               human_det[2][1],
                               human_det[2][2],
                               human_det[2][3]]).reshape(1, 4)

              for object_det in det_db[im_id]:
                  if (np.max(object_det[5]) > object_thres) and not (np.all(object_det[2] == human_det[2])):
                      # This is a valid object
                      obox = np.array([object_det[2][0],
                                       object_det[2][1],
                                       object_det[2][2],
                                       object_det[2][3]]).reshape(1, 4)

                      ibox = np.array([min(hbox[0, 0], obox[0, 0]),
                                       min(hbox[0, 1], obox[0, 1]),
                                       max(hbox[0, 2], obox[0, 2]),
                                       max(hbox[0, 3], obox[0, 3])]).reshape(1, 4)
                      spa_map_raw = gen_spatial_map(human_det[2], object_det[2])
                      spa_map_raw = spa_map_raw[np.newaxis, : ,: ,:]
                      spa_maps_raw = np.concatenate((spa_maps_raw, spa_map_raw))
                      hboxes_raw = np.concatenate((hboxes_raw, hbox))
                      oboxes_raw = np.concatenate((oboxes_raw, obox))
                      iboxes_raw = np.concatenate((iboxes_raw, ibox))
                      obj_classes.append(object_det[4])
                      hscores.append(human_det[5])
                      oscores.append(object_det[5])
                      num_cand += 1
      if num_cand == 0:
          all_results[im_id] = im_results
          continue

      hboxes_raw = hboxes_raw[np.newaxis, :, :]
      oboxes_raw = oboxes_raw[np.newaxis, :, :]
      iboxes_raw = iboxes_raw[np.newaxis, :, :]
      spa_maps_raw = spa_maps_raw[np.newaxis, :, :, :, :]

      hboxes_t = torch.from_numpy(hboxes_raw * im_scales[0])
      oboxes_t = torch.from_numpy(oboxes_raw * im_scales[0])
      iboxes_t = torch.from_numpy(iboxes_raw * im_scales[0])
      spa_maps_t = torch.from_numpy(spa_maps_raw)

      hboxes.data.resize_(hboxes_t.size()).copy_(hboxes_t)
      oboxes.data.resize_(oboxes_t.size()).copy_(oboxes_t)
      iboxes.data.resize_(iboxes_t.size()).copy_(iboxes_t)
      spa_maps.data.resize_(spa_maps_t.size()).copy_(spa_maps_t)

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
          hoi_prob, bin_prob, RCNN_loss_cls, RCNN_loss_bin = \
              fasterRCNN(im_data, im_info, hboxes, oboxes, iboxes, hoi_classes, bin_classes, hoi_masks, spa_maps, num_hois)

      for j in range(num_cand):
          temp = []
          temp.append(hboxes_raw[0, j])  # Human box
          temp.append(oboxes_raw[0, j])  # Object box
          temp.append(obj_classes[j])  # Object class
          temp.append(hoi_prob.cpu().data.numpy()[0, j].tolist())  # Score (600)
          temp.append(hscores[j])  # Human score
          temp.append(oscores[j])  # Object score
          temp.append(bin_prob.cpu().data.numpy()[0, j].tolist())  # binary score
          im_results.append(temp)

      all_results[im_id] = im_results

  if not os.path.exists(args.output_dir):
      os.mkdir(args.output_dir)

  print('Saving results ...')
  with open(output_path, 'wb') as f:
      pickle.dump(all_results, f)

  generate_HICO_detection(output_path, 'output/results', 1.0, 0.0)

  os.chdir('benchmark')
  os.system('matlab -nodesktop -nosplash -r "Generate_detection ' + '../output/results/' + '/;quit;"')
