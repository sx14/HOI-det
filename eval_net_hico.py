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
from roi_data_layer.pose_map import gen_pose_obj_map, gen_pose_obj_map1
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
from generate_HICO_detection import generate_HICO_detection, org_obj2hoi

from datasets.hico2 import hico2
from datasets.hico2 import refine_human_box_with_skeleton
from datasets.pose_map import est_part_boxes, gen_part_boxes, gen_part_boxes1
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
                      default=6, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=91451, type=int)

  args = parser.parse_args()
  return args



def _get_image_blob(im, dp):
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

  # dp_orig = dp.astype(np.float32, copy=True)
  # dp_orig -= cfg.DEPTH_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  im_scales = []
  processed_ims = []
  # processed_dps = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    # dp = cv2.resize(dp_orig, None, None, fx=im_scale, fy=im_scale,
    #         interpolation=cv2.INTER_LINEAR)

    im_scales.append(im_scale)
    processed_ims.append(im)
    # processed_dps.append(dp)

  # Create a blob to hold the input images
  im_blob = im_list_to_blob(processed_ims, 3)
  # dp_blob = im_list_to_blob(processed_dps, 7)
  dp_blob = np.zeros((1,1,1,1))
  return im_blob, dp_blob, im_scales


if __name__ == '__main__':
  output_dir = 'output/hico_full'
  output_path = output_dir+'/all_hoi_detections.pkl'
  # print('Saving results ...')
  # with open(output_path, 'rb') as f:
  #     all_results = pickle.load(f)
  #
  # generate_HICO_detection(all_results, output_dir, 1.0, 0.0)
  os.chdir('eval_hico')
  os.system('bash source ~/.bashrc')
  os.system('bash matlab -nodesktop -nosplash -r "Generate_detection ' + '../output/hico_full' + '/;quit;"')
  os.chdir('..')



