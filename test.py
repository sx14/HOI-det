# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient

from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='hico', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res101',
                      default='vgg16', type=str)
  parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save models', default="output",
                      type=str)
  parser.add_argument('--nw', dest='num_workers',
                      help='number of workers to load data',
                      default=0, type=int)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      default=True,
                      action='store_true')
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')                      
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether to perform class_agnostic bbox regression',
                      action='store_true')

  args = parser.parse_args()
  return args



def test(fasterRCNN):
  fasterRCNN.eval()

  args = parse_args()

  if args.dataset == "hico":
      args.imdb_name = "hico_2016_train"
      args.imdbval_name = "hico_2016_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  else:
      print('Only support HICO-DET dataset now.')

  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  np.random.seed(cfg.RNG_SEED)

  #torch.backends.cudnn.benchmark = True
  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.
  cfg.TRAIN.USE_FLIPPED = True
  cfg.USE_GPU_NMS = args.cuda
  imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, False)

  print('{:d} roidb entries'.format(len(roidb)))

  output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)


  dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                           imdb.num_classes, training=False, normalize=False)

  dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)

  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_hois = torch.LongTensor(1)
  hboxes = torch.FloatTensor(1)
  oboxes = torch.FloatTensor(1)
  iboxes = torch.FloatTensor(1)
  hoi_classes = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_hois = num_hois.cuda()
    hboxes = hboxes.cuda()
    oboxes = oboxes.cuda()
    iboxes = iboxes.cuda()
    hoi_classes = hoi_classes.cuda()

  # make variable
  im_data = Variable(im_data)
  im_info = Variable(im_info)
  num_hois = Variable(num_hois)
  hboxes = Variable(hboxes)
  oboxes = Variable(oboxes)
  iboxes = Variable(iboxes)
  hoi_classes = Variable(hoi_classes)

  if args.cuda:
    cfg.CUDA = True

  diff_sum = 0.0
  data_iter = iter(dataloader)
  for id, data in enumerate(data_iter):
    im_data.data.resize_(data[0].size()).copy_(data[0])
    im_info.data.resize_(data[1].size()).copy_(data[1])
    hboxes.data.resize_(data[2].size()).copy_(data[2])
    oboxes.data.resize_(data[3].size()).copy_(data[3])
    iboxes.data.resize_(data[4].size()).copy_(data[4])
    hoi_classes.resize_(data[5].size()).copy_(data[5])
    num_hois.data.resize_(data[6].size()).copy_(data[6])

    cls_prob, RCNN_loss_cls = fasterRCNN(im_data, im_info, hboxes, oboxes, iboxes, hoi_classes, num_hois)

    img_avg_label_num = np.sum(hoi_classes) * 1.0 / hoi_classes.shape[0]
    img_avg_diff = avg_diff(cls_prob, hoi_classes)
    diff_sum += img_avg_diff
    print('%d: %.4f/%.4f' % (id, img_avg_diff, img_avg_label_num))

  print('Overall diff: %.4f' % (diff_sum / len(roidb)))

  fasterRCNN.train()


def avg_diff(pred_probs, gt_classes):

    def diff(pred, gt):
        t = pred != gt
        return t.sum()

    pred_probs = pred_probs.cpu().data.numpy()
    gt_classes = gt_classes.cpu().data.numpy()
    pred_probs[pred_probs < 0.5] = 0
    pred_probs[pred_probs >= 0.5] = 1

    diff_sum = 0.0
    for i in range(pred_probs.shape[0]):
        diff_sum += diff(pred_probs[i, :], gt_classes[i, :])

    return diff_sum / pred_probs.shape[0]

