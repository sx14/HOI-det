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
                      default='hico_full', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res101',
                      default='res101', type=str)
  parser.add_argument('--start_epoch', dest='start_epoch',
                      help='starting epoch',
                      default=1, type=int)
  parser.add_argument('--epochs', dest='max_epochs',
                      help='number of epochs to train',
                      default=8, type=int)
  parser.add_argument('--disp_interval', dest='disp_interval',
                      help='number of iterations to display',
                      default=100, type=int)
  parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                      help='number of iterations to display',
                      default=10000, type=int)

  parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save models', default="weights",
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

# config optimization
  parser.add_argument('--o', dest='optimizer',
                      help='training optimizer',
                      default="sgd", type=str)
  parser.add_argument('--lr', dest='lr',
                      help='starting learning rate',
                      default=0.00001, type=float)
  parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch',
                      default=1, type=int)
  parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=0.1, type=float)

# set training session
  parser.add_argument('--s', dest='session',
                      help='training session',
                      default=1, type=int)

# resume trained model
  parser.add_argument('--r', dest='resume',
                      help='resume checkpoint or not',
                      default=False, type=bool)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load model',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load model',
                      default=0, type=int)
# log and display
  parser.add_argument('--use_tfb', dest='use_tfboard',
                      help='whether use tensorboard',
                      default=True,
                      action='store_true')

  args = parser.parse_args()
  return args


class sampler(Sampler):
  def __init__(self, train_size, batch_size):
    self.num_data = train_size
    self.num_per_batch = int(train_size / batch_size)
    self.batch_size = batch_size
    self.range = torch.arange(0, batch_size).view(1, batch_size).long()
    self.leftover_flag = False
    if train_size % batch_size:
      self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
      self.leftover_flag = True

  def __iter__(self):
    rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
    self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

    self.rand_num_view = self.rand_num.view(-1)

    if self.leftover_flag:
      self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

    return iter(self.rand_num_view)

  def __len__(self):
    return self.num_data


if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)

  if args.dataset == "hico_mini":
      args.imdb_name = "hico2_mini_train"
      args.imdbval_name = "hico2_mini_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '75']
  elif args.dataset == "hico_full":
      args.imdb_name = "hico2_full_train"
      args.imdbval_name = "hico_full_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '75']
  else:
      print('Only support HICO-DET dataset now.')

  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)
  np.random.seed(cfg.RNG_SEED)

  #torch.backends.cudnn.benchmark = True
  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.
  cfg.TRAIN.USE_FLIPPED = True
  cfg.USE_GPU_NMS = args.cuda
  imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
  train_size = len(roidb)

  print('{:d} roidb entries'.format(len(roidb)))

  output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  sampler_batch = sampler(train_size, args.batch_size)

  dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                           imdb.num_classes, training=True)

  dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                            sampler=sampler_batch, num_workers=args.num_workers)

  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  dp_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_hois = torch.LongTensor(1)
  hboxes = torch.FloatTensor(1)
  oboxes = torch.FloatTensor(1)
  iboxes = torch.FloatTensor(1)
  pboxes = torch.FloatTensor(1)
  hoi_classes = torch.FloatTensor(1)
  vrb_classes = torch.FloatTensor(1)
  bin_classes = torch.FloatTensor(1)
  hoi_masks = torch.FloatTensor(1)
  vrb_masks = torch.FloatTensor(1)
  spa_maps = torch.FloatTensor(1)
  pose_maps = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda:
    im_data = im_data.cuda()
    dp_data = dp_data.cuda()
    im_info = im_info.cuda()
    num_hois = num_hois.cuda()
    hboxes = hboxes.cuda()
    oboxes = oboxes.cuda()
    iboxes = iboxes.cuda()
    pboxes = pboxes.cuda()
    hoi_classes = hoi_classes.cuda()
    vrb_classes = vrb_classes.cuda()
    bin_classes = bin_classes.cuda()
    hoi_masks = hoi_masks.cuda()
    vrb_masks = vrb_masks.cuda()
    spa_maps = spa_maps.cuda()
    pose_maps = pose_maps.cuda()

  # make variable
  im_data = Variable(im_data)
  dp_data = Variable(dp_data)
  im_info = Variable(im_info)
  num_hois = Variable(num_hois)
  hboxes = Variable(hboxes)
  oboxes = Variable(oboxes)
  iboxes = Variable(iboxes)
  pboxes = Variable(pboxes)
  hoi_classes = Variable(hoi_classes)
  vrb_classes = Variable(vrb_classes)
  bin_classes = Variable(bin_classes)
  hoi_masks = Variable(hoi_masks)
  vrb_masks = Variable(vrb_masks)
  spa_maps = Variable(spa_maps)
  pose_maps = Variable(pose_maps)

  if args.cuda:
    cfg.CUDA = True

  # initilize the network here.
  if args.net == 'vgg16':
    fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res101':
    fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
    fasterRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res152':
    fasterRCNN = resnet(imdb.classes, 152, pretrained=True, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()

  fasterRCNN.create_architecture()

  lr = cfg.TRAIN.LEARNING_RATE
  lr = args.lr
  #tr_momentum = cfg.TRAIN.MOMENTUM
  #tr_momentum = args.momentum

  params = []
  for key, value in dict(fasterRCNN.named_parameters()).items():
    if value.requires_grad:
      if 'bias' in key:
        params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
      else:
        params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

  if args.optimizer == "adam":
    lr = lr * 0.1
    optimizer = torch.optim.Adam(params)

  elif args.optimizer == "sgd":
    optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

  if args.cuda:
    fasterRCNN.cuda()

  if args.resume:
    load_name = os.path.join(output_dir,
      'ho_spa_rcnn3_lf_no_nis_vrb_sft_glb_part_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
    print("loading checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    args.session = checkpoint['session']
    args.start_epoch = checkpoint['epoch']
    fasterRCNN.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr = optimizer.param_groups[0]['lr']
    if 'pooling_mode' in checkpoint.keys():
      cfg.POOLING_MODE = checkpoint['pooling_mode']
    print("loaded checkpoint %s" % (load_name))

  if args.mGPUs:
    fasterRCNN = nn.DataParallel(fasterRCNN)

  iters_per_epoch = int(train_size / args.batch_size)

  if args.use_tfboard:
    if os.path.exists('logs'):
        import shutil
        shutil.rmtree('logs')

    from tensorboardX import SummaryWriter
    logger = SummaryWriter("logs")

  for epoch in range(args.start_epoch, args.max_epochs + 1):
    # setting to train mode
    fasterRCNN.train()
    loss_temp = 0
    loss_cls_temp = 0
    loss_bin_temp = 0
    start = time.time()

    ld_time = 0

    if epoch % (args.lr_decay_step + 1) == 0:
        adjust_learning_rate(optimizer, args.lr_decay_gamma)
        lr *= args.lr_decay_gamma

    data_iter = iter(dataloader)
    for step in range(iters_per_epoch):
      ld_start = time.time()
      data = next(data_iter)
      im_data.data.resize_(data[0].size()).copy_(data[0])
      dp_data.data.resize_(data[1].size()).copy_(data[1])
      im_info.data.resize_(data[2].size()).copy_(data[2])
      hboxes.data.resize_(data[3].size()).copy_(data[3])
      oboxes.data.resize_(data[4].size()).copy_(data[4])
      iboxes.data.resize_(data[5].size()).copy_(data[5])
      pboxes.data.resize_(data[6].size()).copy_(data[6])
      hoi_classes.resize_(data[7].size()).copy_(data[7])
      vrb_classes.resize_(data[8].size()).copy_(data[8])
      bin_classes.resize_(data[9].size()).copy_(data[9])
      hoi_masks.resize_(data[10].size()).copy_(data[10])
      vrb_masks.resize_(data[11].size()).copy_(data[11])
      spa_maps.data.resize_(data[12].size()).copy_(data[12])
      pose_maps.data.resize_(data[13].size()).copy_(data[13])
      num_hois.data.resize_(data[14].size()).copy_(data[14])
      ld_end = time.time()
      ld_time += (ld_end-ld_start)

      if num_hois.data.item() < 2:
          continue

      fasterRCNN.zero_grad()
      cls_prob, bin_prob, RCNN_loss_cls, RCNN_loss_bin = \
          fasterRCNN(im_data, dp_data, im_info,
                     hboxes, oboxes, iboxes, pboxes,
                     vrb_classes, bin_classes, vrb_masks,
                     spa_maps, pose_maps, num_hois)

      loss = RCNN_loss_cls.mean()

      if args.mGPUs:
          loss_cls = RCNN_loss_cls.mean().item()
          loss_bin = 0
      else:
          loss_cls = RCNN_loss_cls.item()
          loss_bin = 0

      loss_temp += loss.item()
      loss_bin_temp += loss_bin
      loss_cls_temp += loss_cls

      # backward
      optimizer.zero_grad()
      loss.backward()
      if args.net == "vgg16":
          clip_gradient(fasterRCNN, 10.)
      optimizer.step()

      if step % args.disp_interval == 0:
        end = time.time()
        if step > 0:
          loss_temp /= (args.disp_interval + 1)
          loss_bin_temp /= (args.disp_interval + 1)
          loss_cls_temp /= (args.disp_interval + 1)

        nNeg = torch.sum(bin_classes[:, :, 1]).item()
        nPos = bin_classes.shape[1] - nNeg

        print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                                % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
        print("loss_cls: %.4f, loss_bin: %.4f" % (loss_cls, loss_bin))
        print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (nPos, nNeg, end-start))

        if args.use_tfboard:
          info = {
            'loss': loss_temp,
            'loss_bin': loss_bin_temp,
            'loss_cls': loss_cls_temp
          }
          logger.add_scalars("logs_s_{}/losses".format(args.session), info, (epoch - 1) * iters_per_epoch + step)

        loss_temp = 0
        loss_cls_temp = 0
        loss_bin_temp = 0
        start = time.time() 
        ld_time = 0

    save_name = os.path.join(output_dir, 'ho_spa_rcnn3_lf_no_nis_vrb_sft_glb_part_{}_{}_{}.pth'.format(args.session, epoch, step))
    save_checkpoint({
      'session': args.session,
      'epoch': epoch + 1,
      'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
      'optimizer': optimizer.state_dict(),
      'pooling_mode': cfg.POOLING_MODE,
      'class_agnostic': args.class_agnostic,
    }, save_name)
    print('save model: {}'.format(save_name))

  if args.use_tfboard:
    logger.close()
